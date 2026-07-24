import argparse
import json
import os
from pathlib import Path
import pickle
import random
import re
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from os import listdir, makedirs, path
from time import time, sleep
from typing import Callable, List, Optional, Tuple, cast

import accelerate.utils as autils
import av
import comet_ml
import fsspec
import imageio
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from accelerate import Accelerator
from accelerate.tracking import CometMLTracker
from fsspec import AbstractFileSystem
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from tqdm.auto import tqdm
import torch.distributed.checkpoint
from torch.distributed.checkpoint.filesystem import _metadata_fn


def get_recammaster_cam_c2w(cam_c2w_path, num_frames=-1, num_cams=-1, relative_cam_index=None):

    def parse_matrix(matrix_str):  # Insane way to store camera info
        rows = matrix_str.strip().split("] [")
        matrix = []
        for row in rows:
            row = row.replace("[", "").replace("]", "")
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_cam_c2w(cam_c2w_all):
        cam_w2c_first = np.linalg.inv(cam_c2w_all[relative_cam_index, 0])[None, None]
        cam_c2w_all = cam_w2c_first @ cam_c2w_all
        return cam_c2w_all

    with open(cam_c2w_path, "r") as file:
        cam_data = json.load(file)

    num_frames = num_frames if num_frames > 0 else len(cam_data.keys())
    num_cams = num_cams if num_cams > 0 else len(cam_data[next(iter(cam_data.keys()))].keys())

    cam_c2w_all = []
    for i in range(num_cams):
        cam_c2w = [parse_matrix(cam_data[f"frame{j}"][f"cam{i + 1:02d}"]) for j in range(num_frames)]
        cam_c2w = np.stack(cam_c2w, axis=0).transpose(0, 2, 1)  # f 4 4
        cam_c2w = cam_c2w[:, :, [1, 2, 0, 3]]
        cam_c2w[:, :3, 1] *= -1.0
        cam_c2w[:, :3, 3] /= 100.0
        cam_c2w_all.append(cam_c2w)
    cam_c2w_all = np.stack(cam_c2w_all, axis=0)  # n f 4 4

    if relative_cam_index is not None:
        cam_c2w_all = get_relative_cam_c2w(cam_c2w_all)
    return cam_c2w_all


def load_video(
    video_path: str, grid_size: Optional[Tuple[int, int]] = None, indices: Optional[List[Tuple[int, int]]] = None,
):
    assert (grid_size is None) == (indices is None), "`grid_size` and `indices` must be either both given or both not."
    indexing = indices is not None

    container = av.open(video_path)
    fps = float(container.streams.video[0].average_rate)  # Get average frame rate from video stream

    frames = [[] for _ in indices] if indexing else []
    for frame in container.decode(video=0):
        frame = frame.to_image()  # Convert to PIL Image
        if indexing:
            width, height = frame.size  # i indexes width, j indexes height
            assert width % grid_size[0] == 0 and height % grid_size[1] == 0,\
                "Frame width and height must be multiple of `grid_size` width and height"
            grid_width, grid_height = width // grid_size[0], height // grid_size[1]
            for k, (i, j) in enumerate(indices):
                frame_patch = frame.crop((i * grid_width, j * grid_height, (i + 1) * grid_width, (j + 1) * grid_height))
                assert frame_patch.size == (grid_width, grid_height),\
                    f"Cropped frame patch {frame_patch.size} must match grid width and height."
                frames[k].append(frame_patch)
        else:
            frames.append(frame)
    container.close()

    return frames, fps


def save_video(output_path, video, fps, quality=None, imageio_params=None):
    imageio_params = imageio_params if imageio_params is not None else {}
    if quality is not None:
        imageio_params["quality"] = quality
    if path.splitext(output_path)[1] == ".gif":
        imageio_params["loop"] = 0

    writer = imageio.get_writer(output_path, fps=fps, **imageio_params)

    for i in range(video.shape[0]):
        writer.append_data(video[i])
    writer.close()


def depths_to_normals(depths, mask, intrinsics=None, to_uint8=False):
    num_frames, height, width = depths.shape

    depths_masked = np.copy(depths)  # Create a masked version of depth where invalid areas are NaN
    depths_masked[~mask] = np.nan

    dy, dx = np.gradient(depths_masked, axis=(1, 2))  # Compute gradient
    normals = np.zeros((num_frames, height, width, 2), dtype=np.float32)  # Create normal vectors
    normals[:, :, :, 0] = -dx
    normals[:, :, :, 1] = -dy

    if intrinsics is not None:
        # Adjust for perspective projection using intrinsics
        y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")  # Create pixel coordinate grids
        x_grid = np.tile(x_grid[None, :, :], (num_frames, 1, 1))  # Expand dimensions to match the frames
        y_grid = np.tile(y_grid[None, :, :], (num_frames, 1, 1))

        fx = intrinsics[:, 0][:, None, None]  # Extract intrinsics, shape: f 1 1
        fy = intrinsics[:, 1][:, None, None]
        cx = intrinsics[:, 2][:, None, None]
        cy = intrinsics[:, 3][:, None, None]
        ndc_x = (x_grid - cx) / fx  # Compute normalized device coordinates
        ndc_y = (y_grid - cy) / fy

        normals[:, :, :, 0] += ndc_x  # Adjust normal vectors for perspective projection
        normals[:, :, :, 1] += ndc_y

    norms = np.sqrt(np.sum(normals ** 2, axis=3, keepdims=True))  # Normalize vectors to unit length
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals = normals / norms

    normals[(~mask[..., None]) | np.isnan(normals)] = 0.0  # Apply mask again and eliminate NaN from the beginning
    if to_uint8:
        normals = (np.clip((normals + 1.0) * 0.5, 0.0, 1.0) * 255).astype(np.uint8)
    return normals  # f h w 2


def depths_to_video(depths, mask, include_normals=False, use_intrinsics=True, intrinsics=None, to_uint8=False):
    depths = depths.astype(np.float32)
    depths[~mask] = 1.0
    disparity = 1 / depths
    disparity[~mask] = 0.0
    disparity_min, disparity_max = disparity[mask].min(), disparity[mask].max()
    disparity = (disparity - disparity_min) / (disparity_max - disparity_min)  # Normalize to [0, 1]
    if to_uint8:
        disparity = (np.clip(disparity, 0, 1) * 255).astype(np.uint8)
    if include_normals:
        if use_intrinsics:
            assert intrinsics is not None
        else:
            intrinsics = None
        normals = depths_to_normals(depths, mask, intrinsics, to_uint8=to_uint8)
        disparity = np.concatenate((normals, disparity[..., None]), axis=-1)
    return disparity


def bernoulli(p):
    return random.random() < p


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image

    def load_data(self, file_path):
        return self.load_image(file_path)

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path_full = path.join(self.base_path, data[key])
                data[key] = self.load_data(path_full)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data

    def __len__(self):
        return len(self.data) * self.repeat


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81, frame_interval=1,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=3840 * 2160, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        skip_shorter_than_num_frames=True,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            frame_interval = args.frame_interval
            skip_shorter_than_num_frames = args.skip_shorter_than_num_frames
            data_file_keys = args.data_file_keys.split(",")
            data_file_keys = list(dict.fromkeys(data_file_keys)) # deduplicate with order-preserving
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.skip_shorter_than_num_frames = skip_shorter_than_num_frames
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        frame_interval = self.frame_interval
        num_global_frames = (num_frames - 1) * frame_interval + 1
        if int(reader.count_frames()) < num_global_frames:
            num_frames = int((int(reader.count_frames()) - 1) // frame_interval + 1)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        if self.skip_shorter_than_num_frames and num_frames < self.num_frames:
            return None
        frame_interval = self.frame_interval
        num_global_frames = (num_frames - 1) * frame_interval + 1
        frames = []
        for frame_id in range(num_global_frames):
            frame = reader.get_data(frame_id)
            if frame_id % frame_interval == 0:
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame, *self.get_height_width(frame))
                frames.append(frame)
        reader.close()
        return frames

    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension

    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension

    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                relative_path = data[key]
                path_full = path.join(self.base_path, relative_path)
                try:
                    data[key] = self.load_data(path_full)
                except Exception as e:
                    print(f"from diffsynth_studio.diffsynth.trainers.utils: failed to load key = {key}, path = {path_full}. Error: {e}")
                    return None

                if data[key] is None:
                    # original code
                    # warnings.warn(f"Cannot load file {data[key]}.")
                    print(f"from diffsynth_studio.diffsynth.trainers.utils: cannot load key = {key} because data[key] = {data[key]}. The path is {path_full}, Maybe this is because the number of frames is not enough")
                    return None # in train.py when pre-computing, we will skip this sample
                
                if key == "video":  # TODO: Make more flexible, maybe?
                    data["video_file_path"] = relative_path
        return data

    def __len__(self):
        return len(self.data) * self.repeat


class WarpedVideoDataset(torch.utils.data.Dataset):
    # TODO: The dataset should behave slightly differently depending on if we are preprocessing or not ...
    #       maybe try a preprocess flag?
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81, frame_interval=1,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        repeat=1,
        debug_path=None,
        config_path=None,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            frame_interval = args.frame_interval
            repeat = args.dataset_repeat
            debug_path = args.debug_path
            config = args.config_path

        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.repeat = repeat
        self.debug_path = debug_path

        self.config = {}
        if config_path is not None:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)["dataset"]

        self.use_dynamic_mask = False

        # TODO: Support dynamic resolution
        assert metadata_path is not None, "Metadata must be provided for WarpedVideoDataset"
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:  # Assume csv
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def load_data(self, file_path):
        if file_path.endswith(".pth"):
            data = torch.load(file_path, map_location="cpu", weights_only=True)
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
        else:
            return None
        return data

    def load_video_and_data(self, video_path, warped_video_path, warped_video_info_path, source_index, target_index):
        try:
            warped_video_info = np.load(path.join(self.base_path, warped_video_info_path))

            cam_c2w_multiview = warped_video_info["cam_c2w"]
            intrinsics_multiview = warped_video_info["intrinsics"]
            mask_multiview = warped_video_info["mask"]

            _, num_views, num_frames, height, width = mask_multiview.shape
            assert (height, width) == mask_multiview.shape[3:5] == (self.height, self.width)

            [source_video, target_video], fps = load_video(
                path.join(self.base_path, video_path),
                grid_size=(num_views, 1),
                indices=[(source_index, 0), (target_index, 0)],
            )
            [warped_source_video, warped_target_video], _ = load_video(
                path.join(self.base_path, warped_video_path),
                grid_size=(num_views, num_views),
                indices=[(source_index, source_index), (source_index, target_index)],
            )
            assert (width, height) ==\
                source_video[0].size == target_video[0].size ==\
                warped_source_video[0].size == warped_target_video[0].size

            # Mask has three entries for the final dimension: [ color_mask geometry_mask motion_mask ]
            # We will return them separately here
            source_color_mask = np.ones((num_frames, height, width), dtype=np.bool_)
            source_geometry_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
            warped_target_color_mask = mask_multiview[source_index][target_index]
            warped_target_geometry_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
            if self.use_dynamic_mask and "dynamic_mask" in warped_video_info:
                raise NotImplementedError("Currently using motion masks is not implemented!")
                dynamic_mask_multiview = warped_video_info["dynamic_mask"]
            else:
                source_motion_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
                warped_target_motion_mask = np.zeros((num_frames, height, width), dtype=np.bool_)

            video_and_data = {
                # Videos
                "source_video": source_video,
                "target_video": target_video,
                "warped_source_video": warped_source_video,
                "warped_target_video": warped_target_video,
                # Cameras
                "source_cam_c2w": cam_c2w_multiview[source_index],
                "target_cam_c2w": cam_c2w_multiview[target_index],
                "source_intrinsics": intrinsics_multiview[source_index],
                "target_intrinsics": intrinsics_multiview[target_index],
                # Masks
                "source_color_mask": source_color_mask,
                "source_geometry_mask": source_geometry_mask,
                "source_motion_mask": source_motion_mask,
                "warped_target_color_mask": warped_target_color_mask,
                "warped_target_geometry_mask": warped_target_geometry_mask,
                "warped_target_motion_mask": warped_target_motion_mask,
                # Misc
                "fps": fps,
            }
            if (
                "static_mask" in warped_video_info and
                "depths" in warped_video_info and "depths_mask" in warped_video_info
            ):
                video_and_data["target_depths"] = warped_video_info["depths"][target_index]
                video_and_data["target_depths_mask"] = warped_video_info["depths_mask"][target_index]
                video_and_data["target_static_mask"] = warped_video_info["static_mask"][target_index]
            return video_and_data

        except Exception as e:
            exception_message = (
                f"Unable to process data with video_path=`{video_path}`, source=`{source_index}`, "
                f"target=`{target_index}`. Exception: {e}\n{traceback.format_exc()}\n"
            )
            if hasattr(self, "accelerator"):
                exception_message = f"[{self.accelerator.local_process_index}] {exception_message}"
            tqdm.write(exception_message)
            return None

    def bernoulli(self, config_key, default):
        return bernoulli(self.config.get(config_key, default))

    def align_cam_c2w(self, source_cam_c2w, target_cam_c2w):
        # By default, align to source (a.k.a. source t=0 cam_c2w is identity)
        align_w2c = np.linalg.inv(source_cam_c2w[0])[None]
        source_cam_c2w = align_w2c @ source_cam_c2w
        target_cam_c2w = align_w2c @ target_cam_c2w
        return source_cam_c2w, target_cam_c2w

    def preprocess_and_augment_data(self, video_and_data, reverse_time=False, use_geometry=False):
        num_frames, height, width = video_and_data["source_color_mask"].shape
        global_num_frames = (self.num_frames - 1) * self.frame_interval + 1
        assert self.num_frames >= global_num_frames

        start_index = random.randint(0, num_frames - global_num_frames)
        reverse_index = -1 if reverse_time else 1
        for key in video_and_data.keys():
            if not isinstance(video_and_data[key], (list, tuple, np.ndarray, torch.Tensor)):
                continue
            video_and_data[key] =\
                video_and_data[key][start_index:][::self.frame_interval][:self.num_frames][::reverse_index]
                                  # ^splice start ^skip frame interval   ^cut at num frames  ^optionally reverse

        video_and_data["source_cam_c2w"], video_and_data["target_cam_c2w"] =\
            self.align_cam_c2w(video_and_data["source_cam_c2w"], video_and_data["target_cam_c2w"])

        # TODO: This is very jank, need to fix via better data preprocessing!
        # Namely, we shouldn't be using mask from (ground truth!) target video, we should be warping the mask
        # *in addition* to warping the video and using that instead
        if "target_depths" in video_and_data and use_geometry:
            depths = video_and_data["target_depths"]
            depths_mask = video_and_data["target_depths_mask"]
            static_mask = video_and_data["target_static_mask"]
            color_mask = video_and_data["warped_target_color_mask"]
            warped_target_video = np.stack([np.array(frame) for frame in video_and_data["warped_target_video"]], axis=0)

            depths_video = depths_to_video(
                depths, depths_mask, to_uint8=True,
                include_normals=True, use_intrinsics=False, intrinsics=video_and_data["target_intrinsics"],
            )
            fill_mask = depths_mask & static_mask & ~color_mask  # All valid & static regions which do *not* have color
            warped_target_video = warped_target_video * ~fill_mask[..., None] + depths_video * fill_mask[..., None]
            video_and_data["warped_target_video"] =\
                [Image.fromarray(warped_target_video[i]) for i in range(self.num_frames)]
            video_and_data["warped_target_geometry_mask"] = fill_mask
            video_and_data["depths_video"] = depths_video

    def visualize(self, video_and_data, debug_once=True):
        if self.debug_path is None:
            return

        def pil_frames_to_np(pil_frames):
            np_frames = np.stack([np.array(frame) for frame in pil_frames], axis=0)
            return np_frames

        source_video = pil_frames_to_np(video_and_data["source_video"])
        warped_target_video = pil_frames_to_np(video_and_data["warped_target_video"])
        target_video = pil_frames_to_np(video_and_data["target_video"])

        warped_target_masks = np.stack(                       # R             G                B
            [video_and_data[f"warped_target_{key}"] for key in ("color_mask", "geometry_mask", "motion_mask")], axis=-1,
        )
        warped_target_masks = warped_target_masks.astype(np.uint8) * 255

        depths_video = video_and_data["depths_video"] if "depths_video" in video_and_data\
            else np.zeros_like(source_video)

        # Horizontally concatenate
        vis_video = np.concatenate(
            (source_video, depths_video, warped_target_video, warped_target_masks, target_video), axis=2,
        )
        vis_name = "vis_WarpedVideoDataset"
        if hasattr(self, "accelerator"):
            vis_name = f"{vis_name}_rank{self.accelerator.local_process_index}"
        makedirs(self.debug_path, exist_ok=True)
        save_video(path.join(self.debug_path, f"{vis_name}.mp4"), vis_video, fps=video_and_data["fps"], quality=8)
        save_video(path.join(self.debug_path, f"{vis_name}.gif"), vis_video, fps=video_and_data["fps"], quality=6)

        if debug_once:  # Only debug once
            self.debug_path = None

    def __getitem__(self, data_id):
        video_and_data = None
        for _ in range(16):  # Retry 8 times
            data = self.data[data_id % len(self.data)].copy()
            video_and_data = self.load_video_and_data(
                video_path=data["video"],
                warped_video_path=data["warped_video"],
                warped_video_info_path=data["warped_video_info"],
                source_index=data["source_index"],
                target_index=data["target_index"],
            )
            if video_and_data is None:  # Try again with another random element!
                data_id = random.randint(0, len(self) - 1)
            else:
                break
        if video_and_data is None:
            return None

        reverse_time = self.bernoulli("reverse_time", 0.5)
        use_geometry = self.bernoulli("use_geometry", 0.5) and data["source_index"] != data["target_index"]
        self.preprocess_and_augment_data(video_and_data, reverse_time=reverse_time, use_geometry=use_geometry)

        self.visualize(video_and_data, debug_once=True)

        data_info = {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "frame_interval": 1,  # Frame interval has already been applied
            "fps": video_and_data["fps"],
        }
        output_data = {
            # Training target/label
            "target_video": video_and_data["target_video"],
            # Training inputs
            "source_video": video_and_data["source_video"],
            "warped_target_video": video_and_data["warped_target_video"],
            "target_cam_c2w": video_and_data["target_cam_c2w"],
            "target_intrinsics": video_and_data["target_intrinsics"],
            "source_color_mask": video_and_data["source_color_mask"],
            "source_geometry_mask": video_and_data["source_geometry_mask"],
            "source_motion_mask": video_and_data["source_motion_mask"],
            "warped_target_color_mask": video_and_data["warped_target_color_mask"],
            "warped_target_geometry_mask": video_and_data["warped_target_geometry_mask"],
            "warped_target_motion_mask": video_and_data["warped_target_motion_mask"],
            "prompt": data["prompt"],
            # Data augmentation (already handled by dataset)
            "is_reverse_time": reverse_time,
            "is_use_geometry": use_geometry,
            # Data augmentation (to be done by model pipeline)
            "drop_source": self.bernoulli("drop_source", 0.0),
            "drop_warped_target": self.bernoulli("drop_warped_target", 0.0),
            "drop_prompt": self.bernoulli("drop_prompt", 0.0),
            # Shape and frames
            "info": data_info,
        }
        return output_data

    def __len__(self):
        return len(self.data) * self.repeat

    @staticmethod
    def collate(data):
        assert len(data) > 0, "Cannot collate data of batch size 0."
        if any([d is None for d in data]):
            return None
        # assert all([list(d.keys()) == list(data[0].keys()) for d in data]),\
        #     f"All data keys must match, got:\n    {'\n    '.join([str(list(d.keys())) for d in data])}"
        assert all([list(d.keys()) == list(data[0].keys()) for d in data]), \
            "All data keys must match, got:\n    " + "\n    ".join([str(list(d.keys())) for d in data])
        data_collated = {}
        for key in data[0].keys():
            value = [d[key] for d in data]
            # Always stack tensors
            if isinstance(value[0], torch.Tensor):
                data_collated[key] = torch.stack(value, dim=0)
            elif isinstance(value[0], np.ndarray):
                data_collated[key] = np.stack(value, axis=0)
            elif key == "info":
                # assert all([v == value[0] for v in value]),\
                #     f"All `info` within a batch must match, got:\n    {'\n    '.join([str(v) for v in value])}"
                assert all([v == value[0] for v in value]), \
                    "All `info` within a batch must match, got:\n    " + "\n    ".join([str(v) for v in value])
                data_collated[key] = value[0]
            else:
                data_collated[key] = value
        return data_collated


class TensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81, frame_interval=1,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video_latents", "prompt_emb"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            frame_interval = args.frame_interval
            data_file_keys = args.data_file_keys.split(",")
            data_file_keys = list(dict.fromkeys(data_file_keys)) # deduplicate with order-preserving
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.repeat = repeat

        # TODO: Support dynamic resolution

        assert metadata_path is not None, "Metadata must be provided for TensorDataset"
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:  # Assume csv
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        self.data_info = {
            "num_frames": args.num_frames,
            "frame_interval": args.frame_interval,
            "height": args.height,
            "width": args.width,
        }

    def load_data(self, file_path):
        data = torch.load(file_path, map_location="cpu", weights_only=True)
        return data

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path_full = path.join(self.base_path, data[key])
                data[key] = self.load_data(path_full)
                if data[key] is None:
                    # warnings.warn(f"cannot load file {data[key]}.")
                    print(f"from diffsynth_studio.diffsynth.trainers.utils: cannot load key = {key} because data[key] = {data[key]}.")
                    return None
        data["info"] = deepcopy(self.data_info)
        return data

    def __len__(self):
        return len(self.data) * self.repeat


class ReCamMasterTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81, frame_interval=1,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920 * 1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("source_latents", "target_latents", "cameras", "prompt_emb"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            frame_interval = args.frame_interval
            data_file_keys = args.data_file_keys.split(",")
            data_file_keys = list(dict.fromkeys(data_file_keys)) # deduplicate with order-preserving
            repeat = args.dataset_repeat

        self.base_path = base_path
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.repeat = repeat

        # TODO: Support dynamic resolution

        assert metadata_path is not None, "Metadata must be provided for TensorDataset"
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:  # Assume csv
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        self.data_info = {
            "num_frames": args.num_frames,
            "frame_interval": args.frame_interval,
            "height": args.height,
            "width": args.width,
        }

    def load_data(self, file_path):
        if file_path.endswith(".pth"):
            data = torch.load(file_path, map_location="cpu", weights_only=True)
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
        else:
            return None
        return data

    def load_cameras(self, cameras_path, source_index, target_index):
        cam_c2w = get_recammaster_cam_c2w(cameras_path, relative_cam_index=source_index)
        return cam_c2w[target_index]

    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path_full = path.join(self.base_path, data[key])
                if key == "cameras":
                    data[key] = self.load_cameras(path_full, data["source_index"], data["target_index"])
                else:
                    data[key] = self.load_data(path_full)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        data["info"] = deepcopy(self.data_info)
        return data

    def __len__(self):
        return len(self.data) * self.repeat

    @staticmethod
    def collate(data):
        assert len(data) > 0, "Cannot collate data of batch size 0."
        # assert all([list(d.keys()) == list(data[0].keys()) for d in data]),\
        #     f"All data keys must match, got:\n    {'\n    '.join([str(list(d.keys())) for d in data])}"
        assert all([list(d.keys()) == list(data[0].keys()) for d in data]), \
            "All data keys must match, got:\n    " + "\n    ".join([str(list(d.keys())) for d in data])
        data_collated = {}
        for key in data[0].keys():
            value = [d[key] for d in data]
            # Always stack tensors
            if isinstance(value[0], torch.Tensor):
                data_collated[key] = torch.stack(value, dim=0)
            elif isinstance(value[0], np.ndarray):
                data_collated[key] = np.stack(value, axis=0)
            elif key == "info":
                # assert all([v == value[0] for v in value]),\
                #     f"All `info` within a batch must match, got:\n    {'\n    '.join([str(v) for v in value])}"
                assert all([v == value[0] for v in value]), \
                    "All `info` within a batch must match, got:\n    " + "\n    ".join([str(v) for v in value])

                data_collated[key] = value[0]
            else:
                data_collated[key] = value
        return data_collated


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names

    def trainable_named_modules(self):
        trainable_named_modules = {name: param for name, param in self.named_parameters() if param.requires_grad}
        return trainable_named_modules

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

def finish_no_rename(self, metadata, results) -> None:
    """
    Monkeypatch method for the _FileSystemWriter.finish() method used in torch distributed checkpointing, needed to replace
    the use of the rename operation with a copy-and-delete to enable support for S3 mountpoint saving.

    This is a very brittle and un-ideal approach, perhaps a better future approach would be to override the file system writer
    class with an S3-compatible version and drop that in.
    """
    storage_md = {}
    for wr_list in results:
        storage_md.update({wr.index: wr.storage_data for wr in wr_list})
    metadata.storage_data = storage_md

    metadata.storage_meta = self.storage_meta()

    tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
    with self.fs.create_stream(tmp_path, "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
        if self.sync_files:
            try:
                os.fsync(metadata_file.fileno())
            except AttributeError:
                os.sync()

    # delete in-case other checkpoints were present.
    if self.fs.exists(self.metadata_path):
        self.fs.rm_file(self.metadata_path)

    # S3 Mount doesn't support rename
    # self.fs.rename(tmp_path, self.metadata_path)
    fs: AbstractFileSystem
    fs, _ = fsspec.core.url_to_fs(str(self.metadata_path))
    fs.put(str(tmp_path), str(self.metadata_path))
    fs.rm(str(tmp_path))

class ModelLogger:
    def __init__(
        self,
        output_path,
        run_name,
        log_with_wandb=False,
        log_loss_every_n_steps=-1,
        log_model_every_n_steps=-1,
        info_to_save=None,
        state_dict_converter=lambda x: x,
        ckpt_s3_path: Optional[str] = None,
        async_upload: bool = False,
        max_workers: int= 1,
        s3_upload_max_retries: int = 5,
        s3_upload_retry_delay: int = 30,
        local_save_max_retries: int = 5,
        local_save_retry_delay: int = 30,
    ):
        self.output_path = output_path
        self.run_name = run_name
        self.state_dict_converter = state_dict_converter
        self.log_with_wandb = log_with_wandb
        self.log_loss_every_n_steps = log_loss_every_n_steps
        self.log_model_every_n_steps = log_model_every_n_steps
        self.info_to_save = info_to_save
        self.max_steps = -1

        # Retry configuration
        self.s3_upload_max_retries = s3_upload_max_retries
        self.s3_upload_retry_delay = s3_upload_retry_delay
        self.local_save_max_retries = local_save_max_retries
        self.local_save_retry_delay = local_save_retry_delay

        self.s3_save_path = ckpt_s3_path
        if self.s3_save_path is not None:
            self.fs: AbstractFileSystem = fsspec.filesystem("s3")
            self.executor: ThreadPoolExecutor | None = None
            if async_upload:
                self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def log_model(self, save_name: str, accelerator: Accelerator, model):
        path_full = path.join(self.output_path, self.run_name, save_name)
        if not path_full.endswith("/"):
            path_full = path_full + "/"
        if accelerator.is_main_process:
            tqdm.write(f"Logging model to: {path_full} ...")
            tqdm.write(f"absolute path: {path.abspath(path_full)}")
            t = time()

        # Setup for local save (outside retry loop for distributed consistency)
        makedirs(path_full, exist_ok=True)

        # Pre-create FSDP subdirectories to prevent race conditions on S3 mountpoint
        if getattr(accelerator.state, "fsdp_plugin", None):
            # Count models and optimizers from accelerator
            num_models = len(accelerator._models) if hasattr(accelerator, '_models') else 1
            num_optimizers = len(accelerator._optimizers) if hasattr(accelerator, '_optimizers') else 1

            # All ranks create directories so each rank's FUSE cache sees them
            for i in range(num_models):
                model_dir = path.join(path_full, f"pytorch_model_fsdp_{i}")
                makedirs(model_dir, exist_ok=True)
            for i in range(num_optimizers):
                optimizer_dir = path.join(path_full, f"optimizer_{i}")
                makedirs(optimizer_dir, exist_ok=True)
            if accelerator.is_main_process:
                tqdm.write(f"[ModelLogger] Pre-created FSDP subdirectories for {num_models} model(s) and {num_optimizers} optimizer(s)")

            # Critical: Wait for directory creation to propagate across S3 mountpoint
            accelerator.wait_for_everyone()

            # Additional safety: Small sleep to ensure S3 mountpoint consistency
            if accelerator.is_main_process:
                tqdm.write("[ModelLogger] Waiting for directory propagation on S3 mountpoint...")
            sleep(5)  # Allow time for S3 mountpoint FUSE propagation
            accelerator.wait_for_everyone()

        torch.cuda.empty_cache()  # Prevent NCCL OOM error
        # Monkeypatch the finish method to use copy-and-delete instead of rename
        torch.distributed.checkpoint.filesystem._FileSystemWriter.finish = finish_no_rename  # type: ignore
        accelerator.wait_for_everyone()

        # Local save with retry logic (coordinated across all ranks)
        save_success = False
        for attempt in range(1, self.local_save_max_retries + 1):
            # Re-create FSDP directories on retry to refresh FUSE cache visibility
            if attempt > 1 and getattr(accelerator.state, "fsdp_plugin", None):
                for i in range(num_models):
                    makedirs(path.join(path_full, f"pytorch_model_fsdp_{i}"), exist_ok=True)
                for i in range(num_optimizers):
                    makedirs(path.join(path_full, f"optimizer_{i}"), exist_ok=True)
                accelerator.wait_for_everyone()
                sleep(5)
                accelerator.wait_for_everyone()

            local_success = False
            try:
                accelerator.save_state(path_full, safe_serialization=True)
                local_success = True
                if accelerator.is_main_process:
                    tqdm.write(f"[ModelLogger] Local save successful on this rank (attempt {attempt}/{self.local_save_max_retries})")
            except BaseException as e:
                # Re-raise interrupts that should terminate immediately
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                if accelerator.is_main_process:
                    tqdm.write(f"[ModelLogger] Local save failed on this rank (attempt {attempt}/{self.local_save_max_retries}): {type(e).__name__}: {e}")

            # Coordinate across ranks: all ranks must agree on success before proceeding
            # This ensures all ranks retry together, avoiding desync in collective operations
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Use all_reduce to check if ALL ranks succeeded
                success_tensor = torch.tensor([int(local_success)], dtype=torch.int32, device=accelerator.device)
                torch.distributed.all_reduce(success_tensor, op=torch.distributed.ReduceOp.MIN)  # MIN: 1 only if all are 1
                all_ranks_succeeded = bool(success_tensor.item())
            else:
                # Single process: just use local result
                all_ranks_succeeded = local_success

            if all_ranks_succeeded:
                save_success = True
                if accelerator.is_main_process:
                    tqdm.write(f"[ModelLogger] All ranks saved successfully (attempt {attempt}/{self.local_save_max_retries})")
                break
            elif attempt < self.local_save_max_retries:
                # At least one rank failed - all ranks sleep and retry together
                if accelerator.is_main_process:
                    tqdm.write(f"[ModelLogger] At least one rank failed - all ranks retrying after {self.local_save_retry_delay}s...")
                sleep(self.local_save_retry_delay)
            else:
                if accelerator.is_main_process:
                    tqdm.write(f"[ModelLogger] Save failed after {self.local_save_max_retries} attempts on at least one rank. Skipping S3 upload.")

        # Post-save synchronization and cleanup (outside retry loop for distributed consistency)
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            tqdm.write(f"Model logging time: {time() - t:.5f}s")
            # Save info of current training run!
            if save_success:
                info_path = path.join(self.output_path, self.run_name, "info.yaml")
                if self.info_to_save is not None and not path.isfile(info_path):
                    with open(info_path, "w") as file:
                        yaml.dump(self.info_to_save, file, indent=4, default_flow_style=False, sort_keys=False)

        # Only upload if local save succeeded
        if save_success and self.s3_save_path is not None:
            self._submit_upload_model(path_full)

    def _submit_upload_model(self, save_path: str):
        if not self.s3_save_path:
            print("No S3 save path provided, skipping model upload")
            return
        if self.executor is not None:
            print(f"Asynchronously uploading model to S3: {save_path}")
            self.executor.submit(self._upload_model, save_path)
        else:
            print(f"Synchronously uploading model to S3: {save_path}")
            self._upload_model(save_path)

    def _upload_model(self, save_path: str):
        """
        Upload local checkpoint to S3 with retry logic.

        Assumes the checkpoint is in a directory that contains files only for that particular checkpoint, and will upload to S3 under a directory of the same name.

        Example:
        save_path = "/path/to/checkpoint/step-100/step-100.safetensors"
        ckpt_dir_path = "/path/to/checkpoint/step-100"
        ckpt_dirname = "step-100"
        remote_upload_path = "<remote>/path/to/checkpoint/step-100"
        """
        start_time = time()
        assert self.s3_save_path is not None, "No S3 save path provided"
        ckpt_dir_path = os.path.abspath(os.path.dirname(save_path))
        ckpt_dirname = ckpt_dir_path.split("/")[-1]
        s3_upload_path = os.path.join(self.s3_save_path, ckpt_dirname)

        upload_success = False
        for attempt in range(1, self.s3_upload_max_retries + 1):
            try:
                print(f"[ModelLogger] Uploading {ckpt_dirname} to S3 (attempt {attempt}/{self.s3_upload_max_retries})")
                print(f"[ModelLogger] Source: {ckpt_dir_path}")
                print(f"[ModelLogger] Destination: {s3_upload_path}")

                # Perform S3 upload
                self.fs.put(ckpt_dir_path, s3_upload_path, recursive=True)

                elapsed = time() - start_time
                print(f"[ModelLogger] S3 upload successful in {elapsed:.2f}s (attempt {attempt}/{self.s3_upload_max_retries})")
                upload_success = True
                break

            except Exception as e:
                print(f"[ModelLogger] S3 upload failed (attempt {attempt}/{self.s3_upload_max_retries}): {type(e).__name__}: {e}")
                if attempt < self.s3_upload_max_retries:
                    print(f"[ModelLogger] Waiting {self.s3_upload_retry_delay}s before retry...")
                    sleep(self.s3_upload_retry_delay)
                else:
                    print(f"[ModelLogger] S3 upload failed after {self.s3_upload_max_retries} attempts: {s3_upload_path}")

        return upload_success

    def log_loss(self, loss, step_id, epoch_id):
        self.max_steps = max(self.max_steps, step_id)
        if self.log_with_wandb:
            import wandb
            wandb.log(
                {"Training loss": loss.item(), "Step": step_id, "Epoch": epoch_id},
                step=step_id + epoch_id * self.max_steps,
            )

    @torch.no_grad()
    def on_step_end(self, loss, accelerator, model, step_id, epoch_id):
        loss_synced = accelerator.gather(loss).mean()
        if self.log_loss_every_n_steps < 1 and self.log_model_every_n_steps < 1:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if self.log_loss_every_n_steps > 0 and (step_id + 1) % self.log_loss_every_n_steps == 0:
                self.log_loss(loss_synced, step_id, epoch_id)
        if self.log_model_every_n_steps > 0 and (step_id + 1) % self.log_model_every_n_steps == 0:
            self.log_model(f"epoch={epoch_id}_step={step_id + 1}", accelerator, model)
        accelerator.wait_for_everyone()
        return loss_synced

    @torch.no_grad()
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        self.log_model(f"epoch={epoch_id}_step=end", accelerator, model)
        accelerator.wait_for_everyone()


def recursive_print_dict(d, indent = 0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("    " * indent, f"{k}:")
            recursive_print_dict(v, indent+1)
        else:
            print("    " * indent, f"{k}: {v}")


def get_comet_tracker(api_key: str, project_name: str, experiment_name: str) -> CometMLTracker:
    return CometMLTracker(
        run_name=project_name,
        api_key=api_key,
        experiment_config=comet_ml.ExperimentConfig(name=experiment_name),
    )


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_size: int = -1,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    shuffle_dataset: bool = True,
    use_preprocessed_data: bool = False,
    resume_from_state: str = None,
    run_name: str = "training",
    log_with_wandb: bool = False,
    wandb_init_fn: Callable | None = None,
    profiler_init_fn: Callable = None,
    comet_api_key: str | None = None,
    comet_project_name: str | None = None,
    comet_experiment_name: str | None = None,
):
    if batch_size > 0:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle_dataset, collate_fn=dataset.collate,
        )
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=shuffle_dataset, collate_fn=lambda x: x[0])

    # Setup Comet
    comet_tracker: CometMLTracker | None = None
    if comet_api_key is not None:
        assert wandb_init_fn is None, "Cannot use Comet and WandB together."
        comet_tracker = get_comet_tracker(comet_api_key, comet_project_name, comet_experiment_name)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with=comet_tracker)
    if comet_tracker is not None:
        accelerator.init_trackers(comet_tracker.run_name)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    dataloader.dataset.accelerator = accelerator  # For debugging visualization

    if resume_from_state is not None:
        accelerator.load_state(resume_from_state)
        print(f"Loaded training state from: {resume_from_state}")
        step_match = re.search(r"step=(\d+)", resume_from_state)
        steps_to_skip = 0 if step_match is None else int(step_match.group(1))
        print(f"Skipping {steps_to_skip} batches to restore training state")
        accelerator.skip_first_batches(dataloader, steps_to_skip)

    if wandb_init_fn is not None:
        wandb_init_fn(accelerator, model)

    profiler = None
    if profiler_init_fn is not None:
        profiler = profiler_init_fn(accelerator)

    # When using FSDP, we need to manually set the pipeline's device and dtype for intermediate variables
    if getattr(accelerator.state, "fsdp_plugin", None):
        model.pipe.device = accelerator.device
        model.pipe.torch_dtype =\
            {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[accelerator.mixed_precision]
        # We DON'T want to do: model.pipe.to(dtype=some_dtype, device=accelerator.device)
        # This will also set all the model weights to be CUDA and bfloat16, but for FSDP w/ CPU offloading we want to
        # keep them on CPU by default (and FSDP moves and casts them as needed).

    if accelerator.is_main_process:
        num_trainable_params = 0
        print("\nTrainable parameters:")
        for name, param in model.trainable_named_modules().items():
            name_unwrapped = name.replace("._checkpoint_wrapped_module", "")
            print(f"    {name_unwrapped}: {tuple(param.shape)}")
            num_trainable_params += np.prod(param.shape)
        print(f"Number of trainable parameters: {num_trainable_params:_}")

        # print("\nWorld Rerender config:")
        # recursive_print_dict(model.world_rerender_config, indent=1)
        # print()
    accelerator.wait_for_everyone()

    if not use_preprocessed_data:
        model.accelerator = accelerator

    if profiler is not None:
        profiler.start()

    global_step = 0
    for epoch_id in range(num_epochs):
        tqdm_prefix = (
            f"[{accelerator.process_index}] run=`{run_name}`, preprocess={use_preprocessed_data}, "
            f"per_gpu_bs={batch_size}, epoch={epoch_id}"
        )
        progress = tqdm(enumerate(dataloader), desc=tqdm_prefix, total=len(dataloader))
        for step_id, data in progress:
            with accelerator.accumulate(model):
                if step_id == 0:
                    optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                loss_synced: torch.Tensor = model_logger.on_step_end(loss.detach(), accelerator, model, step_id, epoch_id)
                accelerator.log({"training_loss": loss_synced.item()}, step=global_step)
                progress.set_description(
                    f"{tqdm_prefix}, step={step_id}, loss={loss_synced.item():.7f}, "
                    f"peak_gpu_memory={torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}GiB"
                )
                scheduler.step()
                global_step += 1
            if profiler is not None:
                profiler.step()

        model_logger.on_epoch_end(accelerator, model, epoch_id)

    if profiler is not None:
        profiler.stop()


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    makedirs(path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, path.join(output_path, "data_cache", f"{data_id}.pth"))


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser


def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser
