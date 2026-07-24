from PIL import Image


def crop_and_resize(video, target_height, target_width, resample="lanczos"):
    video_cropped_and_resized = []
    resample_method = {"bilinear": Image.Resampling.BILINEAR, "lanczos": Image.Resampling.LANCZOS}[resample]

    for frame in video:
        width, height = frame.size

        if (height, width) != (target_height, target_width):
            target_aspect = target_width / target_height
            frame_aspect = width / height

            if frame_aspect > target_aspect:
                # Frame is wider than target, crop width
                new_width = round(height * target_aspect)
                left = (width - new_width) // 2
                frame = frame.crop((left, 0, left + new_width, height))
            elif frame_aspect < target_aspect:
                # Frame is taller than target, crop height
                new_height = round(width / target_aspect)
                top = (height - new_height) // 2
                frame = frame.crop((0, top, width, top + new_height))

            # Resize to target dimensions
            frame = frame.resize((target_width, target_height), resample_method)

        video_cropped_and_resized.append(frame)

    return video_cropped_and_resized
