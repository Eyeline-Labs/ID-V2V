# VACE Preprocessing Function Usage

This document explains how to use the `run_vace_preprocess` function from other Python files.

## Overview

The `run_vace_preprocess` function allows you to use VACE preprocessing capabilities as a standalone function that can be imported and called from other Python scripts, without needing to run the script from the command line.

## Basic Usage

```python
from annotation_tools.vace.vace_preproccess import run_vace_preprocess

# Basic usage with auto-detected models path
result = run_vace_preprocess(
    task="depth",
    video="path/to/your/video.mp4",
    pre_save_dir="output/depth_result",
    save_fps=30
)

print(f"Result: {result}")
```

## Function Parameters

### Required Parameters
- `task` (str): The task to run (must be in VACE_PREPROCCESS_CONFIGS)

### Optional Parameters
- `video` (str): Path to video files, comma-separated for multiple videos
- `image` (str): Path to image files, comma-separated for multiple images
- `mask` (str): Path to mask files, comma-separated for multiple masks
- `bbox` (list): List of bounding boxes as [x1, y1, x2, y2]
- `caption` (str): Caption text
- `label` (str): Labels, comma-separated for multiple labels
- `mode` (str): Task-specific mode
- `direction` (str): Outpainting direction, comma-separated
- `expand_ratio` (float): Outpainting expansion ratio
- `expand_num` (int): Number of frames to extend
- `maskaug_mode` (str): Mask augmentation mode
- `maskaug_ratio` (float): Mask augmentation ratio
- `pre_save_dir` (str): Output directory path
- `save_fps` (int): FPS for saved videos (default: 16)
- `models_base_path` (str): Base path for model files (auto-detected if None)

## Model Path Resolution

The function automatically handles model path resolution in several ways:

### 1. Auto-detection (Default)
If `models_base_path` is not provided, the function will automatically try to detect the models directory by looking in common locations:
- `./models` (relative to the script)
- `../models` (parent directory)
- `../../models` (two levels up)
- `../../../models` (three levels up)
- `./annotation_tools/vace/models` (relative to current working directory)

### 2. Explicit Path
You can provide the `models_base_path` parameter explicitly:

```python
result = run_vace_preprocess(
    task="depth",
    video="path/to/video.mp4",
    models_base_path="/path/to/your/models"
)
```

### 3. Environment Variable
You can also set the `VACE_MODELS_PATH` environment variable:

```bash
export VACE_MODELS_PATH="/path/to/your/models"
```

## Example Use Cases

### Video Depth Estimation
```python
result = run_vace_preprocess(
    task="depth",
    video="input.mp4",
    pre_save_dir="output/depth",
    save_fps=30
)
```

### Image Depth Estimation
```python
result = run_vace_preprocess(
    task="image_depth",
    image="input.jpg",
    pre_save_dir="output/image_depth"
)
```

### Layout with Bounding Box
```python
result = run_vace_preprocess(
    task="layout_bbox",
    bbox=[[100, 100, 300, 400]],  # [x1, y1, x2, y2]
    label="person",
    pre_save_dir="output/layout"
)
```

### Video Inpainting
```python
result = run_vace_preprocess(
    task="inpainting_mask",
    video="input.mp4",
    mask="mask.png",
    pre_save_dir="output/inpainting"
)
```

### Video Outpainting
```python
result = run_vace_preprocess(
    task="outpainting",
    video="input.mp4",
    direction="left,right",
    expand_ratio=0.25,
    pre_save_dir="output/outpainting"
)
```

## Return Value

The function returns a dictionary containing paths to the saved results:

```python
{
    'src_video': 'output/depth/src_video-depth.mp4',
    'src_mask': 'output/depth/src_mask-depth.mp4',
    'src_ref_images': 'output/depth/src_ref_image-depth.png'
}
```

## Error Handling

The function includes robust error handling:
- Validates that the task exists in the configuration
- Checks that required inputs are provided
- Provides clear error messages for missing files or invalid parameters
- Warns if model paths cannot be auto-detected

## Available Tasks

The following tasks are available (check `VACE_PREPROCCESS_CONFIGS` for the complete list):

### Video Tasks
- `depth`: Video depth estimation
- `depthv2`: Video depth estimation (v2)
- `flow`: Video optical flow
- `gray`: Video grayscale conversion
- `pose`: Video pose estimation
- `pose_body`: Video body pose estimation
- `scribble`: Video scribble generation
- `inpainting`: Video inpainting
- `outpainting`: Video outpainting
- `layout_bbox`: Layout with bounding box
- `layout_track`: Layout tracking

### Image Tasks
- `image_depth`: Image depth estimation
- `image_gray`: Image grayscale conversion
- `image_pose`: Image pose estimation
- `image_scribble`: Image scribble generation
- `image_inpainting`: Image inpainting
- `image_outpainting`: Image outpainting

## Troubleshooting

### Model Path Issues
If you encounter model loading errors:
1. Check that the models directory exists and contains the required model files
2. Provide the `models_base_path` parameter explicitly
3. Set the `VACE_MODELS_PATH` environment variable
4. Check the console output for auto-detection messages

### Import Issues
If you have import issues:
1. Make sure the `annotation_tools` directory is in your Python path
2. Use absolute imports: `from annotation_tools.vace.vace_preproccess import run_vace_preprocess`
3. Or add the directory to your path: `sys.path.append('/path/to/annotation_tools')`

### Task Not Found
If you get "Unsupport task" error:
1. Check the available tasks in `VACE_PREPROCCESS_CONFIGS`
2. Make sure the task name is spelled correctly
3. Check the configuration files for the complete list of available tasks 