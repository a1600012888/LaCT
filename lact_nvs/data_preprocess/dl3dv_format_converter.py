#!/usr/bin/env python3
"""
DL3DV Format Converter

This script converts DL3DV benchmark data to the standard format used by LaCT NVS.
It follows the processing rules from process_dl3dv.py:
1. Applies lens distortion correction using OpenCV
2. Performs coordinate system transformations
3. Creates undistorted images in images_undistort/ folder
4. Converts transforms.json to opencv_cameras.json format
5. Skips vertical videos (height > width)
6. Creates a dl3dv_sample_data_path.json file
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
import cv2


def process_one_scene(scene_path, output_dir=None):
    """
    Process one scene following the rules from process_dl3dv.py

    This is copied from Ziwen Chen's code here: https://github.com/arthurhero/Long-LRM/blob/main/data/prosess_dl3dv.py
    
    Args:
        scene_path: Path to one scene folder
        output_dir: Optional output directory (if None, uses scene_path)
    
    Returns:
        bool: True if successful, False otherwise
    """
    scene_path = Path(scene_path)
    if output_dir is None:
        output_dir = scene_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {scene_path}")
    scene_name = scene_path.name
    
    # Read transforms.json
    json_file = scene_path / 'nerfstudio' / 'transforms.json'
    if not json_file.exists():
        print(f"Warning: transforms.json not found in {scene_path / 'nerfstudio'}")
        return False
    
    json_data = json.load(open(json_file, 'r'))
    new_json_file = output_dir / 'opencv_cameras.json'
    new_data = []  # Changed to direct array instead of dict with frames
    
    # Extract camera parameters
    w_ = json_data['w']
    h_ = json_data['h']
    fx_ = json_data['fl_x']
    fy_ = json_data['fl_y']
    cx_ = json_data['cx']
    cy_ = json_data['cy']
    k1, k2, p1, p2 = json_data['k1'], json_data['k2'], json_data['p1'], json_data['p2']
    distort = np.asarray([k1, k2, p1, p2])
    
    # Skip vertical videos
    if h_ > w_:
        print(f"Skip vertical videos for now: {scene_path}, {h_}, {w_}")
        return False
    
    num_frames = len(json_data['frames'])
    print(f"num_frames: {num_frames}")

    # Create undistort folder
    undistort_dir = output_dir / 'images_undistort'
    undistort_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        frame = json_data['frames'][i]
        file_path = 'images_4/' + frame['file_path'].split('/')[-1]

        # Load and undistort image
        image_path = scene_path / 'nerfstudio' / file_path
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
            
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue
            
        h, w, _ = image.shape
        fx, fy, cx, cy = fx_/w_*w, fy_/h_*h, cx_/w_*w, cy_/h_*h
        intr = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Apply undistortion
        new_intr, roi = cv2.getOptimalNewCameraMatrix(intr, distort, (w, h), 0, (w, h))
        dst = cv2.undistort(image, intr, distort, None, new_intr)
        image = dst
        h, w, _ = image.shape
        fx, fy, cx, cy = new_intr[0, 0], new_intr[1, 1], new_intr[0, 2], new_intr[1, 2]

        # Save undistorted image
        output_image_path = undistort_dir / frame['file_path'].split('/')[-1]
        cv2.imwrite(str(output_image_path), image)

        # Transform camera pose (following process_dl3dv.py rules)
        c2w = np.asarray(frame["transform_matrix"])
        c2w[0:3, 1:3] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1
        w2c = np.linalg.inv(c2w)
        
        # Create frame data with reordered fields to match GSO format
        file_path = 'images_undistort/' + frame['file_path'].split('/')[-1]
        frame_new = {
            "w": w, "h": h,  # Reordered to match GSO format
            "fx": fx, "fy": fy, 
            "cx": cx, "cy": cy,
            "w2c": w2c.tolist(),
            "file_path": file_path
        }
        new_data.append(frame_new)  # Changed from append to frames array
    
    # Save opencv_cameras.json
    json.dump(new_data, open(new_json_file, 'w'), indent=4)
    print(f"Successfully processed {scene_path}")
    return True


def convert_folder(folder_path, output_dir=None):
    """
    Convert a single DL3DV folder to standard format.
    
    Args:
        folder_path: Path to the folder containing nerfstudio/ subfolder
        output_dir: Optional output directory (if None, uses folder_path)
    """
    folder_path = Path(folder_path)
    nerfstudio_path = folder_path / 'nerfstudio'
    
    if not nerfstudio_path.exists():
        print(f"Warning: nerfstudio folder not found in {folder_path}")
        return False
    
    # Process the scene following process_dl3dv.py rules
    success = process_one_scene(folder_path, output_dir)
    
    if success and output_dir is None:
        # If processing in-place, also move images_4 to images
        images_4_path = nerfstudio_path / 'images_4'
        images_path = folder_path / 'images'
        
        if images_4_path.exists():
            if images_path.exists():
                shutil.rmtree(images_path)
            shutil.move(str(images_4_path), str(images_path))
        
        # Remove nerfstudio folder
        shutil.rmtree(nerfstudio_path)
    
    return success


def create_sample_data_path_json(benchmark_dir, output_file):
    """
    Create dl3dv_sample_data_path.json file.
    
    Args:
        benchmark_dir: Path to the dl3dv_benchmark directory
        output_file: Path to the output JSON file
    """
    benchmark_dir = Path(benchmark_dir)
    data_paths = []
    
    # Find all folders in the benchmark directory
    for item in benchmark_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if the folder has opencv_cameras.json
            cameras_file = item / 'opencv_cameras.json'
            if cameras_file.exists():
                # Create relative path from data_example
                relative_path = f"dl3dv_processed/{item.name}/opencv_cameras.json"
                data_paths.append(relative_path)
    
    # Sort for consistent output
    data_paths.sort()
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data_paths, f, indent=2)
    
    print(f"Created {output_file} with {len(data_paths)} data paths")


def main():
    """Main function to convert all DL3DV benchmark data."""
    # Define paths
    current_dir = Path(__file__).parent
    data_example_dir = current_dir.parent / 'data_example'
    benchmark_dir = data_example_dir / 'dl3dv_benchmark'
    output_json = data_example_dir / 'dl3dv_sample_data_path.json'
    
    # Optional: Define different input and output directories
    input_dir = benchmark_dir  # Can be changed to different input directory
    output_base_dir = data_example_dir / 'dl3dv_processed'  # Different output directory
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return
    
    print("Starting DL3DV format conversion...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_base_dir}")
    
    # Convert each folder
    converted_count = 0
    for item in input_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            print(f"Processing {item.name}...")
            
            # Create output directory for this scene
            scene_output_dir = output_base_dir / item.name
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            if convert_folder(item, scene_output_dir):
                converted_count += 1
    
    print(f"Converted {converted_count} folders")
    
    # Create sample data path JSON (pointing to the new output directory)
    create_sample_data_path_json(output_base_dir, output_json)
    
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main() 