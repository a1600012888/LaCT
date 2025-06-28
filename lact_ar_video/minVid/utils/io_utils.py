import numpy as np
import imageio
import os

def export_to_video(video_frames, output_video_path, fps=8):
    """
    Exports a video from a numpy array of frames.
    
    Args:
        video_frames (np.ndarray): Video frames in numpy format with shape [num_frames, height, width, channels]
                                   with values in range [0, 255] and dtype np.uint8
        output_video_path (str): Path to save the video
        fps (int, optional): Frames per second. Defaults to 8.
    """
    if isinstance(video_frames, list):
        # If input is a list of PIL images or frames, convert to numpy array
        if hasattr(video_frames[0], "numpy"):  # PIL images
            video_frames = np.stack([np.array(frame) for frame in video_frames], axis=0)
        else:
            video_frames = np.stack(video_frames, axis=0)
    
    # Ensure values are uint8
    if video_frames.dtype != np.uint8:
        if np.max(video_frames) <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
        else:
            video_frames = video_frames.astype(np.uint8)
    
    # Write video using imageio
    imageio.mimsave(output_video_path, video_frames, fps=fps)


def save_video(video_tensor, save_path, save_fps=24):
    """
    Save a video tensor to disk using imageio.
    
    Args:
        video_tensor (torch.Tensor): Video tensor of shape [b, f, c, h, w] in uint8 format
                                    or [f, c, h, w] for a single video
        save_path (str): Path to save the video to
        save_fps (int): Frames per second for the saved video (default: 24)
    """
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Handle both batch and single video cases
    if len(video_tensor.shape) == 5:
        # Batch of videos [b, f, c, h, w]
        batch_size = video_tensor.shape[0]
        
        # If save_path doesn't have an extension, add one
        base_path, ext = os.path.splitext(save_path)
        if not ext:
            ext = '.mp4'
            
        # Save each video in the batch
        for i in range(batch_size):
            # Generate path for each video in the batch
            if batch_size > 1:
                video_path = f"{base_path}_{i}{ext}"
            else:
                video_path = f"{base_path}{ext}"
            
            # Extract single video and convert to numpy
            single_video = video_tensor[i].detach().cpu().numpy()
            
            # Ensure we have the right format - imageio expects [f, h, w, c]
            single_video = np.transpose(single_video, (0, 2, 3, 1))
            
            # Write video to disk
            imageio.mimsave(video_path, single_video, fps=save_fps)
            
            print(f"Saved video to {video_path} at {save_fps} FPS")
    
    else:
        # Single video [f, c, h, w]
        # Convert to numpy and transpose to [f, h, w, c]
        video_np = video_tensor.detach().cpu().numpy()
        video_np = np.transpose(video_np, (0, 2, 3, 1))
        
        # Write video to disk
        imageio.mimsave(save_path, video_np, fps=save_fps)
        
        print(f"Saved video to {save_path} at {save_fps} FPS")