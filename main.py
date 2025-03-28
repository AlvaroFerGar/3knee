import numpy as np
import open3d as o3d
import cv2
import os

def load_mp4_frames(video_path):
    """
    Load frames from an MP4 video file.
    
    Args:
        video_path (str): Path to the MP4 video file
    
    Returns:
        list: List of extracted frames
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    
    cap.release()
    return frames

def segment_knee(frame):
    """
    Segment the knee using Otsu's thresholding method.
    
    Args:
        frame (numpy.ndarray): Grayscale image frame
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    # Segmentation parameters
    GAUSSIAN_BLUR_KERNEL = (5, 5)  # Gaussian blur kernel size
    MORPH_OPEN_KERNEL_SIZE = 3  # Size of kernel for opening operation
    MORPH_CLOSE_KERNEL_SIZE = 5  # Size of kernel for closing operation
    
    AREA_THRESHOLD_PERCENT = 0.01  # Minimum blob size as percentage of image
    
    # 1. Reduce noise with Gaussian blur
    blurred = cv2.GaussianBlur(frame, 
                                GAUSSIAN_BLUR_KERNEL, 
                                0)  # Sigma 0 means auto-calculate
    
    # 2. Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(blurred, 
                                   30,  # Threshold value is ignored with THRESH_OTSU 
                                   255, 
                                   cv2.THRESH_BINARY_INV)
    
    # 3. Morphological operations to clean up the image
    # Create kernels
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (MORPH_OPEN_KERNEL_SIZE, MORPH_OPEN_KERNEL_SIZE)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (MORPH_CLOSE_KERNEL_SIZE, MORPH_CLOSE_KERNEL_SIZE)
    )
    
    # Opening to remove small white noise
    opened = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, open_kernel)
    
    # Closing to close small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    
    # 4. Find connected components
    num_labels, labels = cv2.connectedComponents(closed)
    
    # 5. Filter components based on size
    # Calculate area threshold
    min_area = frame.size * AREA_THRESHOLD_PERCENT
    
    # Find components and their sizes
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort components by size (excluding background)
    component_sizes = dict(zip(unique[1:], counts[1:]))
    sorted_components = sorted(
        component_sizes.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Create the final mask
    mask = np.ones(frame.shape, dtype=np.uint8)*255
    
    # Add only sufficiently large components
    for component, size in sorted_components:
        if size > min_area:
            mask[labels == component] = 0
    
    return mask

def save_segmented_frames(frames, masks):
    """
    Save segmented frames with transparent background for debugging.
    
    Args:
        frames (list): Original grayscale frames
        masks (list): Segmentation masks
    """
    # Create output directory if it doesn't exist
    os.makedirs('debug_frames', exist_ok=True)
    
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        # Create a 4-channel image (RGBA)
        rgba_image = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        
        # Set the RGB channels to the original grayscale image
        rgba_image[:,:,0] = frame
        rgba_image[:,:,1] = frame
        rgba_image[:,:,2] = frame
        
        # Set alpha channel based on the mask
        rgba_image[:,:,3] = mask
        
        # Save the image
        output_path = f'debug_frames/frame_{i:04d}_segmented.png'
        cv2.imwrite(output_path, rgba_image)
        
        # Optional: save original and mask separately for additional debugging
        cv2.imwrite(f'debug_frames/frame_{i:04d}_original.png', frame)
        cv2.imwrite(f'debug_frames/frame_{i:04d}_mask.png', mask)

def visualize_3d_volume(frames, masks):
    """
    Create a 3D voxel grid visualization from masked frames.
    
    Args:
        frames (list): Original grayscale frames
        masks (list): Segmentation masks
    """
    # Collect voxels
    voxels = []
    colors = []

    for z, (frame, mask) in enumerate(zip(frames, masks)):
        # Find indices where mask is non-zero (foreground)
        y_indices, x_indices = np.where(mask >= 255)
        
        for x, y in zip(x_indices, y_indices):
            # Create 3D point (x, y, z) and use frame intensity as color
            voxels.append([x, y, z])
            # Normalize pixel intensity to 0-1 range for color
            color_intensity = frame[y, x] / 255.0
            colors.append([color_intensity, color_intensity, color_intensity])

    # Convert to numpy arrays
    voxels = np.array(voxels)
    colors = np.array(colors)

    # Create Open3D point cloud first
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxels)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create voxel grid with a specific voxel size
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, 
        voxel_size=1.1  # Adjust this value to control voxel size and density
    )

    # Visualize
    o3d.visualization.draw_geometries([voxel_grid])

def main():
    # Path to your MP4 video file
    video_path = 'a.mp4'
    
    # Load frames
    frames = load_mp4_frames(video_path)
    
    # Segment knee in each frame
    masks = [segment_knee(frame) for frame in frames]
    
    # Save segmented frames for debugging
    save_segmented_frames(frames, masks)
    
    # Visualize 3D volume
    visualize_3d_volume(frames, masks)
    
    print(f"Processed {len(frames)} frames")

if __name__ == "__main__":
    main()