import numpy as np
import open3d as o3d
import cv2
import os
from tqdm import tqdm
import time

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

def interpolate_frames(frames, num_intermediate=2):
    """
    Genera frames intermedios interpolando entre frames consecutivos.
    
    Args:
        frames (list): Lista de frames originales (imágenes en escala de grises)
        num_intermediate (int): Número de frames intermedios a generar entre cada par
    
    Returns:
        list: Lista aumentada de frames, incluyendo los originales e interpolados
    """
    import numpy as np
    
    if len(frames) < 2:
        return frames
    
    interpolated_frames = []
    
    # Agregar el primer frame original
    interpolated_frames.append(frames[0])
    
    # Interpolar entre cada par de frames consecutivos
    for i in range(len(frames) - 1):
        frame_current = frames[i].astype(float)
        frame_next = frames[i + 1].astype(float)
        
        # Generar frames intermedios
        for j in range(1, num_intermediate + 1):
            # Calcular factor de interpolación (0.0 a 1.0)
            alpha = j / (num_intermediate + 1)
            
            # Interpolación lineal: new_frame = (1-alpha)*frame1 + alpha*frame2
            interpolated = ((1 - alpha) * frame_current + alpha * frame_next).astype(np.uint8)
            interpolated_frames.append(interpolated)
        
        # Agregar el siguiente frame original (excepto el último que se agrega fuera del bucle)
        if i < len(frames) - 2:
            interpolated_frames.append(frames[i + 1])
    
    # Agregar el último frame original
    interpolated_frames.append(frames[-1])
    
    return interpolated_frames

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
        
        # Check if the frame is grayscale or RGB
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # Grayscale image
            rgba_image[:,:,0] = frame if len(frame.shape) == 2 else frame[:,:,0]
            rgba_image[:,:,1] = frame if len(frame.shape) == 2 else frame[:,:,0]
            rgba_image[:,:,2] = frame if len(frame.shape) == 2 else frame[:,:,0]
        else:
            # RGB image
            rgba_image[:,:,0] = frame[:,:,0]
            rgba_image[:,:,1] = frame[:,:,1]
            rgba_image[:,:,2] = frame[:,:,2]
        
        # Set alpha channel based on the mask
        rgba_image[:,:,3] = mask
        
        # Save the image
        output_path = f'debug_frames/frame_{i:04d}_segmented.png'
        cv2.imwrite(output_path, rgba_image)
        
        # Optional: save original and mask separately for additional debugging
        #cv2.imwrite(f'debug_frames/frame_{i:04d}_original.png', frame)
        #cv2.imwrite(f'debug_frames/frame_{i:04d}_mask.png', mask)
        

def apply_colormap(frames, colormap_name='jet', normalize=True):
    """
    Aplica un mapa de colores a los frames en escala de grises.
    
    Args:
        frames (list): Lista de frames en escala de grises
        colormap_name (str): Nombre del mapa de colores de OpenCV ('jet', 'hot', 'hsv', etc.)
        normalize (bool): Si es True, normaliza los valores de intensidad antes de aplicar el colormap
    
    Returns:
        list: Lista de frames con el mapa de colores aplicado (RGB)
    """
    import cv2
    import numpy as np
    
    # Mapear nombres de colormaps a constantes de OpenCV
    colormap_dict = {
        'autumn': cv2.COLORMAP_AUTUMN,
        'bone': cv2.COLORMAP_BONE,
        'jet': cv2.COLORMAP_JET,
        'winter': cv2.COLORMAP_WINTER,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'ocean': cv2.COLORMAP_OCEAN,
        'summer': cv2.COLORMAP_SUMMER,
        'spring': cv2.COLORMAP_SPRING,
        'cool': cv2.COLORMAP_COOL,
        'hsv': cv2.COLORMAP_HSV,
        'pink': cv2.COLORMAP_PINK,
        'hot': cv2.COLORMAP_HOT,
        'parula': cv2.COLORMAP_PARULA,
        'magma': cv2.COLORMAP_MAGMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'plasma': cv2.COLORMAP_PLASMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'cividis': cv2.COLORMAP_CIVIDIS,
        'twilight': cv2.COLORMAP_TWILIGHT,
        'twilight_shifted': cv2.COLORMAP_TWILIGHT_SHIFTED
    }
    
    # Verificar que el colormap solicitado existe
    if colormap_name not in colormap_dict:
        print(f"Colormap '{colormap_name}' no disponible. Usando 'jet' por defecto.")
        colormap_name = 'jet'
    
    colormap = colormap_dict[colormap_name]
    colored_frames = []
    
    for frame in frames:
        # Normalizar el frame si es necesario
        if normalize:
            # Convertir a float32 para evitar problemas con la normalización
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        else:
            frame_normalized = frame.copy()
            
        # Aplicar el mapa de colores
        colored_frame = cv2.applyColorMap(frame_normalized, colormap)
        
        # Convertir de BGR (OpenCV) a RGB para Open3D
        colored_frame = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)
        
        colored_frames.append(colored_frame)
    
    return colored_frames

def visualize_3d_volume(frames, masks, use_color=True, voxel_size=1.1):
    """
    Create a 3D voxel grid visualization from masked frames using an efficient
    visibility check based on a 5-channel representation.
    
    Args:
        frames (list): Original frames (grayscale or color)
        masks (list): Segmentation masks
        use_color (bool): Whether to use color information or grayscale
        voxel_size (float): Size of voxels in the final grid
    """
    
    start_time = time.time()
    
    # Get dimensions
    height, width = masks[0].shape[:2]
    depth = len(frames)
    
    print(f"Volume dimensions: {width}x{height}x{depth}")
    
    # Create 5-channel volume matrix: R, G, B, Z, Alpha
    print("Creating 5-channel volume matrix...")
    volume = np.zeros((height, width, 5), dtype=np.float32)
    
    # Fill the volume matrix with data from frames and masks
    print("Filling volume matrix with frame and mask data...")
    for z, (frame, mask) in enumerate(tqdm(zip(frames, masks), total=len(masks), desc="Processing frames")):
        # Process only pixels where the mask is non-zero
        mask_indices = mask >= 255
        
        if not np.any(mask_indices):
            continue
        
        # Update the alpha channel (from mask)
        volume[:, :, 4] = np.maximum(volume[:, :, 4], mask_indices.astype(np.float32))
        
        # For pixels with non-zero masks, update the z-value if this is the highest z so far
        # This ensures we store the frontmost visible z-value
        current_z_values = volume[:, :, 3]
        update_indices = (mask_indices) & ((z > current_z_values) | (current_z_values == 0))
        
        if np.any(update_indices):
            # Update z-channel
            volume[update_indices, 3] = z
            
            # Update color channels
            if use_color and len(frame.shape) == 3:  # Color frame (RGB)
                volume[update_indices, 0] = frame[update_indices, 0] / 255.0  # R
                volume[update_indices, 1] = frame[update_indices, 1] / 255.0  # G
                volume[update_indices, 2] = frame[update_indices, 2] / 255.0  # B
            else:  # Grayscale
                intensity = frame[update_indices] / 255.0
                volume[update_indices, 0] = intensity  # R
                volume[update_indices, 1] = intensity  # G
                volume[update_indices, 2] = intensity  # B
    
    # Now determine visible voxels (those with at least one empty/transparent neighbor)
    print("Determining visible voxels...")
    
    # Create 3D mask array matching the dimensions of frames
    visible_voxels = np.zeros((height, width, depth), dtype=bool)
    alpha_volume = np.zeros((height, width, depth), dtype=bool)
    
    # First, mark all voxels from the masks
    for z, mask in enumerate(tqdm(masks, desc="Creating 3D alpha volume")):
        alpha_volume[:, :, z] = (mask >= 255)
    
    # Check for empty neighbors (6-connectivity)
    print("Checking for visible faces...")
    
    # Pad the alpha volume to handle boundary checks more easily
    padded_alpha = np.pad(alpha_volume, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    # Define the 6 neighbor directions
    neighbors = [
        (0, 0, 1),  # front
        (0, 0, -1), # back
        (0, 1, 0),  # top
        (0, -1, 0), # bottom
        (1, 0, 0),  # right
        (-1, 0, 0)  # left
    ]
    
    # Check each voxel's neighbors
    for z in tqdm(range(depth), desc="Finding visible voxels"):
        for y in range(height):
            for x in range(width):
                # Skip empty voxels
                if not alpha_volume[y, x, z]:
                    continue
                
                # Check if any of the 6 neighbors is empty
                for dx, dy, dz in neighbors:
                    nx, ny, nz = x + dx + 1, y + dy + 1, z + dz + 1  # +1 due to padding
                    
                    # If any neighbor is empty, mark this voxel as visible
                    if not padded_alpha[ny, nx, nz]:
                        visible_voxels[y, x, z] = True
                        break
    
    # Count visible voxels
    num_visible = np.sum(visible_voxels)
    print(f"Found {num_visible} visible voxels out of {np.sum(alpha_volume)} total filled voxels")
    
    # Create point cloud for visible voxels
    print("Creating point cloud from visible voxels...")
    points = []
    colors = []
    
    for z in tqdm(range(depth), desc="Creating visible voxel point cloud"):
        y_indices, x_indices = np.where(visible_voxels[:, :, z])
        
        for y, x in zip(y_indices, x_indices):
            points.append([x, y, z])
            
            # Get color from volume matrix
            if use_color:
                colors.append([
                    volume[y, x, 0],  # R
                    volume[y, x, 1],  # G
                    volume[y, x, 2]   # B
                ])
            else:
                # Use z-value for grayscale coloring
                normalized_z = z / depth
                colors.append([normalized_z, normalized_z, normalized_z])
    
    print(f"Created point cloud with {len(points)} points")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Create voxel grid
    print(f"Creating voxel grid with size {voxel_size}")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size
    )
    
    # Visualize
    print("Visualizing voxel grid")
    o3d.visualization.draw_geometries([voxel_grid])

def main():
    # Path to your MP4 video file
    video_path = 'a.mp4'
    
    # Parámetros configurables
    num_intermediate_frames = 10  # Número de frames a interpolar entre cada par
    apply_color = True           # Aplicar mapa de colores a los frames
    colormap_name = 'twilight_shifted'        # Nombre del mapa de colores
    
    # Load frames
    frames = load_mp4_frames(video_path)
    print(f"Loaded {len(frames)} original frames")
    
    # Interpolar frames para aumentar la resolución temporal
    interpolated_frames = interpolate_frames(frames, num_intermediate=num_intermediate_frames)
    print(f"Generated {len(interpolated_frames)} frames after interpolation")
    
    # Aplicar mapa de colores si está habilitado
    if apply_color:
        colored_frames = apply_colormap(interpolated_frames, colormap_name=colormap_name)
        print(f"Applied '{colormap_name}' colormap to frames")
        # Usamos los frames coloreados para visualización
        frames_for_viz = colored_frames
    else:
        # Usamos los frames en escala de grises
        frames_for_viz = interpolated_frames
    
    # Segment knee in each frame
    masks = [segment_knee(frame) for frame in interpolated_frames]
    
    # Save segmented frames for debugging
    save_segmented_frames(frames_for_viz, masks)
    
    # Visualize 3D volume
    #visualize_3d_volume(frames_for_viz, masks, use_color=apply_color)
    

if __name__ == "__main__":
    main()