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

def visualize_3d_volume(frames, masks, use_color=True, voxel_size=1.1, downsample_factor=1, max_points=2000000, preserve_boundary_frames=True):
    """
    Create a 3D voxel grid visualization from masked frames.
    Args:
        frames (list): Original frames (grayscale or color)
        masks (list): Segmentation masks
        use_color (bool): Whether to use color information or grayscale
        voxel_size (float): Size of voxels in the final grid
        downsample_factor (int): Skip every N pixels to reduce data
        max_points (int): Maximum number of points to process
        preserve_boundary_frames (bool): Whether to preserve all points in first and last frames
    """
    import numpy as np
    import open3d as o3d
    from tqdm import tqdm
    import time
    
    start_time = time.time()
    
    # Preprocessing: determine total number of points and calculate appropriate downsampling
    total_points = sum(np.sum(mask >= 255) for mask in masks)
    print(f"Total potential points: {total_points}")
    
    # Auto-adjust downsample factor if needed
    if total_points > max_points and downsample_factor == 1:
        # Count points in first and last frame if we're preserving them
        first_last_points = 0
        if preserve_boundary_frames and len(masks) >= 2:
            first_last_points = np.sum(masks[0] >= 255) + np.sum(masks[-1] >= 255)
            remaining_points = total_points - first_last_points
            # Calculate downsample factor for middle frames
            if remaining_points > 0:
                downsample_factor = int(np.ceil((total_points - first_last_points) / 
                                             (max_points - first_last_points)))
        else:
            downsample_factor = int(np.ceil(total_points / max_points))
            
        print(f"Auto-adjusting downsample factor to {downsample_factor}")
    
    # Pre-allocate arrays instead of appending
    estimated_points = min(total_points, max_points)
    voxels = np.zeros((estimated_points, 3), dtype=np.float32)
    colors = np.zeros((estimated_points, 3), dtype=np.float32)
    
    # Vectorized operations for faster processing
    idx = 0
    for z, (frame, mask) in enumerate(tqdm(zip(frames, masks), total=len(masks), desc="Processing frames")):
        # Find indices where mask is non-zero (foreground)
        y_indices, x_indices = np.where(mask >= 255)
        
        # Apply downsampling, but preserve first and last frames if requested
        is_boundary_frame = preserve_boundary_frames and (z == 0 or z == len(frames) - 1)
        
        if downsample_factor > 1 and not is_boundary_frame:
            points_count = len(y_indices)
            indices = np.arange(0, points_count, downsample_factor)
            y_indices = y_indices[indices]
            x_indices = x_indices[indices]
        
        # Check if adding these points would exceed our limits
        num_points = len(y_indices)
        if idx + num_points > estimated_points:
            # If this is a boundary frame and we're preserving them, make special efforts
            if is_boundary_frame and idx < estimated_points:
                # Take as many points as we can from boundary frame
                available_space = estimated_points - idx
                # Systematic sampling to preserve structure if we can't take all points
                if available_space < num_points:
                    sampling_rate = max(1, int(num_points / available_space))
                    y_indices = y_indices[::sampling_rate][:available_space]
                    x_indices = x_indices[::sampling_rate][:available_space]
                    num_points = len(y_indices)
                    print(f"Preserved {num_points} points from boundary frame {z} (sampling 1/{sampling_rate})")
            else:
                # Standard truncation for non-boundary frames
                num_points = estimated_points - idx
                if num_points <= 0:
                    print(f"Reached maximum point limit. Using {idx} points.")
                    break
                y_indices = y_indices[:num_points]
                x_indices = x_indices[:num_points]
        
        if is_boundary_frame:
            print(f"Frame {z}: Preserving all {num_points} points (boundary frame)")
            
        # Create batch of 3D points
        voxels[idx:idx+num_points, 0] = x_indices
        voxels[idx:idx+num_points, 1] = y_indices
        voxels[idx:idx+num_points, 2] = z
        
        # Determine colors in a vectorized way
        if use_color and len(frame.shape) == 3:  # Color frame (RGB)
            # Vectorized color extraction
            colors[idx:idx+num_points, 0] = frame[y_indices, x_indices, 0] / 255.0  # R
            colors[idx:idx+num_points, 1] = frame[y_indices, x_indices, 1] / 255.0  # G
            colors[idx:idx+num_points, 2] = frame[y_indices, x_indices, 2] / 255.0  # B
        else:  # Grayscale
            # Vectorized grayscale extraction
            intensity = frame[y_indices, x_indices] / 255.0
            colors[idx:idx+num_points, 0] = intensity  # R
            colors[idx:idx+num_points, 1] = intensity  # G
            colors[idx:idx+num_points, 2] = intensity  # B
            
        idx += num_points
    
    # Trim arrays to actual size
    voxels = voxels[:idx]
    colors = colors[:idx]
    
    print(f"Generados {idx} voxels en {time.time() - start_time:.2f} segundos")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxels)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Creada nube de puntos")
    
    # Optional: Further downsample the point cloud if still too large, 
    # but preserve boundary points if possible
    if len(voxels) > max_points // 2:
        # Extract boundary points (z=0 and z=max)
        if preserve_boundary_frames:
            z_values = np.asarray(pcd.points)[:, 2]
            min_z, max_z = np.min(z_values), np.max(z_values)
            
            boundary_indices = np.where((z_values == min_z) | (z_values == max_z))[0]
            boundary_points = np.asarray(pcd.points)[boundary_indices]
            boundary_colors = np.asarray(pcd.colors)[boundary_indices]
            
            # Downsample non-boundary points
            non_boundary_indices = np.where((z_values != min_z) & (z_values != max_z))[0]
            non_boundary_pcd = o3d.geometry.PointCloud()
            non_boundary_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[non_boundary_indices])
            non_boundary_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[non_boundary_indices])
            
            # Downsample non-boundary points
            print(f"Downsampling non-boundary point cloud with voxel size {voxel_size/2}")
            downsampled_pcd = non_boundary_pcd.voxel_down_sample(voxel_size=voxel_size/2)
            
            # Combine boundary and downsampled points
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_colors = np.asarray(downsampled_pcd.colors)
            
            combined_points = np.vstack((boundary_points, downsampled_points))
            combined_colors = np.vstack((boundary_colors, downsampled_colors))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_points)
            pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            
            print(f"Point cloud downsampled to {len(pcd.points)} points (preserved {len(boundary_points)} boundary points)")
        else:
            # Standard downsampling for all points
            print(f"Downsampling entire point cloud with voxel size {voxel_size/2}")
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size/2)
            print(f"Point cloud downsampled to {len(pcd.points)} points")
    
    # Create voxel grid
    print(f"Creando malla de voxeles con tamaño {voxel_size}")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size
    )
    print("Creada malla de voxeles")
    
    # Visualize
    print("Visualizando malla de voxeles")
    o3d.visualization.draw_geometries([voxel_grid])
    
    total_time = time.time() - start_time
    print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")
    
    return voxel_grid, pcd

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
    visualize_3d_volume(frames_for_viz, masks, use_color=apply_color)
    

if __name__ == "__main__":
    main()