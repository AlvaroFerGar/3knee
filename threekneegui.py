from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QSlider, QCheckBox, 
                            QComboBox, QGroupBox, QFileDialog, QSpinBox, QDoubleSpinBox)
import open3d as o3d
import os
from threekneecore import threekneeCore
import numpy as np

class Open3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.vis = None
        self.geometry = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create a placeholder label
        self.placeholder = QLabel("3D visualization will appear here")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.placeholder)
        
    def setup_visualization(self, geometry):
        self.geometry = geometry
        
        # Create Open3D visualization window
        if self.vis is not None:
            self.vis.destroy_window()
            
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Knee MRI 3D View", width=800, height=600)
        self.vis.add_geometry(geometry)
        
        # Set rendering options
        render_option = self.vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Optimize view
        self.vis.get_view_control().set_zoom(0.7)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # For now, handle visualization in a separate window
        # Embedding Open3D in Qt can be complex and platform-dependent
        self.vis.run()

class threekneeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.processing_thread = None
        self.voxel_grid = None
        self.pcd = None
        
    def init_ui(self):
        self.setWindowTitle("MRI Knee Processing GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(400)
        main_layout.addWidget(control_panel)
        
        # File selection
        file_group = QGroupBox("Input File")
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        file_layout.addWidget(self.file_path_label)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Interpolation settings
        interp_group = QGroupBox("Frame Interpolation")
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Intermediate Frames:"))
        self.interp_frames_spin = QSpinBox()
        self.interp_frames_spin.setRange(0, 50)
        self.interp_frames_spin.setValue(10)
        interp_layout.addWidget(self.interp_frames_spin)
        interp_group.setLayout(interp_layout)
        control_layout.addWidget(interp_group)
        
        # CLAHE settings
        clahe_group = QGroupBox("CLAHE Enhancement")
        clahe_layout = QVBoxLayout()
        self.apply_clahe_checkbox = QCheckBox("Apply CLAHE")
        self.apply_clahe_checkbox.setChecked(True)
        clahe_layout.addWidget(self.apply_clahe_checkbox)
        
        clahe_params_layout = QHBoxLayout()
        clahe_params_layout.addWidget(QLabel("Clip Limit:"))
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.5, 10.0)
        self.clahe_clip_spin.setValue(2.0)
        self.clahe_clip_spin.setSingleStep(0.5)
        clahe_params_layout.addWidget(self.clahe_clip_spin)
        
        clahe_params_layout.addWidget(QLabel("Tile Size:"))
        self.clahe_tile_spin = QSpinBox()
        self.clahe_tile_spin.setRange(2, 16)
        self.clahe_tile_spin.setValue(8)
        clahe_params_layout.addWidget(self.clahe_tile_spin)
        
        clahe_layout.addLayout(clahe_params_layout)
        clahe_group.setLayout(clahe_layout)
        control_layout.addWidget(clahe_group)
        
        # Colormap settings
        color_group = QGroupBox("Colormap")
        color_layout = QVBoxLayout()
        self.apply_color_checkbox = QCheckBox("Apply Colormap")
        self.apply_color_checkbox.setChecked(True)
        color_layout.addWidget(self.apply_color_checkbox)
        
        color_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        colormaps = ['turbo', 'viridis', 'plasma', 'inferno', 'magma', 
                     'cividis', 'twilight', 'twilight_shifted', 'hsv', 'jet']
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('turbo')
        color_layout.addWidget(self.colormap_combo)
        color_group.setLayout(color_layout)
        control_layout.addWidget(color_group)
        
        # Segmentation settings
        segment_group = QGroupBox("Segmentation")
        segment_layout = QVBoxLayout()
        segment_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_value_label = QLabel("127")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        segment_layout.addWidget(self.threshold_slider)
        segment_layout.addWidget(self.threshold_value_label)
        segment_group.setLayout(segment_layout)
        control_layout.addWidget(segment_group)
        
        # 3D visualization settings
        vis_group = QGroupBox("3D Visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(QLabel("Voxel Size:"))
        self.voxel_size_spin = QDoubleSpinBox()
        self.voxel_size_spin.setRange(0.1, 5.0)
        self.voxel_size_spin.setValue(1.1)
        self.voxel_size_spin.setSingleStep(0.1)
        vis_layout.addWidget(self.voxel_size_spin)
        
        self.save_debug_checkbox = QCheckBox("Save Debug Frames")
        self.save_debug_checkbox.setChecked(True)
        vis_layout.addWidget(self.save_debug_checkbox)
        
        vis_group.setLayout(vis_layout)
        control_layout.addWidget(vis_group)
        
        # Process button
        self.process_button = QPushButton("Process and Visualize")
        self.process_button.clicked.connect(self.process_data)
        self.process_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        control_layout.addWidget(self.process_button)
        
        # Status area
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        
        # Progress log
        log_group = QGroupBox("Progress Log")
        log_layout = QVBoxLayout()
        self.log_text = QLabel("Ready to process...")
        self.log_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_text.setWordWrap(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        control_layout.addWidget(log_group)
        
        # 3D visualization area
        self.open3d_widget = Open3DWidget()
        main_layout.addWidget(self.open3d_widget, 1)
        
    def update_threshold_label(self):
        self.threshold_value_label.setText(str(self.threshold_slider.value()))
        
    def browse_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select MP4 File", "", "MP4 Files (*.mp4)"
        )
        
        if file_path:
            self.file_path_label.setText(os.path.basename(file_path))
            self.file_path_label.setToolTip(file_path)
            self.status_label.setText(f"Selected file: {os.path.basename(file_path)}")
            
    def process_data(self):
        file_path = self.file_path_label.toolTip()
        
        if not file_path or not os.path.exists(file_path):
            self.status_label.setText("Error: Please select a valid MP4 file")
            return
            
        # Gather all parameters
        params = {
            'video_path': file_path,
            'num_intermediate_frames': self.interp_frames_spin.value(),
            'apply_clahe': self.apply_clahe_checkbox.isChecked(),
            'clip_limit': self.clahe_clip_spin.value(),
            'tile_grid_size': (self.clahe_tile_spin.value(), self.clahe_tile_spin.value()),
            'apply_color': self.apply_color_checkbox.isChecked(),
            'colormap_name': self.colormap_combo.currentText(),
            'segment_threshold': self.threshold_slider.value(),
            'voxel_size': self.voxel_size_spin.value(),
            'save_debug_frames': self.save_debug_checkbox.isChecked()
        }
        
        # Disable UI while processing
        self.process_button.setEnabled(False)
        self.process_button.setText("Processing...")
        self.status_label.setText("Processing data...")
        self.log_text.setText("Starting processing...\n")
        
        # Start processing thread
        self.processing_thread = threekneeCore(params)
        self.processing_thread.progress.connect(self.update_log)
        self.processing_thread.finished.connect(self.processing_complete)
        self.processing_thread.start()
        
    def update_log(self, message):
        current_text = self.log_text.text()
        self.log_text.setText(f"{current_text}\n{message}")
        
    def processing_complete(self, voxel_grid, pcd):
        self.voxel_grid = voxel_grid
        self.pcd = pcd
        
        # Re-enable UI
        self.process_button.setEnabled(True)
        self.process_button.setText("Process and Visualize")
        self.status_label.setText("Processing complete")
        
        # Update visualization
        self.open3d_widget.setup_visualization(voxel_grid)