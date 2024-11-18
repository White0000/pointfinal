# YOLOv8 & Transformer Multi-modal Detection System

## Introduction
This project integrates YOLOv8 for object detection and a Transformer-based model for trajectory prediction, enabling multi-modal data processing such as point clouds, images, and videos. The system is tailored for applications in intelligent transportation and autonomous driving, offering real-time detection, trajectory forecasting, and comprehensive visualization.

---

## Key Features
- **Point Cloud Processing**: Performs voxel downsampling and image projection for optimized data handling.
- **Real-time Object Detection**: YOLOv8 supports robust and accurate detection of objects in images and videos.
- **Trajectory Prediction**: A Transformer model predicts future positions, velocities, and accelerations of dynamic objects.
- **3D Visualization**: Renders point clouds, predicted trajectories, and dynamic movements in an interactive environment.
- **Graphical User Interface**: PyQt5-based GUI with easy access to model training, prediction, and visualization features.

---

## Requirements
- **Python**: 3.11.9 (2024.2)  
- **CUDA**: 12.2  
- **GPU**: NVIDIA RTX 4080  

### Dependencies
```plaintext
numpy==1.24.4
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
open3d==0.18.0
PyQt5==5.15.11
ultralytics==8.3.31
opencv-python==4.10.0.84
matplotlib==3.9.2
pandas==2.2.3
seaborn==0.13.2
scipy==1.14.1
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/White0000/pointfinal.git
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python project_root/gui/main_ui.py
   ```

---

## Project Structure
```
project_root/
├── gui/                 # GUI-related code
│   ├── main_ui.py       # Main interface entry
│   ├── gui_controls.py  # Logic and multi-threading control
├── models/              # YOLOv8 and Transformer models
├── preprocessing/       # Data processing scripts
│   ├── dataset_loader.py    # Dataset loading
│   ├── pointcloud_preprocessing.py  # Point cloud handling
├── training/            # Model training scripts
│   ├── train_transformer.py
│   ├── train_yolo.py
├── utils/               # Utility functions
│   ├── metrics.py       # Evaluation metrics and visualization tools
├── visualization/       # Visualization tools
│   ├── pointcloud_visualizer.py
```

---

## Core Functions
### 1. **Point Cloud Processing**
- **Voxel Downsampling**: Reduces the size of point cloud data while preserving spatial structure.
- **Projection to Image Plane**: Projects 3D point clouds to 2D image coordinates using calibration matrices.
- **Script**: `pointcloud_preprocessing.py`

### 2. **YOLOv8 Object Detection**
- **Functionality**: Detects objects in images and videos with bounding boxes and confidence scores.
- **Script**: `yolo_model.py`

### 3. **Trajectory Prediction**
- **Functionality**: Predicts future trajectories of objects, including:
  - **Positions**: Future locations in 3D space.
  - **Velocities**: Movement speeds and directions.
  - **Accelerations**: Rate of change in velocity.
- **Script**: `transformer_model.py`

### 4. **3D Visualization**
- **Purpose**: Visualizes point clouds, trajectories, and dynamic vectors in a 3D interactive environment.
- **Script**: `pointcloud_visualizer.py`

---

## GUI Usage
### **Main Features**
1. **Load Data**: Select point cloud, image, or video files for processing.
2. **YOLOv8 Detection**:
   - Load an image or video.
   - Click `YOLOv8 Detection` to visualize object detection results.
3. **Train Transformer**:
   - Provide training data paths through the interface.
   - Click `Train Transformer Model` to train a trajectory prediction model.
4. **Run Trajectory Prediction**:
   - After loading point cloud data, click `Run Transformer Prediction` to compute and display:
     - Future positions.
     - Velocities.
     - Accelerations.
   - The results are shown in the log area and can be visualized in 3D.
5. **3D Visualization**:
   - Click `Start Visualization` to render point clouds and prediction results interactively.

---

## Workflow Example
1. Start the application: `python project_root/gui/main_ui.py`.
2. Load a point cloud file via the "Load Data" button.
3. Train the Transformer model with the "Train Transformer Model" button.
4. Predict future trajectories with the "Run Transformer Prediction" button.
5. Visualize the results with the "Start Visualization" button.

---

## Notes
- Ensure CUDA is properly configured for GPU acceleration.
- Adjust data paths and model save paths as needed for your local environment.
