![Python](https://img.shields.io/badge/Language-Python_3.11-blue)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-red)
![MediaPipe](https://img.shields.io/badge/Framework-MediaPipe-green)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-yellow)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange)

# Installation & Requirements

To prevent dependency conflicts, it is highly recommended to execute each module within its own Python Virtual Environment (venv).

**General Setup**:

1. **Clone** this repository.
```bash
   git clone https://github.com/Quercius/automotive-adas-dms-suite.git
``` 
2. **Navigate** to the desired assignment folder, for example:
```bash
   cd 01_Route_Planning
```
3. Create and activate a **virtual environment**.
```bash
  python -m venv venv
  venv\Scripts\activate
```
4. **Install dependencies**:
```bash
   pip install -r requirements.txt
```
5. **Execute** the specific module. Refer to the *Execution* instruction in the sections above for the exact run command. For example:
```bash
    python Eval_algorithms.py
```

# Autonomous Driving & Driver Monitoring Systems

This repository contains a collection of three distinct computer vision and path-planning projects developed for autonomous driving and driver safety applications. The projects are divided into three main modules: Route Planning, Lane & Obstacle Detection, and a Driver Monitoring System.

---

## 01. Route Planning (`01_Route_Planning`)

This module evaluates and compares the performance of pathfinding algorithms on real-world OpenStreetMap data, specifically the cities of Aosta and Turin.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1fdd2291-23b3-4561-8d0b-32649c9a9c0c" alt="Gameplay Screenshot 2" width="50%">
  <img src="https://github.com/user-attachments/assets/9fe3f671-1756-47ea-9c02-1c578e08e646" alt="Aosta Path Routing" width="41.7%">
</p>


* **Algorithms Suite**: Features a comparative analysis between Dijkstra's algorithm and multiple variants of the A* algorithm driven by three distinct geographic heuristics: Manhattan, Euclidean, and Haversine distances.

* **Performance & Heuristic Tightness**: Incorporates velocity-homogenized cost functions (converting distance metrics to time estimates in seconds). By introducing a calibrated urban velocity scaling factor (40 km/h) , the Haversine heuristic achieves a mathematical sweet spot—drastically shrinking the exploration space (up to an 8.5x computational reduction in dense networks like Turin) while ensuring absolute path-planning optimality.
  
* **Execution**: To evaluate and compare the average number of iterations for the implemented algorithms, run the `eval_algorithms.py` script.

    ```bash

    python Eval_algorithms.py

    ```

    The standalone scripts `Astar.py` and `Dijkstra.py` can be run independently.

---

## 02. Lane & Obstacle Detection (`02_Lane_Obstacle_Detection`)

> ⚠️ **DATASET DEPENDENCY NOTE** ⚠️
> 
> To successfully run this algorithm, you **must** download the external **PandaSet Dataset** (specifically the front-facing camera frames), as it is too large to be hosted in this repository. 
> 
> 1. Request or download the dataset from the official [Kaggle PandaSet Dataset](https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset).
> 2. Extract the downloaded archive.
> 3. Place the `PandaSetSensorData` folder directly in the root directory of this repository (as shown in the Repository Structure above), ensuring the path `PandaSetSensorData/archive/044/Camera/front_camera/` contains the target `.jpg` images.


<div align="center">
  <img width="80%" alt="Lane Detection" src="https://github.com/user-attachments/assets/8b79ebe5-8afb-498d-bb29-6f41f0f2bc14" />
</div>


This section details the logical and mathematical architecture of the software, following the fundamental steps of the GOLD (Generic Obstacle and Lane Detection) algorithm.


* **Bird's Eye View (BEV)**: The perspective transformation removes the camera's foreshortening effect to obtain a top-down view where lane delimiters appear parallel. 

* **Lane Enhancement**: A spatial kernel acts as a Ridge Detector, responding positively to narrow, bright peaks while zeroing out areas of constant brightness.

* **Iterative Binarization & Peak Detection**: The enhanced image is converted into a binary format using an adaptive thresholding algorithm. The binary image is projected onto the X-axis by summing the white pixel values along each vertical column to locate lane positions.

* **Advanced Features**: The algorithm distinguishes between continuous and dashed lane delimiters by analyzing the vertical distribution of white pixels. The detected lanes are projected back from the Bird's Eye View to the perspective of the original camera.

* **YOLOv8 Integration & Obstacle Masking**: The YOLOv8 model is utilized to detect objects such as cars, trucks, and persons in the original frame. For every detected vehicle, a polygonal mask is generated based on its bounding box and projected into the BEV-binarized image. Before the histogram calculation, these areas are filled with black pixels, ensuring that the histogram peaks correspond only to actual road markings.

<img width="1725" height="531" alt="Image" src="https://github.com/user-attachments/assets/9472d646-2902-4e86-802f-1999e39a8563" />

* **Execution**: The script must be launched by passing the target dataset path as an argument.

    ```bash

    python run_gold.py "PandaSetSensorData/archive/044/Camera/front_camera/*.jpg"

    ```

*Note:* Feel free to replace *044* with your preferred dataset folder number. Also, appending *debug* at hte end of the command will trigger a 2x2 diagnostic grid showing the internal algorithm steps.

---

## 03. Driver Monitoring System (`03_Driver_Monitoring`)

The `runDMS.py` is the main module of the system, responsible for video acquisition, facial landmark extraction, and data distribution to the core modules.

* **Distraction Monitoring**: The module encapsulates geometric calculations to detect the driver's focus level. 

    * **Sleep & Microsleep**: The module calculates the Eye Aspect Ratio (EAR) for both eyes.

    * **Owl Distraction**: The orientation of the head is estimated by calculating the ratio of the distance between the nose tip and the left/right edges of the face.

    * **Lizard Distraction**: Evaluates the horizontal position of the iris center relative to the inner and outer corners of the eye to determine if the gaze is shifted.

    * **Temporal Logic**: The module utilizes a deque structure as a 30-second sliding window to store the duration of isolated distraction events. Any active warning is locked until the driver returns to a Focused state for a continuous period of more than 2 seconds.

* **Remote Photoplethysmography (rPPG)**: This module implements the complete pipeline for estimating the driver's heart rate directly from the webcam feed. 

    * The module isolates the skin pixels by defining a Region of Interest (ROI) on the driver's forehead.

    * The core signal separation is performed using `sklearn.decomposition.FastICA` with the deflation approach and a cubic non-linearity function. 

    * A second-order Butterworth bandpass filter is applied to strictly isolate plausible human heart rate frequencies, followed by a Fast Fourier Transform (FFT).

* **Execution**: The script must be launched using the standard command `python runDMS.py`.

    ```bash

    python runDMS.py

    ```

    It is possible to append the `debug` or `--verbose` keyword at the end of the command to show more internal algorithm metrics. 

    *Note: The system has been specifically realized and tested on Python 3.11.9 using mediapipe==0.10.20, as the MediaPipe package exhibits known compatibility issues with newer Python versions.*
