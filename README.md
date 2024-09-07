# Child and Therapist Detection & Tracking System

This project aims to analyze long-duration videos involving children with Autism Spectrum Disorder (ASD) and therapists by detecting and tracking them throughout the video. The system assigns unique IDs, handles occlusions, and tracks re-entries using a combination of object detection (YOLO) and tracking (SORT). This README describes the logic behind the model predictions and provides instructions to easily reproduce the results.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Model and Tracker Explanation](#model-and-tracker-explanation)
  - [YOLO Object Detection](#yolo-object-detection)
  - [SORT Tracking Algorithm](#sort-tracking-algorithm)
  - [Appearance Feature Extraction](#appearance-feature-extraction)
  - [Tracking Logic](#tracking-logic)
  - [Optimizations for Occlusions and Re-Entries](#optimizations-for-occlusions-and-re-entries)
- [Usage](#usage)
  - [Input Format](#input-format)
  - [How to Run the Code](#how-to-run-the-code)
  - [Output Format](#output-format)

## Project Overview
This system analyzes long-duration videos to detect and track children and therapists in therapy sessions. The project uses a trained YOLO model to detect both entities and SORT (Simple Online and Realtime Tracking) to assign unique IDs and track objects across frames. The tracker also handles occlusions and re-identifies objects when they re-enter the frame.

**Key Features:**
- Detects children and therapists using a YOLO model.
- Assigns unique IDs for tracking over time.
- Uses appearance features (e.g., color histograms) for improved tracking accuracy.
- Handles occlusions and object re-entries after leaving the frame.
- Generates bounding boxes with class labels and unique IDs for visualization.

## Requirements
To reproduce the results, ensure you have the following requirements installed:

- Python 3.x
- OpenCV: `pip install opencv-python`
- Numpy: `pip install numpy`
- Ultralytics YOLO: `pip install ultralytics`
- Scipy: `pip install scipy`

You can also install all requirements from a `requirements.txt` file:

```bash
pip install -r requirements.txt


File Structure
The project consists of the following files:

.
├── best.pt                 # Trained YOLO model weights
├── Tracking.py             # Main tracking script
├── Track_AID.ipynb         # Notebook containing Model training script
├── README.md               # This README file
├── test_video.mp4          # Input test video
└── output/                 # Folder for storing output videos


Model and Tracker Explanation
YOLO Object Detection
We use a pre-trained YOLO model for object detection. The model detects two key classes:

Children (Class 0)
Therapists (Class 1)
The YOLO model outputs bounding boxes, confidence scores, and class labels for each detected object in the frame.
results = model(frame, conf=confidence_threshold, iou=0.5)
confidence_threshold: Detections with a confidence score above this threshold (default: 0.7) are considered valid.
iou_threshold: Used to reduce overlapping boxes by filtering out redundant predictions.
SORT Tracking Algorithm
The SORT algorithm (Simple Online and Realtime Tracking) is used to assign unique IDs and track the movement of objects (children and therapists) across frames.

SORT maintains an internal list of KalmanBoxTrackers, each responsible for predicting the position of a tracked object. When a new detection is made, the algorithm matches it to an existing tracker using Intersection over Union (IoU) and updates the tracker. If no matching tracker is found, a new one is created.

Key features of SORT in this project:

Maximum Age: If an object is not detected for a certain number of frames (default: 30), the tracker is deleted.
Minimum Hits: A tracker needs a minimum number of detections (default: 3) before being considered valid.
IoU Threshold: The overlap threshold (default: 0.5) for determining whether a detection corresponds to an existing tracker.
Appearance Feature Extraction
To improve tracking accuracy and re-identification after occlusion, we extract appearance features using color histograms. This provides additional information to distinguish between multiple children or therapists that may be closely positioned.

The appearance feature is calculated for each detection using the color histogram of the detected object's region of interest (ROI):

def extract_appearance_feature(frame, bbox):
    roi = frame[y1:y2, x1:x2]  # Extract region of interest
    feature = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    feature = cv2.normalize(feature, feature).flatten()
    return feature
The feature is then used to match detections to trackers when multiple objects are detected close together.

Tracking Logic
The detection and tracking workflow consists of the following steps:

Frame Processing: Each frame from the video is processed using YOLO for detection.
Non-Maximum Suppression: Low-confidence detections and overlapping boxes are suppressed.
Tracking: SORT assigns each valid detection a unique ID and tracks it across frames.
Appearance Matching: The appearance feature is used to handle situations where objects may occlude each other or leave and re-enter the frame.
Visualization: Bounding boxes with unique IDs and class labels (e.g., "therapist 1") are drawn on the frame.
Optimizations for Occlusions and Re-Entries
To handle occlusions and re-entries:

IoU and Appearance Similarity: When a previously tracked object reappears, its appearance is compared to that of the newly detected object, using a combination of IoU and feature similarity. This helps correctly reassign the ID to the re-entered object.
Maximum Age: Tracks objects even when they are temporarily occluded and re-identifies them upon re-entry if their appearance matches.

Usage
Input Format
The input to the system is a video file containing footage of therapy sessions. The video format can be .mp4, .avi, or any format supported by OpenCV.

How to Run the Code
Ensure that the YOLO model (best.pt) is placed in the correct path. To process a video and generate tracking outputs:
python track_system.py --input /path/to/video.mp4 --output /path/to/output/

Output Format
The system outputs a processed video in .mp4 format, saved in the output/ folder. The video contains:

Bounding boxes around each detected child and therapist.
Unique IDs and class labels displayed above each bounding box.
