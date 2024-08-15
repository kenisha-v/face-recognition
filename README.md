# Face Recognition Project

A simple face recognition system using OpenCV and Python.

## Requirements

- Python 3
- OpenCV
- NumPy

Install dependencies with:

```bash
pip install opencv-python numpy
```

## Usage

### 1. Collect Face Data

Run `face_data_collection.py` to capture face data using your webcam.

```bash
python face_data_collection.py
```

- Enter a name when prompted.
- Press q to stop data collection.
- Data is saved as .npy files in ./data/.




### 2. Recognize Faces

Run `face_recognition.py` to recognize faces in real-time.

```bash
python face_recognition.py

```
- Detected faces will display with their associated name.
- Press q to exit.
