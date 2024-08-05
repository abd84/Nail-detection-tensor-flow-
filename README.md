# Nail Detection with TensorFlow

This project uses TensorFlow to detect nails in real-time using a pre-trained model. The project captures video from a webcam, processes each frame to detect nails, and displays the detection results on the screen.

## Requirements

- TensorFlow
- NumPy
- OpenCV
- imutils
- A pre-trained model (Frozen Inference Graph)

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/nail-detection.git
    cd nail-detection
    ```

2. **Install the required packages:**
    ```bash
    pip install tensorflow numpy opencv-python imutils
    ```

3. **Download the pre-trained model:**
    - Place the `frozen_inference_graph.pb` file in the `nail-detection` directory.
    - Ensure you have the `classes.pbtxt` file in the `./record/` directory.

## Running the Code

1. **Run the script:**
    ```bash
    python nail_detection.py
    ```

2. **Quit the application:**
    - Press `q` to stop the webcam stream and close the application.

## Code Explanation

1. **Import Libraries:**
    - `tensorflow`: For loading and running the pre-trained model.
    - `numpy`: For numerical operations.
    - `cv2`: For capturing and processing video frames.
    - `imutils`: For easier handling of video streams.
    - `find_finger` (assumed to be a custom module): For processing images to detect hands.

2. **Arguments:**
    - `model`: Path to the pre-trained model.
    - `labels`: Path to the label map file.
    - `num_classes`: Number of classes in the model.
    - `min_confidence`: Minimum confidence threshold for detections.

3. **Detect Colors:**
    - Detect colors for each class for visualization.

4. **Load the Model:**
    - Load the TensorFlow model into memory.

5. **Start Webcam Stream:**
    - Capture video frames from the webcam.

6. **Process Each Frame:**
    - Flip the frame horizontally for a mirror effect.
    - Detect hands using the `find_finger` module.
    - Convert the frame to the required format for the model.
    - Run the model to get detection results.
    - Draw bounding boxes and labels on the detected nails.
    - Display the output frame.

7. **Terminate the Stream:**
    - Stop the webcam stream and close all OpenCV windows.

