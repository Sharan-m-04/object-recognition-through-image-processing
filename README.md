# Object Recognition through Image Processing

## Project Domain
**Computer Graphics**

## Technologies Used
- Python
- OpenCV
- NumPy
- SSD_Mobilenet_v3 Large COCO Dataset

## Description
This project performs object recognition using image processing techniques. The code uses OpenCV's deep learning module to detect and classify objects in an input image. The model is trained on the COCO dataset, which contains a large number of object classes.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Sharan-m-04/object-recognition-through-image-processing.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd object-recognition-through-image-processing
    ```
3. **Install the required libraries:**
    ```bash
    pip install opencv-python numpy
    ```

## Usage
1. **Prepare your environment:**
    - Make sure you have `input.jpg`, `coco.names`, `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, and `frozen_inference_graph.pb` in the project directory.

2. **Run the script:**
    ```bash
    python main.py
    ```
    This will read the input image, process it to recognize objects, and display the result with bounding boxes and labels.

## Contact
For any queries or issues, please contact [Sharan M](mailto:msharan.hnp@gmail.com).
