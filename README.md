# Real-Time Sudoku Solver

This project is a computer vision application that solves Sudoku puzzles in real-time using a webcam. It detects the Sudoku grid from a video feed, recognizes the digits using a Deep Learning model (CNN), solves the puzzle algorithmically, and overlays the solution back onto the original image using Augmented Reality techniques.

## Features

-   **Real-Time Detection**: Instantly detects Sudoku grids in live video streams.
-   **Deep Learning Recognition**: Uses a Convolutional Neural Network (CNN) to accurately recognize handwritten or printed digits.
-   **Augmented Reality Overlay**: Projects the solved numbers directly onto the paper grid in perfect perspective.
-   **GPU Acceleration**: Optimized to run on NVIDIA GPUs using CUDA and cuDNN for smooth performance.


## Installation & Setup

This project requires **Python 3.10** and an NVIDIA GPU for optimal performance.

### 1. Create a Conda Environment
We recommend using Conda to manage Python and CUDA dependencies easily.

```bash
conda create -n sudoku python=3.10
conda activate sudoku
```

### 2. Install CUDA and cuDNN
To enable GPU support for TensorFlow 2.10, install the required libraries from `conda-forge`:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### 3. Install Python Dependencies
Install the remaining Python packages using pip:

```bash
pip install -r requirements.txt
```

*Note: The `requirements.txt` is pinned to TensorFlow 2.10.1, which is the last version to support native Windows GPU acceleration.*

## Usage

1.  Ensure your Conda environment is active:
    ```bash
    conda activate sudoku
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  Hold a Sudoku puzzle in front of your webcam. Ensure the lighting is good and the grid is clearly visible.
4.  The solution will appear overlaid on the grid.
5.  Press **`q`** to quit the application.

## How It Works

1.  **Image Pre-processing**: The camera feed is converted to grayscale, blurred, and thresholded to binary.
2.  **Contour Detection**: OpenCV finds contours to identify the largest square, which is assumed to be the Sudoku grid.
3.  **Perspective Transform**: The grid is "flattened" into a top-down view.
4.  **Digit Extraction**: The grid is sliced into 81 cells. Each cell is processed to extract the digit (if present).
5.  **Digit Recognition**: A clean 28x28 pixel image of each cell is fed into the CNN model. The model is trained on the standard **MNIST dataset** (with '0' removed), which works smoothly and effectively for this task.
6.  **Solving**: The 9x9 digital representation of the board is passed to a backtracking algorithm (Best-First Search) to find the solution.
7.  **Overlay**: The missing numbers are drawn onto a blank image, which is then warped back to the original perspective and merged with the camera feed.