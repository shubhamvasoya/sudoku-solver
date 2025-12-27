import cv2
import os
from src.model import get_model
from src.vision import recognize_sudoku_and_solve, showImage

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera 0. Trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera 1 either. Exiting.")
            return

    import time
    time.sleep(2)

    cap.set(3, 1280)  
    cap.set(4, 720)

    model = get_model()
    
    model_path = os.path.join("models", "digitRecognition.h5")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_weights(model_path)   

    old_sudoku = None 
    
    print("Model loaded successfully.")
    print("Starting Sudoku Solver. Press 'q' to quit.")
    
    frame_count = 0
    while(True):
        ret, frame = cap.read() 
        if not ret:
            print("Error: Failed to capture frame from camera. Exiting loop.")
            break
        
        try:
            sudoku_frame, old_sudoku = recognize_sudoku_and_solve(frame, model, old_sudoku) 
            showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)
        except Exception as e:
            print(f"\nError during processing: {e}")
            cv2.imshow("Real Time Sudoku Solver", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
