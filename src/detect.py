import time
import sys

import cv2
# from picamera2 import Picamera2
from tflite_support.task import vision
import utils.tf_utils as utils
from utils.tf_utils import init_model
from PIL import Image

THREAD_NUM = 4
DISPLAY_WIDTH = 460
DISPLAY_HEIGHT = 460
FPS_POS = (20, 60)
FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_HEIGHT = 1.5
FPS_WEIGHT = 3
FPS_COLOR = (255, 0, 0)
FPS_AVG_FRAME_COUNT = 20


def main():
    detect(csi_camera=False, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, framework='tf', flip=False)
    # torch_detect(csi_camera=False, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, flip=False)


def detect(csi_camera: bool=False, width: int=1280, height: int=720, framework: str='tf', flip: bool=False):
    """
        Continuously run inference on images acquired from the camera.

        Args:
        csi_camera: True/False whether the Raspberry Pi camera module is a CSI Camera (Pi Camera module).
        width: the width of the frame captured from the camera.
        height: the height of the frame captured from the camera.
        num_threads: the number of CPU threads to run the model.
        enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """
    counter, fps = 0, 0
    fps_start_time = time.time()

    # Get image from the camera module
    if csi_camera:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (width, height)
        picam2.preview_configuration.main.format = 'RGB888'
        picam2.preview_configuration.main.align()
        picam2.configure("preview")
        picam2.start()
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize the object detection model
    if framework == 'tf':
        detector = init_model(enable_edgetpu=False, num_threads=THREAD_NUM)
    else:
        sys.exit('ERROR: Unsupported framework. Only TensorFlow Lite is supported.')

    while True:
        if csi_camera:
            image = picam2.capture_array()
        else:
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

        counter += 1
        if flip:
            image = cv2.flip(image, -1)

        # Convert the image from BGR to RGB as required by the TFLite model
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from  the RGB image
        image_tensor = vision.TensorImage.create_from_array(image_RGB)

        # Run object detection estimation using the model
        detections = detector.detect(image_tensor)

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detections)

        # Calculate the FPS
        if counter % FPS_AVG_FRAME_COUNT == 0:
            fps_end_time = time.time()
            fps = FPS_AVG_FRAME_COUNT / (fps_end_time - fps_start_time)
            fps_start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        cv2.putText(image, fps_text,
                    FPS_POS, FPS_FONT, FPS_HEIGHT, FPS_COLOR, FPS_WEIGHT)

        # Stop the program if the 'Q' key is pressed.
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Camera', image)

    if not csi_camera:
        cap.release()

    cv2.destroyAllWindows()
    
    
def torch_detect(csi_camera: bool=False, width: int=360, height: int=360, flip: bool=False):
    """
        Continuously run inference on images acquired from the camera using a PyTorch model.
    """
    from utils.torch_utils import MediumCustomNetWrapper
    import torch

    counter, fps = 0, 0
    fps_start_time = time.time()

    # Get image from the camera module
    if csi_camera:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (width, height)
        picam2.preview_configuration.main.format = 'RGB888'
        picam2.preview_configuration.main.align()
        picam2.configure("preview")
        picam2.start()
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)


    model = MediumCustomNetWrapper(num_classes=6)
    model.load_state_dict(torch.load('models/MediumNet2.pth', map_location=torch.device('cpu')))
    model.eval()

    while True:
        if csi_camera:
            image = picam2.capture_array()
        else:
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

        counter += 1
        if flip:
            image = cv2.flip(image, -1)


        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        output = model.detect(pil_image)
        cv2.putText(image, f'Prediction: {output}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        fps = 1.0 / (time.time() - fps_start_time)
        fps_start_time = time.time()
        fps_text = 'FPS = {:.1f}'.format(fps)
        cv2.putText(image, fps_text,
                    FPS_POS, FPS_FONT, FPS_HEIGHT, FPS_COLOR, FPS_WEIGHT)

        # Stop the program if the 'Q' key is pressed.
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow('Camera', image)

    if not csi_camera:
        cap.release()
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
