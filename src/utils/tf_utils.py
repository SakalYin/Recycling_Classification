"""Utility functions to display the pose detection results."""

import cv2
import numpy as np
from tflite_support.task import processor
from tflite_support.task import core
from tflite_support.task import vision
from tflite_runtime.interpreter import Interpreter 

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

DEFAULT_MODEL = 'models/efficientdet_lite0.tflite'
CORAL_MODEL = 'models/efficientdet_lite0_edgetpu.tflite'


def draw_boxes(image, boxes, classes, scores, count, labels, threshold=0.3):
    height, width, _ = image.shape
    for i in range(count):
        if scores[i] < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image

def init_model(enable_edgetpu: bool = False, num_threads: int = 4):

    # enable Coral TPU if it's used
    if enable_edgetpu:
        model = CORAL_MODEL
    else:
        model = DEFAULT_MODEL

    # Initialize the object detection model
    base_options = core.BaseOptions(file_name=model)
    detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        detection_options=detection_options
    )
    detector = vision.ObjectDetector.create_from_options(options)
    
    return detector

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.

    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    return image
