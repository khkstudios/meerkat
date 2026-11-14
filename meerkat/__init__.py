"""meerkat: webcam face recognition, tracking and distance measurement."""

import cv2 as _cv2

try:  # OpenCV 5 logs a harmless "Targets are not supported" warning per DNN model
    _cv2.utils.logging.setLogLevel(_cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass
