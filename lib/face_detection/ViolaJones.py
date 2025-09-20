# https://realpython.com/traditional-face-detection-python/
# https://www.youtube.com/watch?v=uEJ71VlUmMQ&t=491s
# https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html

# haarcascade files can be found at the OpenCV repository:
# https://github.com/opencv/opencv/blob/master/data/haarcascades/

import os
from pathlib import Path
import logging
# openCV will also help me later on to draw on images and display them in other attempts
import cv2
from .CascadeType import CascadeType
from ..API.FaceDetector import FaceDetector
# i won't use the Menu module here as I want to keep this as a standalone module and seperate from the main menu

file_path = os.path.dirname(os.path.abspath(__file__))

class ViolaJones(FaceDetector):
    logger = logging.getLogger(__name__)
    
    def __init__(self,input_image: str | Path = "", output_image_dir: str | Path = "", cascade_type: CascadeType = CascadeType.FRONTALFACE_DEFAULT):
        self.input_image_path = Path(input_image)
        self.input_image = cv2.imread(self.input_image_path)
        self.output_dir = Path(output_image_dir)
        
        self.cascade_type = cascade_type
        self.logger.debug(f"Initialized ViolaJones with input image: {self.input_image_path}, output directory: {self.output_dir}, cascade type: {self.cascade_type.name}")

    def set_input_image(self,new_input_image: Path):
        self.input_image_path = Path(new_input_image)
        self.input_image = cv2.imread(self.input_image_path)
        self.logger.debug(f"Initialized ViolaJones with input image: {self.input_image_path}, output directory: {self.output_dir}, cascade type: {self.cascade_type.name}")

    def set_cascade_type(self,new_cascade_type: CascadeType):
        self.cascade_type = new_cascade_type
        self.logger.debug(f"Initialized ViolaJones with input image: {self.input_image_path}, output directory: {self.output_dir}, cascade type: {self.cascade_type.name}")

    def detect_faces(self, image_path):
        return cv2.CascadeClassifier(self.cascade_type.value).detectMultiScale(
            cv2.cvtColor(
                self.input_image,
                cv2.COLOR_BGR2GRAY
            )
        )

    def detect_and_draw_faces(self):
        self.detected_faces = self.detect_faces(self.input_image_path)
        
        self.logger.debug(f"Detected {len(self.detected_faces)} faces")
        for (x, y, width, height) in self.detected_faces:
            cv2.rectangle(
                self.input_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )
        return self.detected_faces

    def save_output_image(self,save_in_sub_dir: bool = False):
        if save_in_sub_dir:
            self.output_dir = Path(self.output_dir / f"{self.input_image_path.stem}")
        self.output_dir.mkdir(parents=True,exist_ok=True) 
        self.output_image_path = self.output_dir / f"{self.input_image_path.stem}_viola_jones_output.{self.input_image_path.suffix}"
        cv2.imwrite(self.output_image_path, self.input_image)
        self.logger.info(f"Output image saved to {self.output_image_path}")

    
    def save_output_faces(self, save_in_sub_dir: bool = True):
        if save_in_sub_dir:
            self.output_dir = Path(self.output_dir / f"{self.input_image_path.stem}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i , (x, y, width, height) in enumerate(self.detected_faces):
            cropped_img = self.input_image[y:(y + height), x:(x + width)]
            self.output_image_path = self.output_dir / f"face_{i:0>2}_viola_jones_output.{self.input_image_path.suffix}" 
            cv2.imwrite(self.output_image_path, cropped_img)
            self.logger.info(f"Cropped image saved to {self.output_image_path}")

    def show_output_image(self):
        cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Faces", 300, 700)
        cv2.imshow("Detected Faces", self.input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def list_settings(self):
        print(f"input image path: {self.input_image_path}")
        print(f"output directory: {self.output_dir}")
        print(f"cascade type:     {self.cascade_type.value}")