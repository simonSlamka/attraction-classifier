import logging
import cv2 as cv
import numpy as np
import dlib
from typing import Optional

logging.basicConfig(level=logging.INFO)


class FaceGrabber:
    def __init__(self):
        self.cascades = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt_tree.xml"
        ]
        self.detector = dlib.get_frontal_face_detector() # load face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat") # load face predictor
        self.mmod = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat") # load face detector
        self.paddingBy = 0.1 # padding by 10%

    def grab_faces(self, img: np.ndarray, bGray: bool = False) -> Optional[np.ndarray]:

        if bGray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale

        detected = None

        if detected is None:
            faces = self.detector(img) # detect faces
            if len(faces) > 0:
                detected = faces[0]
                detected = (detected.left(), detected.top(), detected.width(), detected.height())
                logging.info("Face detected by dlib")

        if detected is None:
            faces = self.mmod(img)
            if len(faces) > 0:
                detected = faces[0]
                detected = (detected.rect.left(), detected.rect.top(), detected.rect.width(), detected.rect.height())
                logging.info("Face detected by mmod")

        if detected is None:
            for cascade in self.cascades:
                cascadeClassifier = cv.CascadeClassifier(cv.data.haarcascades + cascade)
                faces = cascadeClassifier.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5) # detect faces
                if len(faces) > 0:
                    detected = faces[0]
                    logging.info(f"Face detected by {cascade}")
                    break

        if detected is not None: # if face detected
            x, y, w, h = detected # grab first face
            padW = int(self.paddingBy * w) # get padding width
            padH = int(self.paddingBy * h) # get padding height
            imgH, imgW, _ = img.shape # get image dims
            x = max(0, x - padW)
            y = max(0, y - padH)
            w = min(imgW - x, w + 2 * padW)
            h = min(imgH - y, h + 2 * padH)
            x = max(0, x - (w - detected[2]) // 2) # center the face horizontally
            y = max(0, y - (h - detected[3]) // 2) # center the face vertically
            face = img[y:y+h, x:x+w] # crop face
            return face

        return None