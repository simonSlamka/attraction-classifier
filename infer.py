from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
import logging
import cv2 as cv
import numpy as np
import dlib
from typing import Optional
from PIL import Image

logging.basicConfig(level=logging.INFO)

def grab_faces(img: np.ndarray) -> Optional[np.ndarray]:
    cascades = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt_tree.xml"
    ]

    detector = dlib.get_frontal_face_detector() # load face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat") # load face predictor
    mmod = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat") # load face detector

    paddingBy = 0.15 # padding by 10%

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale

    detected = None

    for cascade in cascades:
        cascadeClassifier = cv.CascadeClassifier(cv.data.haarcascades + cascade)
        faces = cascadeClassifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # detect faces
        if len(faces) > 0:
            detected = faces[0]
            logging.info(f"Face detected by {cascade}")
            break

    if detected is None:
        faces = detector(gray) # detect faces
        if len(faces) > 0:
            detected = faces[0]
            detected = (detected.left(), detected.top(), detected.width(), detected.height())
            logging.info("Face detected by dlib")

    if detected is None:
        faces = mmod(img)
        if len(faces) > 0:
            detected = faces[0]
            detected = (detected.rect.left(), detected.rect.top(), detected.rect.width(), detected.rect.height())
            logging.info("Face detected by mmod")

    if detected is not None: # if face detected
        x, y, w, h = detected # grab first face
        padW = int(paddingBy * w) # get padding width
        padH = int(paddingBy * h) # get padding height
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

model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
processor = ViTImageProcessor.from_pretrained("ongkn/attraction-classifier")

pipe = pipeline("image-classification", model=model, feature_extractor=processor)

face = Image.open("emitest.jpeg")
face = grab_faces(np.array(face))
face = Image.fromarray(face)
result = pipe(face)

print(result)