import gradio as gr
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
import cv2 as cv
import dlib
import logging
from typing import Optional


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

    paddingBy = 0.1 # padding by 10%

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert to grayscale

    detected = None

    for cascade in cascades:
        cascadeClassifier = cv.CascadeClassifier(cv.data.haarcascades + cascade)
        faces = cascadeClassifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) # detect faces
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

def classify_image(input):
    face = grab_faces(np.array(input))
    if face is None:
        return "No face detected", 0, input
    face = Image.fromarray(face)
    result = pipe(face)
    return result[0]["label"], result[0]["score"], face

iface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs=["text", "number", "image"],
    title="Attraction Classifier - subjective",
    description=f"Takes in a (224, 224) image and outputs an attraction class: {'pos', 'neg'}. Face detection, cropping, and resizing are done internally. Uploaded images are not stored by us, but may be stored by HF. Refer to their privacy policy for details."
)
iface.launch()