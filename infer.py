from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
from face_grab import FaceGrabber

faceGrabber = FaceGrabber()
model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
processor = ViTImageProcessor.from_pretrained("ongkn/attraction-classifier")

pipe = pipeline("image-classification", model=model, feature_extractor=processor)

face = Image.open("nnGirl.png")
face = faceGrabber.grab_faces(np.array(face))
face = Image.fromarray(face)
result = pipe(face)

print(result)