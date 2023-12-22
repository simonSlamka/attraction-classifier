from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
from face_grab import FaceGrabber
import logging

logging.basicConfig(level=logging.INFO)



class AttractionClassifier:
    def __init__(self):
        self.faceGrabber = FaceGrabber()
        self.model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
        self.processor = ViTImageProcessor.from_pretrained("ongkn/attraction-classifier")
        self.pipe = pipeline("image-classification", model=self.model, feature_extractor=self.processor)

    def classify_image(self, image_path, bCentralCrop=False):
        image = Image.open(image_path).convert("RGB")
        if bCentralCrop:
            width, height = image.size
            thirdH = height // 4
            thirdW = width // 6
            startY = thirdH - 40
            endY = startY + thirdH + 175
            startX = thirdW - 40
            endX = startX + thirdW + 700
            image = image.crop((startX, startY, endX, endY))
            image.show()
            breakpoint()
        face = self.faceGrabber.grab_faces(np.array(image))
        if face is None:
            logging.warning("No face detected")
            return None
        face = Image.fromarray(face)
        result = self.pipe(face)
        # face.show()
        return result, face


if __name__ == "__main__":
    attr = AttractionClassifier()
    result, _ = attr.classify_image("Screenshot from 2023-12-18 02-11-31.png")
    print(result[0])