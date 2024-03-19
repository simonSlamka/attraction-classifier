from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
from face_grab import FaceGrabber
import logging
from torch.nn.functional import softmax
from torchvision import transforms


logging.basicConfig(level=logging.INFO)



class AttractionClassifier:
    def __init__(self):
        self.faceGrabber = FaceGrabber()
        self.model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
        self.processor = ViTImageProcessor.from_pretrained("ongkn/attraction-classifier")

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
        face = face.resize((224, 224))
        face = transforms.ToTensor()(face)
        # face.show()
        logits = self.model(face.unsqueeze(0)).logits
        probs = softmax(logits, dim=1)
        topIdx = logits.cpu()[0, :].detach().numpy().argsort()[-1]
        topClass = self.model.config.id2label[topIdx]
        topScore = probs[0, topIdx].item()
        result = [{"label": topClass, "score": topScore}]
        return result, face


if __name__ == "__main__":
    attr = AttractionClassifier()
    result, _ = attr.classify_image("faten.png")
    print(result[0])