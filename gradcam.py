from pytorch_grad_cam import GradCAM
from pytorch_gram_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_gram_cam.utils.image import show_cam_on_image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import cv2 as cv


model = load_model("model.keras")
last = [model.layer4[-1]]
inputImg = Image.open("test.jpg").convert("RGB").resize((224, 224))
inputImg = np.array(inputImg)
cam = GradCAM(model=model, target_layers=last, use_cuda=True)

targets = ClassifierOutputTarget([0, 1])
grayscaleCam = cam(inputImg, targets)
grayscaleCam = grayscaleCam[0, :]
visualization = show_cam_on_image(inputImg, grayscaleCam, use_rgb=True)