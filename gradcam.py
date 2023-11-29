from transformers import ViTFeatureExtractor, ViTForImageClassification
import warnings
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2 as cv
import torch
from typing import List, Callable, Optional
import logging
from face_grab import FaceGrabber

# original borrowed from https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/HuggingFace.ipynb
# thanks @jacobgil
# further mods beyond this commit by @simonSlamka

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module,
                          input_image: Image,
                          method: Callable=GradCAM,
                          threshold: float=0.5):
    with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            grayscale_cam[grayscale_cam < threshold] = 0
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
        return np.hstack(results)
    
    
def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    for i in indices:
        print(f"Predicted class (sorted from most confident) {i}: {model.config.id2label[i]}, confidence: {probabilities[0][i].item()}")

def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0],
                                   14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

faceGrabber = FaceGrabber()

image = Image.open("red2.jpg").convert("RGB")
face = faceGrabber.grab_faces(np.array(image))
if face is not None:
    image = Image.fromarray(face)

img_tensor = transforms.ToTensor()(image)

model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "pos")),
                       ClassifierOutputTarget(category_name_to_index(model, "neg"))]
target_layer_dff = model.vit.layernorm
target_layer_gradcam = model.vit.encoder.layer[-2].output
image_resized = image.resize((224, 224))
tensor_resized = transforms.ToTensor()(image_resized)

dff_image = run_dff_on_image(model=model,
                            target_layer=target_layer_dff,
                            classifier=model.classifier,
                            img_pil=image_resized,
                            img_tensor=tensor_resized,
                            reshape_transform=reshape_transform_vit_huggingface,
                            n_components=3,
                            top_k=3,
                            threshold=0)
cv.namedWindow("DFF Image", cv.WINDOW_KEEPRATIO)
cv.imshow("DFF Image", cv.cvtColor(dff_image, cv.COLOR_BGR2RGB))
cv.resizeWindow("DFF Image", 1200, 1200)
cv.waitKey(0)
cv.destroyAllWindows()
grad_cam_image = run_grad_cam_on_image(model=model,
                                    target_layer=target_layer_gradcam,
                                    targets_for_gradcam=targets_for_gradcam,
                                    input_tensor=tensor_resized,
                                    input_image=image_resized,
                                    reshape_transform=reshape_transform_vit_huggingface,
                                    threshold=0)
cv.namedWindow("Grad-CAM Image", cv.WINDOW_KEEPRATIO)
cv.imshow("Grad-CAM Image", grad_cam_image)
cv.resizeWindow("Grad-CAM Image", 1200, 1200)
cv.waitKey(0)
cv.destroyAllWindows()
print_top_categories(model, tensor_resized)