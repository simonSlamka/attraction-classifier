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
from tqdm import tqdm

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

class GradCam():
    def __init__(self):
        pass
    
    def category_name_to_index(self, model, category_name):
        name_to_index = dict((v, k) for k, v in model.config.id2label.items())
        return name_to_index[category_name]
        
    def run_grad_cam_on_image(self, model: torch.nn.Module,
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
        
        
    def get_top_category(self, model, img_tensor, top_k=5):
        logits = model(img_tensor.unsqueeze(0)).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        topIdx = logits.cpu()[0, :].detach().numpy().argsort()[-1]
        topClass = model.config.id2label[topIdx]
        topScore = probabilities[0][topIdx].item()
        return [{"label": topClass, "score": topScore}]

    def reshape_transform_vit_huggingface(self, x):
        activations = x[:, 1:, :]
        activations = activations.view(activations.shape[0],
                                    14, 14, activations.shape[2])
        activations = activations.transpose(2, 3).transpose(1, 2)
        return activations

def process_frame(frame, faceGrabber: FaceGrabber, gradCam: GradCam, model: ViTForImageClassification, target_layer_dff, target_layer_gradcam):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    orig = frame.copy()
    face = faceGrabber.grab_faces(frame)
    # image = Image.open("Screenshot from 2023-12-04 15-09-43.png").convert("RGB")
    # face = faceGrabber.grab_faces(np.array(image))
    # if face is not None:
    #     image = Image.fromarray(face)
    if face is not None:
        frame = Image.fromarray(face)
    else:
        return orig, orig
    img_tensor = transforms.ToTensor()(frame)
    image_resized = frame.resize((224, 224))
    tensor_resized = transforms.ToTensor()(image_resized)
    dff_image = run_dff_on_image(model=model,
                                target_layer=target_layer_dff,
                                classifier=model.classifier,
                                img_pil=image_resized,
                                img_tensor=tensor_resized,
                                reshape_transform=gradCam.reshape_transform_vit_huggingface,
                                n_components=5,
                                top_k=10,
                                threshold=0,
                                output_size=None) #(500, 500))
    res = gradCam.get_top_category(model, tensor_resized)
    cls = res[0]["label"]
    score = res[0]["score"]
    clsIdx = gradCam.category_name_to_index(model, cls)
    clsTarget = ClassifierOutputTarget(clsIdx)
    grad_cam_image = gradCam.run_grad_cam_on_image(model=model,
                                        target_layer=target_layer_gradcam,
                                        targets_for_gradcam=[clsTarget],
                                        input_tensor=tensor_resized,
                                        input_image=image_resized,
                                        reshape_transform=gradCam.reshape_transform_vit_huggingface,
                                        threshold=0)

    dff_image = cv.resize(dff_image, (2500, 700))
    dff_image = cv.cvtColor(dff_image, cv.COLOR_BGR2RGB)

    cv.putText(dff_image, f"Class: {cls} | Score: {score:.4f}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    return dff_image, grad_cam_image


if __name__ == "__main__":

    faceGrabber = FaceGrabber()
    gradCam = GradCam()
    model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
    target_layer_dff = model.vit.layernorm
    target_layer_gradcam = model.vit.encoder.layer[-2].output

    cap = cv.VideoCapture("emi.mp4")
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    framerate = cap.get(cv.CAP_PROP_FPS)
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    writer = cv.VideoWriter("emi_out.mp4", cv.VideoWriter_fourcc(*"mp4v"), framerate, (2500, 700))

    for _ in tqdm(range(int(totalFrames)), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame, faceGrabber, gradCam, model, target_layer_dff, target_layer_gradcam)

        cv.namedWindow("DFF Image", cv.WINDOW_KEEPRATIO)
        cv.imshow("DFF Image", processed[0])
        cv.resizeWindow("DFF Image", 2500, 700)
        cv.waitKey(1)
        writer.write(processed[0])

    cap.release()
    writer.release()
    cv.destroyAllWindows()

    # image = Image.open("Screenshot from 2023-12-04 15-09-43.png").convert("RGB")
    # face = faceGrabber.grab_faces(np.array(image))
    # if face is not None:
    #     image = Image.fromarray(face)

    # img_tensor = transforms.ToTensor()(image)

    # model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
    # # targets_for_gradcam = [ClassifierOutputTarget(gradCam.category_name_to_index(model, "pos")),
    # #                     ClassifierOutputTarget(gradCam.category_name_to_index(model, "neg"))]
    # target_layer_dff = model.vit.layernorm
    # target_layer_gradcam = model.vit.encoder.layer[-2].output
    # image_resized = image.resize((224, 224))
    # tensor_resized = transforms.ToTensor()(image_resized)

    # dff_image = run_dff_on_image(model=model,
    #                             target_layer=target_layer_dff,
    #                             classifier=model.classifier,
    #                             img_pil=image_resized,
    #                             img_tensor=tensor_resized,
    #                             reshape_transform=gradCam.reshape_transform_vit_huggingface,
    #                             n_components=5,
    #                             top_k=10,
    #                             threshold=0,
    #                             output_size=None) #(500, 500))
    # cv.namedWindow("DFF Image", cv.WINDOW_KEEPRATIO)
    # cv.imshow("DFF Image", cv.cvtColor(dff_image, cv.COLOR_BGR2RGB))
    # cv.resizeWindow("DFF Image", 2500, 700)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    # res = gradCam.get_top_category(model, tensor_resized)
    # cls = res[0]["label"]
    # clsIdx = gradCam.category_name_to_index(model, cls)
    # clsTarget = ClassifierOutputTarget(clsIdx)
    # grad_cam_image = gradCam.run_grad_cam_on_image(model=model,
    #                                     target_layer=target_layer_gradcam,
    #                                     targets_for_gradcam=[clsTarget],
    #                                     input_tensor=tensor_resized,
    #                                     input_image=image_resized,
    #                                     reshape_transform=gradCam.reshape_transform_vit_huggingface,
    #                                     threshold=0)
    # cv.namedWindow("Grad-CAM Image", cv.WINDOW_KEEPRATIO)
    # cv.imshow("Grad-CAM Image", grad_cam_image)
    # cv.resizeWindow("Grad-CAM Image", 2000, 1250)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(f"Top class: {gradCam.get_top_category(model, tensor_resized)[0]}")