import gradio as gr
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image
import warnings
import logging
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from face_grab import FaceGrabber
from gradcam import GradCam
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
processor = ViTImageProcessor.from_pretrained("ongkn/attraction-classifier")

pipe = pipeline("image-classification", model=model, feature_extractor=processor)

faceGrabber = FaceGrabber()
gradCam = GradCam()

targetsForGradCam = [ClassifierOutputTarget(gradCam.category_name_to_index(model, "pos")),
                       ClassifierOutputTarget(gradCam.category_name_to_index(model, "neg"))]
targetLayerDff = model.vit.layernorm
targetLayerGradCam = model.vit.encoder.layer[-2].output

def classify_image(input):
    face = faceGrabber.grab_faces(np.array(input))
    if face is None:
        return "No face detected", 0, input
    face = Image.fromarray(face)
    faceResized = face.resize((224, 224))
    tensorResized = transforms.ToTensor()(faceResized)
    dffImage = run_dff_on_image(model=model,
                                target_layer=targetLayerDff,
                                classifier=model.classifier,
                                img_pil=faceResized,
                                img_tensor=tensorResized,
                                reshape_transform=gradCam.reshape_transform_vit_huggingface,
                                n_components=5,
                                top_k=10
                                )
    gradCamImage = gradCam.run_grad_cam_on_image(model=model,
                                        target_layer=targetLayerGradCam,
                                        targets_for_gradcam=targetsForGradCam,
                                        input_tensor=tensorResized,
                                        input_image=faceResized,
                                        reshape_transform=gradCam.reshape_transform_vit_huggingface)
    result = pipe(faceResized)
    if result[0]["label"] == "pos" and result[0]["score"] > 0.9 and result[0]["score"] < 0.95:
        return result[0]["label"], result[0]["score"], str("Nice!"), face, dffImage, gradCamImage
    elif result[0]["label"] == "pos" and result[0]["score"] > 0.95:
        return result[0]["label"], result[0]["score"], str("WHOA!!!!"), face, dffImage, gradCamImage
    else:
        return result[0]["label"], result[0]["score"], "Indifferent", face, dffImage, gradCamImage


iface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs=["text", "number", "text", "image", "image", "image"],
    title="Attraction Classifier - subjective",
    description=f"Takes in a (224, 224) image and outputs an attraction class: {'pos', 'neg'}, along with a GradCam/DFF explanation. Face detection, cropping, and resizing are done internally. Uploaded images are not stored by us, but may be stored by HF. Refer to their [privacy policy](https://huggingface.co/privacy) for details.\nAssociated post: https://simtoon.ongakken.com/Projects/Personal/Girl+classifier/desc+-+girl+classifier"
)

iface.launch()