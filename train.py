from datasets import load_dataset
from PIL import Image
from random import randrange
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import os
from torchvision.transforms import Normalize, ToTensor, Compose, RandomResizedCrop
import io
import cv2
import logging
import dlib

logging.basicConfig(level=logging.INFO)

wandb.init(project="girl-classifier", entity="simtoonia")

dsDir = "/home/simtoon/smtn_girls_likeOrNot" # dataset dir

ds = load_dataset(dsDir, split="train") # load girls
ds = ds.train_test_split(test_size=0.2, seed=69) # split 'em

print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}") # sanity check

# ds["train"][randrange(0, 2001)]["image"].show() # display a specimen

labels = ds["train"].features["label"].names # get labels
label2id, id2label = dict(), dict() # create lookup dicts
for i, label in enumerate(labels):
	label2id[label] = str(i)
	id2label[str(i)] = label

cascades = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt_tree.xml"
]

detector = dlib.get_frontal_face_detector() # load face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat") # load face predictor
mmod = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat") # load face detector

paddingBy = 0.05

def grab_faces(inImg, outImg) -> bool:
	img = cv2.imread(inImg) # read image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

	detected = None

	for cascade in cascades:
		cascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
		faces = cascadeClassifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4) # detect faces
		if len(faces) > 0:
			detected = faces[0]
			break

	if detected is None:
		faces = detector(gray) # detect faces
		if len(faces) > 0:
			detected = faces[0]
			detected = (detected.left(), detected.top(), detected.width(), detected.height())

	if detected is None:
		faces = mmod(img)
		if len(faces) > 0:
			detected = faces[0]
			detected = (detected.rect.left(), detected.rect.top(), detected.rect.width(), detected.rect.height())

	if "pos" in inImg and detected is not None: # if positive class and face detected
		x, y, w, h = detected # grab first face
		padW = int(paddingBy * w) # get padding width
		padH = int(paddingBy * h) # get padding height
		x = max(0, x - padW)
		y = max(0, y - padH)
		imgH, imgW, _ = img.shape # get image dims
		w += min(imgW - x, w + 2 * padW)
		h += min(imgH - y, h + 2 * padH)
		face = img[y:y+h, x:x+w] # crop face
		path = os.path.basename(inImg) # get filename
		if not os.path.exists(outImg): # sanity check if path itself exists before saving img
			os.makedirs(outImg) # if not, create it
		if os.path.exists(os.path.join(outImg, path)): # sanity check if face already exists
			logging.warning(f"Face already exists in positive class img: {inImg}!!")
			return False
		else:
			cv2.imwrite(os.path.join(outImg, path), face) # save face
			return True
	elif "neg" in inImg and detected is not None:
		# detect faces
		x, y, w, h = detected # grab first face
		padW = int(paddingBy * w) # get padding width
		padH = int(paddingBy * h) # get padding height
		x = max(0, x - padW)
		y = max(0, y - padH)
		imgH, imgW, _ = img.shape # get image dims
		w += min(imgW - x, w + 2 * padW)
		h += min(imgH - y, h + 2 * padH)
		face = img[y:y+h, x:x+w] # crop face
		path = os.path.basename(inImg) # get filename
		if not os.path.exists(outImg): # sanity check if path itself exists before saving img
			os.makedirs(outImg) # if not, create it
		if os.path.exists(os.path.join(outImg, path)): # sanity check if face already exists
			logging.warning(f"Face already exists in negative class img: {inImg}!!")
			return False
		else:
			cv2.imwrite(os.path.join(outImg, path), face) # save face
			return True
	else:
		if "pos" in inImg:
			logging.error(f"No face detected in positive class img: {inImg}!!")
			return False
		elif "neg" in inImg:
			return False

	return False

def resize_faces(inFace, outFace):
	"""
	The function `resize_faces` takes an input face image, resizes it to a specified size, and saves it
	to an output directory.

	:param inFace: The input face image file path. This is the image that you want to resize
	:param outFace: The parameter "outFace" is the output directory where the resized face images will
	be saved
	"""
	img = cv2.imread(inFace) # read image
	img = cv2.resize(img, (224, 224)) # resize image
	path = os.path.basename(inFace) # get filename
	if os.path.exists(os.path.join(outFace, path)) and cv2.imread(os.path.join(outFace, path)).shape == (224, 224, 3):
		logging.warning(f"Face of correct dims already exists in positive class img: {inFace}!!")
	else:
		os.remove(inFace)
		if (cv2.imwrite(os.path.join(outFace), img)): # save face
			logging.info(f"Face saved from img: {inFace}!!")

def central_crop(inImg, outImg):
	"""
	The function `central_crop` takes an input image, crops it to a central square region of size
	224x224 pixels, and saves the cropped image to an output directory.

	:param inImg: The input image file path. This is the image that you want to crop
	:param outImg: The parameter "outImg" is the output directory where the cropped image will be saved
	"""
	img = cv2.imread(inImg) # read image
	h, w, _ = img.shape # get image dims
	centerY, centerX = h // 2, w // 2 # get center

	startX = centerX - 112 # get start x (224 / 2)
	startY = centerY - 112 # get start y (224 / 2)
	endX = centerX + 112 # get end x (224 / 2)
	endY = centerY + 112 # get end y (224 / 2)

	cropped = img[startY:endY, startX:endX] # crop image

	path = os.path.basename(inImg) # get filename
	if not os.path.exists(outImg): # sanity check if path itself exists before saving img
		os.makedirs(outImg) # if not, create it

	if os.path.exists(os.path.join(outImg, path)): # sanity check if img already exists
		logging.warning(f"Image already exists in negative class img: {inImg}!!")
	else:
		cv2.imwrite(os.path.join(outImg, path), cropped)

def check_img_dims(dir): # sanity check to ensure all the imgs are of the dims (224, 224)
	"""
	The function `check_img_dims` checks if all the images in a given directory have dimensions of
	224x224 pixels and returns a list of paths to images that do not meet this criteria.

	:param dir: The `dir` parameter is the directory path where the images are located
	:return: a list of file paths for images that have dimensions other than (224, 224).
	"""
	mismatched = []

	for root, _, files in os.walk(dir):
		for file in files:
			if file.endswith((".jpg", ".png")):
				path = os.path.join(root, file)
				img = cv2.imread(path)
				h, w, _ = img.shape
				if h != 224 or w != 224:
					mismatched.append(os.path.join(path))

	return mismatched

totalFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dsDir) for f in filenames if f.endswith((".jpg", ".png"))] # grab all files for sanity checking
print(f"Total imgs: {len(totalFiles)}")

posFiles = [f for f in totalFiles if "pos" in f] # grab positive class girls
negFiles = [f for f in totalFiles if "neg" in f] # grab negative class imgs

print(f"Positive faces: {len(posFiles)} | Negative imgs: {len(negFiles)}") # sanity check

# for file in posFiles:
# 	faceDir = os.path.join(dsDir, ".faces", "pos") # positive class face dir
# 	if not os.path.exists(faceDir): # sanity check if path itself exists before saving img
# 		os.makedirs(faceDir) # if not, create it
# 	grab_faces(file, faceDir) # grab faces

# for file in negFiles:
# 	faceDir = os.path.join(dsDir, ".faces", "neg") # negative class face dir
# 	destPath = os.path.join(faceDir, os.path.basename(file)) # get dest path
# 	if not os.path.exists(faceDir):
# 		os.makedirs(faceDir)
# 	didWeGrab = grab_faces(file, faceDir) # grab faces
# 	if not didWeGrab:
# 		central_crop(file, faceDir) # central crop

totalFaces = [os.path.join(dp, f) for dp, dn, filenames in os.walk(f"{dsDir}/.faces") for f in filenames if f.endswith((".jpg", ".png"))] # grab all faces for sanity checking
print(f"Total faces: {len(totalFaces)}")

# for face in totalFaces:
# 	resize_faces(face, face) # resize faces to 224x224

if len(totalFaces) < len(totalFiles): # sanity check
	print("Not all faces were grabbed")

mismatched = check_img_dims(f"{dsDir}/.faces") # sanity check
if len(mismatched) > 0:
	print(f"Found {len(mismatched)} mismatched images")
	print(mismatched)
	raise ValueError("Non-224x224 images found - !! TRAINING WOULD FAIL, SO ABORTING !!")

breakpoint()

imgProcessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k") # load img processor
collator = DefaultDataCollator() # load collator

def toPNG(image): # convert all jpgs to pngs
	"""
	The function `toPNG` converts an image to PNG format if it is not already in PNG format.

	:param image: The parameter "image" is expected to be an instance of the Image class from the Python
	Imaging Library (PIL)
	:return: the image in PNG format. If the input image is already in PNG format, it will be returned
	as is. If the input image is in a different format (e.g. JPEG), it will be converted to PNG format
	before being returned.
	"""
	if image.format != "PNG" and image.format != "png":
		with io.BytesIO() as f:
			image.save(f, "PNG")
			return Image.open(f)
	return image

def transformToPNG(example):
	"""
	The function "transformToPNG" takes an example dictionary as input, converts the image value to PNG
	format using the "toPNG" function, and updates the "image" key in the dictionary with the converted
	image.

	:param example: The parameter "example" is a dictionary that contains an "image" key. The value
	associated with the "image" key is expected to be an image file
	:return: the modified "example" dictionary.
	"""
	example["image"] = toPNG(example["image"])
	return example

# ds["train"] = ds["train"].map(transformToPNG, remove_columns=["image"])
# ds["test"] = ds["test"].map(transformToPNG, remove_columns=["image"])

# ds["train"]["image"] = [toPNG(image) for image in ds["train"]["image"]]
# ds["test"]["image"] = [toPNG(image) for image in ds["test"]["image"]]

# ds["train"] = ds["train"].map(lambda x: {"image": trainPaths[x["__index__"]]}, remove_columns=["image"])
# ds["test"] = ds["test"].map(lambda x: {"image": testPaths[x["__index__"]]}, remove_columns=["image"])

norm = Normalize(mean=imgProcessor.image_mean, std=imgProcessor.image_std)
size = (
	imgProcessor.size["shortest_edge"]
	if "shortest edge" in imgProcessor.size
	else (imgProcessor.size["height"], imgProcessor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ToTensor(), norm])

def transform(example):
	"""
	The function takes an example dictionary, converts the images to RGB format, applies a
	transformation to each image, and returns the modified example dictionary.

	:param example: A dictionary containing the following keys:
	:return: the modified "example" dictionary.
	"""
	# image = Image.open(example["image"])
	# image = image.convert("RGB")
	# image = imgProcessor(image, return_tensors="pt")
	# example["pixel_values"] = image.pixel_values[0]
	# return example

	example["pixel_values"] = [_transforms(img.convert("RGB")) for img in example["image"]]
	del example["image"]
	return example

# ds["train"] = ds["train"].map(transform, batched=True)
# ds["test"] = ds["test"].map(transform, batched=True)

ds = ds.with_transform(transform)

accuracy = evaluate.load("accuracy")

def compute_metrics(evalPred):
	"""
	The function `compute_metrics` takes in a tuple `evalPred` containing predicted values and labels,
	and returns the accuracy of the predictions.

	:param evalPred: The evalPred parameter is a tuple containing two elements: preds and labels
	:return: the result of the accuracy computation.
	"""
	preds, labels = evalPred
	preds = np.argmax(preds, axis=1)

	return accuracy.compute(predictions=preds, references=labels)

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(labels), id2label=id2label, label2id=label2id) # load base model

trainingArgs = TrainingArguments(
	output_dir="./out",
	remove_unused_columns=False,
	evaluation_strategy="epoch",
	save_strategy="epoch",
	learning_rate=5e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	gradient_accumulation_steps=4,
	weight_decay=0.01,
	num_train_epochs=10,
	warmup_ratio=0.1,
	seed=69,
	logging_steps=25,
	load_best_model_at_end=True,
	metric_for_best_model="accuracy",
	push_to_hub=True,
	hub_model_id="attraction-classifier"
)

trainer = Trainer(
	model=model,
	args=trainingArgs,
	data_collator=collator,
	train_dataset=ds["train"],
	eval_dataset=ds["test"],
	tokenizer=imgProcessor,
	compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub()