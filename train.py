from datasets import load_dataset, ClassLabel, Image
from PIL import Image as PILImage
from random import randrange
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback, default_data_collator, AutoConfig
import evaluate
import numpy as np
import wandb
import os
import hashlib
import imagehash
from torchvision.transforms import Normalize, ToTensor, Compose, RandomResizedCrop
import io
import cv2
import logging
import dlib
import gc
from tqdm import tqdm
from termcolor import colored
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from scipy.stats import entropy



# ! TODO: REFACTOR TO MAKE MORE READABLE AND EASIER TO UNDERSTAND

logging.basicConfig(level=logging.WARNING)

wandb.init(project="girl-classifier", entity="simtoonia")

dsDir = "/home/simtoon/smtn_girls_likeOrNot" # dataset dir

for subdir in ["pos", "neg"]:
	subdirPath = os.path.join(dsDir, subdir)
	for root, dirs, files in os.walk(subdirPath):
		for file in files:
			if file.startswith(("train", "test", "val")):
				raise ValueError(f"File {file} starts with a keyword that could confuse split detection")
			if file.startswith("."):
				continue
			if file.endswith((".jpg", ".png", ".jpeg")):
				filePath = os.path.join(root, file)
				try:
					img = PILImage.open(filePath)
					img.verify()
				except (IOError, SyntaxError) as e:
					print(f"{e}: {filePath}")
			else:
				raise ValueError(f"File {file} is not an image")

ds = load_dataset("imagefolder", data_dir=dsDir, split="train") #.cast_column("image", Image(decode=False))
ds = ds.train_test_split(test_size=0.1, seed=69) # split 'em

print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}") # split sanity check

def get_img_hash(path):
	with open(path, "rb") as f:
		return hashlib.md5(f.read()).hexdigest()

def get_dupes_by_hash(dsDir):
	hashes = {}
	dupes = []
	for root, dirs, files in os.walk(dsDir):
		for file in files:
			if file.endswith((".jpg", ".jpeg", ".png")):
				path = os.path.join(root, file)
				hash = get_img_hash(path)
				if hash not in hashes:
					hashes[hash] = path
				else:
					dupes.append((path, hashes[hash]))
	return dupes

def get_img_perceptual_hash(path):
	img = PILImage.open(path)
	return imagehash.phash(img, hash_size=16)

def get_dupes_by_perceptual_hash(dsDir):
	hashes = {}
	dupes = []
	for root, dirs, files in os.walk(dsDir):
		for file in files:
			if file.endswith((".jpg", ".jpeg", ".png")):
				path = os.path.join(root, file)
				hash = get_img_perceptual_hash(path)
				if hash not in hashes:
					hashes[hash] = path
				else:
					dupes.append((path, hashes[hash]))
	return dupes

def calc_mean_shannon(dsDir):
	shannons = []
	for root, dirs, files in os.walk(dsDir):
		for file in files:
			if file.endswith((".jpg", ".jpeg", ".png")):
				path = os.path.join(root, file)
				img = PILImage.open(path)
				shannons.append(entropy(img.histogram()))
	return np.mean(shannons), shannons

for subdir in ["pos", "neg"]:
	subdirPath = os.path.join(dsDir, subdir)
	dupes = get_dupes_by_hash(subdirPath)
	dupes.extend(get_dupes_by_perceptual_hash(subdirPath))
	if len(dupes) > 0:
		dupes = list(set(dupes)) # dedupe dupes LOL
		print(colored(f"Found {len(dupes)} duplicate images in {subdirPath}", "red"))
		for dupe in dupes:
			print(dupe)
		if input("Remove dupes (permanently)? (yes/no): ").lower() == "yes":
			for dupe in dupes:
				if os.path.exists(dupe[0]):
					os.remove(dupe[0])
					print(f"Removed {dupe[0]}")
			print(colored(f"Removed {len(dupes)} duplicate images in {subdirPath}", "green"))
			del dupes
	meanShannon, shannons = calc_mean_shannon(subdirPath)
	print(colored(f"Mean Shannon entropy for {subdirPath}: {meanShannon}", "green"))
	plt.hist(shannons, bins=100)
	# plt.show()


# ds["train"][randrange(0, 2001)]["image"].show() # display a specimen
print(f"{ds['train'].features}")
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

paddingBy = 0.1 # padding by 10%

def grab_faces(inImg, outImg) -> bool:
	"""
	The function `grab_faces` takes an input image, detects faces using various cascades and models,
	crops and saves the detected face if it belongs to the positive class, and returns True if a face is
	successfully detected and saved.

	@param inImg The input image file path. This is the image from which the faces will be detected and
	cropped.
	@param outImg The `outImg` parameter is the output directory where the cropped faces will be saved.

	@return a boolean value. It returns True if a face is detected and successfully saved, and False
	otherwise.
	"""
	img = cv2.imread(inImg) # read image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

	detected = None

	# rotAngles = [0, 90, 180, 270]

	# for angle in rotAngles:
	# 	if angle == 0:
	# 		rotatedImg = img
	# 		rotatedGray = gray
	# 	else:
	# 		pilImg = PILImage.fromarray(img)
	# 		rotatedPilImg = pilImg.rotate(angle)
	# 		rotatedImg = np.array(rotatedPilImg)
	# 		rotatedGray = cv2.cvtColor(rotatedImg, cv2.COLOR_BGR2GRAY)

	if detected is None:
		faces = detector(gray) #rotatedGray) # detect faces
		if len(faces) > 0:
			detected = faces[0]
			detected = (detected.left(), detected.top(), detected.width(), detected.height())

	if detected is None:
		faces = mmod(img) #rotatedImg)
		if len(faces) > 0:
			detected = faces[0]
			detected = (detected.rect.left(), detected.rect.top(), detected.rect.width(), detected.rect.height())

	if detected is None:
		for cascade in cascades:
			cascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)
			faces = cascadeClassifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=6) # detect faces
			if len(faces) > 0:
				detected = faces[0]
				break

			# if detected is not None:
			# 	break

	if "pos" in inImg and detected is not None: # if positive class and face detected
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
		imgH, imgW, _ = img.shape # get image dims
		x = max(0, x - padW)
		y = max(0, y - padH)
		w = min(imgW - x, w + 2 * padW)
		h = min(imgH - y, h + 2 * padH)
		x = max(0, x - (w - detected[2]) // 2) # center the face horizontally
		y = max(0, y - (h - detected[3]) // 2) # center the face vertically
		face = img[y:y+h, x:x+w] # crop face
		path = os.path.basename(inImg) # get filename
		if not os.path.exists(outImg): # sanity check if path itself exists before saving img
			os.makedirs(outImg) # if not, create it
		if os.path.exists(os.path.join(outImg, path)): # sanity check if face already exists
			logging.warning(f"Face already exists in negative class img: {inImg}!!")
			return True # ^ returning True here because if a face already exists, we don't want to central-crop
		else:
			cv2.imwrite(os.path.join(outImg, path), face) # save face
			return True
	else:
		if "pos" in inImg:
			logging.info(f"No face detected in positive class img: {inImg}!!")
			return False
		elif "neg" in inImg:
			return False

	return False

def resize_faces(inFace, outFace):
	"""
	The function `resize_faces` takes an input face image, resizes it to dimensions (224, 224), and
	saves the resized image to the specified output directory.

	@param inFace The parameter "inFace" is the path to the input image file containing the face that
	needs to be resized.
	@param outFace The parameter "outFace" is the output directory where the resized face image will be
	saved.
	"""
	img = cv2.imread(inFace) # read image
	path = os.path.basename(inFace) # get filename
	if os.path.exists(os.path.join(outFace, path)) and cv2.imread(os.path.join(outFace, path)).shape == (224, 224, 3):
		logging.warning(f"Face of correct dims already exists in positive class img: {inFace}!!")
	else:
		img = cv2.resize(img, (224, 224)) # resize image
		os.remove(inFace)
		if (cv2.imwrite(os.path.join(outFace), img)): # save face
			# logging.info(f"Face resized from img: {inFace}!!")
			pass

def central_crop(inImg, outImg):
	"""
	The function `central_crop` takes an input image, crops it to a central square region of size
	(224, 224), and saves the cropped image to an output directory.
	
	@param inImg The input image file path. This is the image that you want to crop.
	@param outImg The parameter "outImg" is the output directory where the cropped image will be saved.
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
	(224, 224) and returns a list of paths to images that do not meet this criteria.

	@param dir The "dir" parameter is the directory path where the images are located. The function will
	iterate through all the files in the directory and its subdirectories, and check if the file is an
	image file with extensions ".jpg", ".jpeg", or ".png". It will then load the image using Open

	@return a list of file paths for images that have dimensions other than (224, 224).
	"""
	mismatched = []

	for root, _, files in os.walk(dir):
		for file in files:
			if file.endswith((".jpg", ".jpeg", ".png")):
				path = os.path.join(root, file)
				img = cv2.imread(path)
				h, w, _ = img.shape
				if h != 224 or w != 224:
					mismatched.append(os.path.join(path))

	return mismatched

totalFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dsDir) for f in filenames if f.endswith((".jpg", ".jpeg", ".png")) and ".faces" not in dp] # grab all files for sanity checking, excluding .faces directory
print(f"Total imgs: {len(totalFiles)}")

posFiles = [f for f in totalFiles if "pos" in f] # grab positive class girls
negFiles = [f for f in totalFiles if "neg" in f] # grab negative class imgs

print(f"Positive imgs: {len(posFiles)} | Negative imgs: {len(negFiles)}") # sanity check

userInput = input("Do you want to extract and resize faces? (yes/no): ")
if userInput.lower() == "yes":
	processedFiles = []
	if os.path.exists("processed.txt"):
		with open("processed.txt", "r") as file:
			processedFiles = file.read().splitlines()
			print(f"Found {len(processedFiles)} processed files")

	for file in tqdm(posFiles, desc="Processing positive imgs"):
		if file not in processedFiles:
			faceDir = os.path.join(dsDir, ".faces", "pos") # positive class face dir
			if not os.path.exists(faceDir): # sanity check if path itself exists before saving img
				os.makedirs(faceDir) # if not, create it
			if grab_faces(file, faceDir): # grab faces
				processedFiles.append(file)

	if input("Also neg? (yes/no): ").lower() == "yes":
		for file in tqdm(negFiles, desc="Processing negative imgs"):
			if file not in processedFiles:
				faceDir = os.path.join(dsDir, ".faces", "neg") # negative class face dir
				destPath = os.path.join(faceDir, os.path.basename(file)) # get dest path
				if not os.path.exists(faceDir):
					os.makedirs(faceDir)
				didWeGrab = grab_faces(file, faceDir) # grab faces
				if didWeGrab:
					processedFiles.append(file)
				# if not didWeGrab: # ! temporarily disabling this check and the central crop to compare results !
				# 	central_crop(file, faceDir) # central crop

totalFaces = [os.path.join(dp, f) for dp, dn, filenames in os.walk(f"{dsDir}/.faces") for f in filenames if f.endswith((".jpg", ".jpeg", ".png"))] # grab all faces for sanity checking
print(f"Total faces: {len(totalFaces)}")
print(colored(f"Positive faces: {len([f for f in totalFaces if 'pos' in f])} | Negative faces: {len([f for f in totalFaces if 'neg' in f])} | Class imbalance: {len([f for f in totalFaces if 'pos' in f]) / len([f for f in totalFaces if 'neg' in f])}", "green"))

if userInput.lower() == "yes":
	for face in tqdm(totalFaces, desc="Resizing faces"):
		if face not in processedFiles:
			resize_faces(face, face) # resize faces to 224x224
			processedFiles.append(face)

	with open("processed.txt", "a") as file:
		for item in processedFiles:
			file.write("%s\n" % item)

if len(totalFaces) < len(totalFiles): # sanity check
	print("Not all faces were grabbed")

mismatched = check_img_dims(f"{dsDir}/.faces") # sanity check
if len(mismatched) > 0:
	print(f"Found {len(mismatched)} mismatched images")
	print(mismatched)
	raise ValueError("Non-(224, 224) images found - !! TRAINING WOULD FAIL, SO ABORTING !!")

faceDs = load_dataset("imagefolder", data_dir=f"{dsDir}/.faces", split="train")
faceDs = faceDs.train_test_split(test_size=0.1, seed=69)
ds = faceDs
del faceDs

gc.collect()

imgProcessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", size=224) # load img processor

def toPNG(image): # convert all jpgs to pngs
	"""
	The function `toPNG` converts an image to PNG format if it is not already in PNG format.

	@param image The parameter "image" is expected to be an instance of the PIL Image class.

	@return the image in PNG format. If the input image is already in PNG format, it will be returned as
	is. If the input image is in a different format (e.g. JPEG), it will be converted to PNG format
	before being returned.
	"""
	if image.format != "PNG" and image.format != "png":
		with io.BytesIO() as f:
			image.save(f, "PNG")
			return PILImage.open(f)
	return image

def transformToPNG(example):
	"""
	The function "transformToPNG" takes an example dictionary as input, converts the image value to PNG
	format using the "toPNG" function, and updates the "image" key in the dictionary with the converted
	image.

	@param example The parameter "example" is a dictionary that contains an "image" key. The value
	associated with the "image" key is expected to be an image object or data that can be converted to a
	PNG format.

	@return the modified "example" dictionary.
	"""
	example["image"] = toPNG(example["image"])
	return example

norm = Normalize(mean=imgProcessor.image_mean, std=imgProcessor.image_std)
size = (
	imgProcessor.size["shortest_edge"]
	if "shortest edge" in imgProcessor.size
	else (imgProcessor.size["height"], imgProcessor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ToTensor(), norm])

def transform(example):
	"""
	The function takes an example dictionary as input, converts the images in the dictionary to RGB
	format, applies some transformations to the images, and returns the modified dictionary.

	@param example The `example` parameter is a dictionary that contains information about an image. It
	has a key "image" which represents the image data. The value of "image" is a list of PIL images.

	@return the modified "example" dictionary.
	"""

	example["pixel_values"] = [_transforms(img.convert("RGB")) for img in example["image"]]
	del example["image"]
	return example

ds = ds.with_transform(transform)

ds.push_to_hub("ongkn/attraction-faces-ds", private=True)

def augment(example):
	example["pixel_values"] = [torch.flip(img, [2]) for img in example["pixel_values"]]
	# example["pixel_values"] = [torch.rot90(img, 1, [2, 3]) for img in example["pixel_values"]]
	return example

# ds = ds.map(augment, num_proc=16)

accuracy = evaluate.load("accuracy")

def collator(x):
	"""
	The function `collator` takes an input `x` and returns a dictionary with some default data
	collation, with an option to enable interpolation of positional encoding.

	@param x The parameter `x` is the input data that you want to collate. It is used as an argument for
	the `default_data_collator` function, which is a function that collates the input data into a
	dictionary format. The resulting dictionary is then returned by the `collator` function

	@return a dictionary.
	"""
	dict = default_data_collator(x)
	# dict["interpolate_pos_encoding"] = True # TODO: enable when using input dims > (224, 224)
	return dict

def compute_metrics(evalPred):
	"""
	The function "compute_metrics" computes the loss between predicted logits and true labels using the
	CrossEntropyLoss function in PyTorch, and also computes the accuracy of the predictions.

	@param evalPred The evalPred parameter is a tuple containing two elements: logits and labels. Logits
	are the predicted values generated by a model, typically before applying a softmax function. Labels
	are the ground truth values or target values for the corresponding inputs.

	@return a dictionary with two key-value pairs. The first key is "loss" and the value is the loss item,
	which is the loss value converted to a Python float. The second key is "accuracy" and the value is the
	accuracy of the predictions.
	"""
	logits, labels = evalPred
	logits = torch.tensor(logits)
	labels = torch.tensor(labels)
	lossFunc = CrossEntropyLoss()
	loss = lossFunc(logits, labels)
	preds = np.argmax(logits, axis=1)
	acc = accuracy.compute(predictions=preds, references=labels)
	return {"loss": loss.item(), "accuracy": acc["accuracy"]}

config = AutoConfig.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(labels), id2label=id2label, label2id=label2id, dropout_rate=0.5) # load config
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config) # load base model

trainingArgs = TrainingArguments(
	output_dir="./out",
	logging_dir="./logs",
	remove_unused_columns=False,
	evaluation_strategy="steps",
	save_strategy="steps",
	learning_rate=5e-5,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	#gradient_accumulation_steps=8, # defaults to 1
	weight_decay=0.02,
	num_train_epochs=10,
	warmup_ratio=0.1,
	lr_scheduler_type="linear", # "polynomial", "constant_with_warmup", "constant", "linear", "cosine"
	seed=69,
	save_steps=150,
	eval_steps=150,
	save_safetensors=True,
	save_total_limit=3,
	logging_steps=20,
	load_best_model_at_end=True,
	metric_for_best_model="loss",
	greater_is_better=False,
	report_to=["wandb", "tensorboard"],
	push_to_hub=True,
	hub_model_id="attraction-classifier"
)

earlyPullOut = EarlyStoppingCallback(
	early_stopping_patience=3,
	#early_stopping_threshold=0.01 # disabled for now
)

trainer = Trainer(
	model=model,
	args=trainingArgs,
	data_collator=collator,
	train_dataset=ds["train"],
	eval_dataset=ds["test"],
	tokenizer=imgProcessor,
	compute_metrics=compute_metrics,
	callbacks=[earlyPullOut]
)

trainer.train()
trainer.push_to_hub()