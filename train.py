import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
import os
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import logging
from typing import Tuple
from huggingface_hub import push_to_hub_keras as push_to_ph
from random import randint


wandb.init(project="girl-classifier", entity="simtoonia") # 😏 init our ... tape

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4" # SHUT UP, TF! I DON'T CARE ABOUT YOUR WARNINGS!

logging.basicConfig(level=logging.INFO) # loggie loggie loggieeeee



class Model: # cue our protagonist, a supersexy model
	def __init__(self, config: dict, numClasses: int) -> None:
		self.config = config # she's got a config
		self.numClasses = numClasses # and she's got a number of classes ... 2, to be exact ... this says how many elements she's gonna spit out

	def create_model(self) -> Sequential:
		model = Sequential([
			layers.Rescaling(1./255, input_shape=(config["imgHeight"], config["imgWidth"], 3)), # (224, 224, 3)
			layers.Conv2D(8, 3, padding=config["padding"]), # padding: same -- no activation 'cause we're mean
			layers.BatchNormalization(), # meanie >:(
			layers.Conv2D(8, 3, padding=config["padding"], activation=config["activation"]),
			layers.Conv2D(16, 3, padding=config["padding"], activation=config["activation"]), # padding: same -- activation: relulu ... "relulu" ... hm, reminds me of ... someone ...
			layers.Dropout(0.2), # a little bit of violence never hurt anyone
			layers.MaxPooling2D(), # 224 -> 112
			layers.Conv2D(16, 3, padding=config["padding"]),
			layers.BatchNormalization(), # gimme some of that sweet, sweet regularization
			layers.Conv2D(32, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(), # 112 -> 56
			layers.Conv2D(32, 3, padding=config["padding"], activation=config["activation"]),
			layers.Conv2D(64, 3, padding=config["padding"]),
			layers.BatchNormalization(), # more regularization, please
			layers.Conv2D(64, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(), # 56 -> 28
			layers.Conv2D(128, 3, padding=config["padding"]),
			layers.BatchNormalization(), # oh, yes!
			layers.Conv2D(128, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(), # 28 -> 14
			layers.Conv2D(128, 3, padding=config["padding"]),
			layers.BatchNormalization(), # AAAAHH!
			layers.Conv2D(128, 3, padding=config["padding"], activation=config["activation"]),
			layers.Dropout(0.5), # rough 'er up a bit more to make 'er spit out some of the ... cells
			layers.Flatten(), # now, grab a rolling pin, look at her intentely, and flatten 'er out so that she's all in one dim
			layers.Dense(196, activation=config["activation"]),
			layers.Dense(156, activation=config["activation"]),
			layers.Dense(128, activation=config["activation"]),
			layers.Dense(64, activation=config["activation"]),
			layers.Dense(32, activation=config["activation"]),
			layers.Dense(16, activation=config["activation"]),
			layers.Dense(8, activation=config["activation"]),
			layers.Dense(4, activation=config["activation"]),
			layers.Dense(self.numClasses, activation="tanh") # finally, we're gonna make 'er spit out 'er answer over the tanh
		])
		print(model.summary())
		return model

def check_data(path: str) -> None:
	pos = list(path.glob("pos/*"))
	neg = list(path.glob("neg/*"))
	logging.info(f"Found {len(pos)} positive samples")
	logging.info(f"Found {len(neg)} negative samples")
	logging.info(f"Total: {len(pos) + len(neg)} samples")
	PIL.Image.open(str(pos[randint(0, len(pos))])).convert("RGB").resize((224, 224)) # pos class sanity check
	PIL.Image.open(str(neg[randint(0, len(neg))])).convert("RGB").resize((224, 224)) # neg class sanity check

def create_datasets(path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	trainDs = tf.keras.utils.image_dataset_from_directory(
		path,
		validation_split=0.1,
		subset="training",
		seed=69, # noice!
		image_size=(config["imgHeight"], config["imgWidth"]),
		batch_size=config["batchSize"]
	)
	valDs = tf.keras.utils.image_dataset_from_directory(
		path,
		validation_split=0.1, # imo, 10% is a good enough share for validation
		subset="validation",
		seed=69, # lol
		image_size=(config["imgHeight"], config["imgWidth"]),
		batch_size=config["batchSize"]
	)
	logging.info(f"Found '{trainDs.class_names}' classes")
	logging.info(f"Found {len(trainDs.class_names)} classes")
	plt.figure(figsize=(20, 20))
	for images, labels in trainDs.take(1):
		for i in range(15):
			plt.subplot(3, 5, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(trainDs.class_names[labels[i]])
			plt.axis("off")
		plt.show()
	return trainDs, valDs

def config_autotune_and_caching(trainDs: tf.data.Dataset, valDs: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	AUTOTUNE = tf.data.AUTOTUNE # not *that* kind of auto tune
	trainDs = trainDs.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # cache and prefetch trainDs to improve performance
	valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE) # cache and prefetch valDs to improve performance
	return trainDs, valDs

def normalize(trainDs: tf.data.Dataset, valDs: tf.data.Dataset) -> tf.data.Dataset:
	normLayer = layers.Rescaling(1./255) # normalize to [0, 1]
	normDs = trainDs.map(lambda x, y: (normLayer(x), y)) # apply normalization layer to trainDs
	return normDs

def create_augmentation() -> Sequential:
	dataAug = keras.Sequential(
		[
			layers.RandomFlip("horizontal", input_shape=(config["imgHeight"], config["imgWidth"], 3)), # flip 'er around and make 'er horizontal
			layers.RandomRotation(0.1), # then, rotate 'er a bit
			layers.RandomZoom(0.1) # zoom 'er in a bit so that we can see 'er parts better
		]
	)
	return dataAug


config = {
	"imgHeight": 224, # decided 'cause one of Google's ViTs uses 224x224
	"imgWidth": 224, # same as above
	"batchSize": 16, # yeah, I'm poor
	"padding": "same", # I want all the convies layers to have the same output size as their input
	"activation": "relu", # Why is relu always so popular at parties? 'cause she always brings positive vibes!
	"optimizer": "adam", # Jensen!
	"metrics": ["accuracy"], # I'm gonna make you scream my na... "accuracy" ... yeah, that's what I meant ... "accuracy" ... definitely ...
	"loss": "sparse_categorical_crossentropy", # kids, cross entropy is just a fancy way of saying the totally not fancy "negative logarithmic likelihood loss"
	"epochs": 20 # started with 20 epochs, but I think I'll need more
}

wandb.config.update(config)


if __name__ == "__main__":
	dataDir = pathlib.Path("/home/simtoon/smtn_girls_likeOrNot/.faces") # hehe, I'm a simp ... you totally don't need to lock your daughter up in a tower to keep her out of my visual field ... totally not ...
	check_data(dataDir)
	trainDs, valDs = create_datasets(dataDir)
	trainDs, valDs = config_autotune_and_caching(trainDs, valDs)
	normDs = normalize(trainDs, valDs)
	dataAug = create_augmentation() # my vision is augmented ... 'cause ... you know ... more data ... more ... vision ... more ... augmentation ... yeah ...
	model = Model(config, 2).create_model()
	model.compile(optimizer=config["optimizer"], loss=config["loss"], metrics=config["metrics"])
	ckptCallback = tf.keras.callbacks.ModelCheckpoint(
		filepath="checkpoints/",
		save_weights_only=True,
		verbose=1,
		save_freq="epoch"
	)
	tbCallback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1) # for diagnosing model performance
	earlyPullOut = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10) # we can't afford to finish if we're not doing well ... we're poor, remember? ... we can't afford to waste time on a kid who's not gonna make it
	trained = model.fit( # try to fit our ... data ... into ... her ... uhm, model ... yeah ...
		trainDs,
		validation_data=valDs,
		epochs=config["epochs"], # think of it as the number of times you're gonna try to ... fit it in ...
		callbacks=[ckptCallback, tbCallback, WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("checkpoints/"), earlyPullOut]
	)
	acc = trained.history["accuracy"]
	valAcc = trained.history["val_accuracy"]
	topAcc = max(acc)
	topValAcc = max(valAcc)
	loss = trained.history["loss"]
	valLoss = trained.history["val_loss"]
	minLoss = min(loss)
	minValLoss = min(valLoss)
	wandb.log({"top_accuracy": topAcc, "top_val_accuracy": topValAcc, "min_loss": minLoss, "min_val_loss": minValLoss}) # sync to W&B
	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(acc, label="Training Accuracy")
	plt.plot(valAcc, label="Validation Accuracy")
	plt.legend(loc="lower right")
	plt.subplot(1, 2, 2)
	plt.plot(loss, label="Training Loss")
	plt.plot(valLoss, label="Validation Loss")
	plt.legend(loc="upper right")
	plt.show()
	model.save("model.keras") # stick a fork in her temporal lobe, connect it to a disk, and extract the model
	artifact = wandb.Artifact("girl-classifier", type="model") # create an artifact object
	artifact.add_file("model.keras") # add the model to the artifact
	wandb.log_artifact(artifact) # spray the artifact all over W&B
	# model.save_weights("weights.keras") # oh, yeah, baby, spray those weights all over me!
	push_to_ph(repo_id="ongkn/attraction-classifier-kerasCNN", log_dir="logs/", model=model) # last, but not least, push to PH