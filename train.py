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
from typing import List, Tuple

wandb.init(project="girl-classifier", entity="simtoonia")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO)



class Model:
	def __init__(self, config: dict, num_classes: int) -> None:
		self.config = config

	def create_model(self) -> Sequential:
		model = Sequential([
			layers.Rescaling(1./255, input_shape=(config["img_height"], config["img_width"], 3)),
			layers.Conv2D(16, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(),
			layers.Conv2D(32, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(),
			layers.Conv2D(64, 3, padding=config["padding"], activation=config["activation"]),
			layers.MaxPooling2D(),
			layers.Dropout(0.2),
			layers.Flatten(),
			layers.Dense(128, activation=config["activation"]),
			layers.Dense(num_classes)
		])
		print(model.summary())
		return model

def check_data(path: str) -> None:
	pos = list(data_dir.glob(f"{path}pos/*"))
	neg = list(data_dir.glob(f"{path}neg/*"))
	logging.info(f"Found {len(pos)} positive samples")
	logging.info(f"Found {len(neg)} negative samples")
	logging.info(f"Total: {len(pos) + len(neg)} samples")
	PIL.Image.open(str(pos[0])).convert("RGB").resize((224, 224)) # pos class sanity check
	PIL.Image.open(str(neg[0])).convert("RGB").resize((224, 224)) # neg class sanity check

def create_datasets(path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	trainDs = tf.keras.utils.image_dataset_from_directory(
		path,
		validation_split=0.1,
		subset="training",
		seed=69,
		image_size=(config["imgHeight"], config["imgWidth"]),
		batch_size=config["batchSize"]
	)
	valDs = tf.keras.utils.image_dataset_from_directory(
		path,
		validation_split=0.1,
		subset="validation",
		seed=69,
		image_size=(config["imgHeight"], config["imgWidth"]),
		batch_size=config["batchSize"]
	)
	logging.info(f"Found '{trainDs.class_names}' classes")
	logging.info(f"Found {len(trainDs.class_names)} classes")
	plt.figure(figsize=(10, 10))
	for images, labels in trainDs.take(1):
		for i in range(9):
			ax = plt.subplot(3, 3, i + 1)
			plt.imshow(images[i].numpy().astype("uint8"))
			plt.title(trainDs.class_names[labels[i]])
			plt.axis("off")
	return trainDs, valDs

def config_autotune_and_caching(trainDs: tf.data.Dataset, valDs: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
	AUTOTUNE = tf.data.AUTOTUNE
	trainDs = trainDs.cache().sbuffle(1000).prefetch(buffer_size=AUTOTUNE)
	valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE)
	return trainDs, valDs

def normalize(trainDs: tf.data.Dataset, valDs: tf.data.Dataset) -> tf.data.Dataset:
	normLayer = layers.Rescaling(1./255)
	normDs = trainDs.map(lambda x, y: (normLayer(x), y))
	return normDs

def create_augmentation() -> Sequential:
	dataAug = keras.Sequential(
		[
			layers.RandomFlip("horizontal", input_shape=(config["imgHeight"], config["imgWidth"], 3)),
			layers.RandomRotation(0.1),
			layers.RandomZoom(0.1)
		]
	)
	return dataAug


config = {
	"imgHeight": 224,
	"imgWidth": 224,
	"batchSize": 16,
	"padding": "same",
	"activation": "relu",
	"optimizer": "adam",
	"metrics": ["accuracy"],
	"loss": "sparse_categorical_crossentropy",
	"epochs": 15
}

wandb.config.update(config)


if __name__ == "__main__":
	dataDir = pathlib.Path("dir/to/data")
	check_data(dataDir)
	trainDs, valDs = create_datasets(dataDir)
	trainDs, valDs = config_autotune_and_caching(trainDs, valDs)
	normDs = normalize(trainDs, valDs)
	dataAug = create_augmentation()
	model = Model(config, len(trainDs.class_names)).create_model()
	model.compile(optimizer=config["optimizer"], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=config["metrics"])
	ckptCallback = tf.keras.callbacks.ModelCheckpoint(
		filepath="checkpoints/",
		save_weights_only=True,
		verbose=1,
		save_freq="epoch"
	)
	tbCallback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
	trained = model.fit(
		trainDs,
		validation_data=valDs,
		epochs=config["epochs"],
		callbacks=[ckptCallback, tbCallback, WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("checkpoints/")],
	)
	acc = trained.history["accuracy"]
	valAcc = trained.history["val_accuracy"]
	topAcc = max(acc)
	topValAcc = max(valAcc)
	loss = trained.history["loss"]
	valLoss = trained.history["val_loss"]
	minLoss = min(loss)
	minValLoss = min(valLoss)
	wandb.log({"top_accuracy": topAcc, "top_val_accuracy": topValAcc, "min_loss": minLoss, "min_val_loss": minValLoss})
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
	model.save("model.h5")
	artifact = wandb.Artifact("girl-classifier", type="model")
	artifact.add_file("model.h5")
	wandb.log_artifact(artifact)
	model.save_weights("weights.h5")