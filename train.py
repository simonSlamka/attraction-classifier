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

wandb.init(project="girl-classifier", entity="ongakken")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO)


def load_data():
    pos = list(data_dir.glob("pos/*"))
    neg = list(data_dir.glob("neg/*"))
    logging.info(f"Found {len(pos)} positive samples")
    logging.info(f"Found {len(neg)} negative samples")
    logging.info(f"Total: {len(pos) + len(neg)} samples")
    PIL.Image.open(str(pos[0])).convert("RGB").resize((224, 224)) # pos class sanity check
    PIL.Image.open(str(neg[0])).convert("RGB").resize((224, 224)) # neg class sanity check
    return pos, neg



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