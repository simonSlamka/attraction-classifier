from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from random import randrange
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import os
from torchvision.transforms import Normalize, ToTensor, Compose
import io

wandb.init(project="smtn_girls_likeOrNot", entity="simtoonia")

ds = load_dataset("/home/simtoon/smtn_girls_likeOrNot", split="train") # load girls
ds = ds.train_test_split(test_size=0.2, seed=69) # split 'em

print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}") # sanity check

ds["train"][randrange(0, 2001)]["image"].show() # display a specimen

labels = ds["train"].features["label"].names # get labels
label2id, id2label = dict(), dict() # create lookup dicts
for i, label in enumerate(labels):
	label2id[label] = str(i)
	id2label[str(i)] = label

imgProcessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k") # load img processor
collator = DefaultDataCollator() # load collator

def toPNG(image): # convert all jpgs to pngs
	if image.format != "PNG" and image.format != "png":
		with io.BytesIO() as f:
			image.save(f, "PNG")
			return Image.open(f)
	return image

ds["train"] = ds["train"].map(toPNG, remove_columns=["image"])
ds["test"] = ds["test"].map(toPNG, remove_columns=["image"])

# ds["train"]["image"] = [toPNG(image) for image in ds["train"]["image"]]
# ds["test"]["image"] = [toPNG(image) for image in ds["test"]["image"]]

# ds["train"] = ds["train"].map(lambda x: {"image": trainPaths[x["__index__"]]}, remove_columns=["image"])
# ds["test"] = ds["test"].map(lambda x: {"image": testPaths[x["__index__"]]}, remove_columns=["image"])

norm = Normalize(mean=imgProcessor.image_mean, std=imgProcessor.image_std)

_transforms = Compose([ToTensor(), norm])

def transform(example):
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
	num_train_epochs=3,
	warmup_ratio=0.1,
	logging_steps=10,
	load_best_model_at_end=True,
	metric_for_best_model="accuracy"

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
