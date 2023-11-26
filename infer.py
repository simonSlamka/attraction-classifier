from transformers import pipeline

pipe = pipeline("image-classification", model="ongkn/attraction-classifier")

result = pipe("tf.jpeg")
print(result)