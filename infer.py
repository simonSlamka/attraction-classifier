from transformers import pipeline

pipe = pipeline("image-classification", model="ongkn/attraction-classifier")

result = pipe("emi2.jpg")
print(result)

# TODO: put in Grad-CAM