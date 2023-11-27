from transformers import pipeline, ViTForImageClassification, ViTFeatureExtractor

model = ViTForImageClassification.from_pretrained("ongkn/attraction-classifier")
featExtractor = ViTFeatureExtractor.from_pretrained("ongkn/attraction-classifier")

pipe = pipeline("image-classification", model=model, feature_extractor=featExtractor)

result = pipe("emi.jpg")
print(result)