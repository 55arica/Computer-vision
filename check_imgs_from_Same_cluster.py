from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from PIL import Image


image_path1 = '/content/images.jpeg'
image_path2 = '/content/images.jpeg'


image1 = Image.open(image_path1)
image2 = Image.open(image_path2)


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


inputs1 = processor(image1, return_tensors="pt")
inputs2 = processor(image2, return_tensors="pt")

with torch.no_grad():
    logits1 = model(**inputs1).logits
    logits2 = model(**inputs2).logits

# Get the predicted labels (the class with the highest logit)
predicted_label1 = logits1.argmax(-1).item()
predicted_label2 = logits2.argmax(-1).item()

# Print the predictions for each image
print(f"Predicted label for Image 1: {model.config.id2label[predicted_label1]}")
print(f"Predicted label for Image 2: {model.config.id2label[predicted_label2]}")

# Compare the labels to see if the images belong to the same class
if predicted_label1 == predicted_label2:
    print("Both images are predicted to be the same class!")
else:
    print("The images are predicted to be different classes.")
