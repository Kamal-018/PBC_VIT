from flask import Flask, request, render_template
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Load your trained PyTorch model and map it to the CPU
model = torch.load('final_model1.pth', map_location=torch.device('cpu'))
model.eval()

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define class mapping (update this based on your model's training)
class_mapping = {
    0: "basophil",
    1: "eosinophil",
    2: "erythroblast",
    3:"ig",
    4:"lymphocyte",
    5:"monocyte",
    6:"neutrophil",
    7:"platelet"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    # Preprocess the image to match the model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            # Preprocess the image and predict
            image = prepare_image(file_path)
            with torch.no_grad():
                predictions = model(image)
                predicted_index = torch.argmax(predictions, dim=1).item()

            # Map the predicted index to a class name
            predicted_class = class_mapping.get(predicted_index, "Unknown class")

            # Clean up uploaded file
            os.remove(file_path)

            return f"Predicted Class: {predicted_class}"
        except Exception as e:
            return f" error occurred during prediction: {str(e)}"
    else:
        return "Cannot Process"

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
