import streamlit as st
# import cv2  # Removed - causes issues in cloud deployments
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO

# ===================== Page Configuration =====================
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide", initial_sidebar_state="expanded")
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) to get predictions from our CNN model")

# ===================== Define CNN Model =====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=1568, out_features=600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=600, out_features=10)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        out = out.view(-1, 1568)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ===================== Load Model =====================
@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    model = CNN()
    
    # Check if model file exists
    try:
        model.load_state_dict(torch.load('CNN_MNIST.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'CNN_MNIST.pth' not found. Please train the model first.")
        return None

# ===================== Preprocessing =====================
def preprocess_image(image, mean_gray=0.1307, stddev_gray=0.3081):
    """Preprocess image for model input"""
    # Resize to 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to grayscale if needed
    image = image.convert('L')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean_gray,), (stddev_gray,))
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor

# ===================== Prediction Function =====================
def predict(image, model):
    """Make prediction on image"""
    if model is None:
        return None
    
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item() * 100, probabilities[0].detach().numpy()

# ===================== Main App =====================
def main():
    # Sidebar
    with st.sidebar:
        st.header("Upload Image")
        image_source = st.radio("Choose image source:", ["Upload File", "Take Screenshot"])
        
        if image_source == "Upload File":
            uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp', 'gif'])
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            # Camera input for screenshots
            camera_image = st.camera_input("Take a picture")
            image = Image.open(camera_image) if camera_image is not None else None
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Main content area
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image")
            
            # Preprocess and display
            image_gray = image.convert('L')
            img_array = np.array(image_gray)
            
            st.subheader("Grayscale & Resized (28x28)")
            if image.size != (28, 28):
                image_resized = image_gray.resize((28, 28), Image.Resampling.LANCZOS)
            else:
                image_resized = image_gray
            st.image(image_resized, caption="Preprocessed Image", width=150)
        
        with col2:
            st.subheader("Prediction Results")
            
            # Make prediction
            prediction, confidence, probabilities = predict(image, model)
            
            if prediction is not None:
                st.metric("Predicted Digit", prediction, f"{confidence:.2f}% confidence")
                
                # Show probability for all digits
                st.subheader("Confidence by Digit")
                prob_data = {str(i): float(probabilities[i]) for i in range(10)}
                st.bar_chart(prob_data)
                
                # Detailed probabilities table
                with st.expander("View detailed probabilities"):
                    for digit, prob in enumerate(probabilities):
                        st.write(f"Digit {digit}: {prob*100:.2f}%")
    else:
        st.info("👈 Upload an image on the left to get started!")
        
        # Show example
        st.subheader("Examples")
        st.write("You can upload images of handwritten digits like:")
        st.write("- Clear digits on white background")
        st.write("- Scanned or photographed handwritten numbers")
        st.write("- For best results, use 28x28 or larger images with black digit on white background")

if __name__ == "__main__":
    main()
