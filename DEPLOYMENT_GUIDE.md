# MNIST Digit Classifier Deployment

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Ensure Your Model is Trained
Make sure you have `CNN_MNIST.pth` in this directory. Run the notebook cells to train the model if you haven't already:
- Run cells 1-9 to load data and define the model
- Run cell 10 to train the model (saves as `CNN_MNIST.pth`)

### Step 3: Run the App
```bash
streamlit run streamlit_app.py
```

This will open the app in your browser at `http://localhost:8501`

## Features

✨ **Upload handwritten digit images** - Support for JPG, PNG, BMP, GIF formats

📊 **Real-time predictions** - Instant classification with confidence scores

📈 **Detailed analytics** - View probability distribution for all 10 digits

📸 **Camera support** - Optionally capture images directly from your webcam

## How to Use

1. **Upload an image** - Use the sidebar to upload a photo of a handwritten digit
2. **View results** - See the prediction and confidence percentage
3. **Analyze probabilities** - Check the bar chart showing confidence for each digit (0-9)

## Troubleshooting

**"Model file not found"** 
- Ensure you've trained the model first by running all cells in the notebook
- The model should be saved as `CNN_MNIST.pth` in the same directory

**Port already in use**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Dependencies issues**
```bash
pip install --upgrade -r requirements.txt
```

## Alternative Deployments

### Option 1: Web API with Flask
```python
from flask import Flask, request, jsonify
import torch
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)
    # ... prediction logic
    return jsonify({'prediction': result})
```

### Option 2: Docker Container
Create a `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

Then build and run:
```bash
docker build -t mnist-classifier .
docker run -p 8501:8501 mnist-classifier
```

### Option 3: Cloud Deployment (HuggingFace Spaces)
1. Push this repository to GitHub
2. Go to https://huggingface.co/spaces
3. Create a new space with Streamlit
4. Connect your GitHub repo
5. Your app will be live!

---

**Model Information:**
- Architecture: CNN with 2 convolutional layers
- Input: 28x28 grayscale images
- Output: Digit classification (0-9)
- Training accuracy: Check the notebook for metrics
