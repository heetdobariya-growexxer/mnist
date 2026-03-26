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

## Cloud Deployment

### Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

**Note**: No `packages.txt` needed since we use `opencv-python-headless` which is self-contained.

### Alternative: Use Headless OpenCV
If you still need OpenCV functionality, use:
```bash
pip install opencv-python-headless
```

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

**OpenCV GUI Error (libGL.so.1)**
- This app now uses headless OpenCV to avoid GUI dependency issues
- If you need full OpenCV, add `packages.txt` to your deployment

**Port already in use**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Dependencies issues**
```bash
pip install --upgrade -r requirements.txt
```

## Files Overview

- `streamlit_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies (uses opencv-python-headless)
- `CNN_MNIST.pth` - Trained model weights
- `DEPLOYMENT_GUIDE.md` - This guide

---

**Model Information:**
- Architecture: CNN with 2 convolutional layers
- Input: 28x28 grayscale images
- Output: Digit classification (0-9)
- Training accuracy: Check the notebook for metrics
