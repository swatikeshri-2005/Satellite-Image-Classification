# 🛰️ Satellite Image Classification

A deep learning project that classifies satellite images into different **land-cover categories** using **Transfer Learning with MobileNetV2** and **TensorFlow/Keras**.

The project includes model training, evaluation, prediction, automated testing, and an interactive **Streamlit web application** for real-time satellite image classification.

---

## 📌 Project Overview

Satellite imagery contains valuable information about Earth's land surface. Automatically identifying land-cover types from satellite images can be useful in:

* 🌍 Land-use and land-cover mapping
* 🏙️ Urban planning
* 🌳 Forest monitoring
* 🌾 Agricultural analysis
* 🛰️ Remote sensing
* 🌊 Environmental monitoring
* 🛣️ Infrastructure analysis

This project uses the **EuroSAT RGB dataset** and a pretrained **MobileNetV2** model to classify satellite images into **10 different land-cover classes**.

---

## ✨ Features

* 🧠 Transfer learning using **MobileNetV2**
* 🖼️ RGB satellite image classification
* 🔟 Classification into 10 land-cover categories
* 🔄 Image augmentation during training
* 📊 Model evaluation
* 🔮 Command-line image prediction
* 🌐 Interactive Streamlit web application
* 📈 Class probability visualization
* 🧪 Automated application testing
* 💾 Trained model saved in `.keras` format

---

## 📊 Dataset

This project uses the **EuroSAT RGB dataset**.

### Dataset Statistics

| Property     |              Value |
| ------------ | -----------------: |
| Total Images |             27,000 |
| Image Type   |                RGB |
| Image Size   | 64 × 64 originally |
| Classes      |                 10 |
| Dataset Type |  Satellite imagery |

### Classes

The model classifies images into:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

---

## 🧠 Model Architecture

The project uses **MobileNetV2** with transfer learning.

### Why MobileNetV2?

MobileNetV2 is a lightweight convolutional neural network that provides a good balance between:

* Accuracy
* Training speed
* Computational efficiency
* Model size

The pretrained ImageNet weights are used as the starting point for feature extraction.

### Training Approach

```text
Satellite Images
       │
       ▼
Image Preprocessing
       │
       ▼
Data Augmentation
       │
       ▼
MobileNetV2
(ImageNet Weights)
       │
       ▼
Feature Extraction
       │
       ▼
Classification Layer
       │
       ▼
10 Land-Cover Classes
```

---

## ⚙️ Training Configuration

| Parameter          | Value              |
| ------------------ | ------------------ |
| Framework          | TensorFlow / Keras |
| Base Model         | MobileNetV2        |
| Pretrained Weights | ImageNet           |
| Input Size         | 224 × 224          |
| Channels           | RGB                |
| Batch Size         | 32                 |
| Epochs             | 15                 |
| Optimizer          | Adam               |
| Learning Rate      | 0.0001             |
| Validation Split   | 20%                |
| Random Seed        | 42                 |

The MobileNetV2 backbone is initially frozen while the classification layers are trained.

---

## 📁 Project Structure

```text
Satellite-Image-Classification/
│
├── app.py
├── test_app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── ...
│
├── models/
│   └── satellite_classifier.keras
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── tests/
    └── ...
```

> `test_app.py` is located in the project root, beside `app.py`.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Satellite-Image-Classification.git
cd Satellite-Image-Classification
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If activation is successful, your terminal should show:

```text
(.venv)
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

Run:

```powershell
python -m src.train
```

The training process will:

1. Load the EuroSAT dataset
2. Preprocess the images
3. Apply data augmentation
4. Load MobileNetV2
5. Train the classification layers
6. Validate the model
7. Save the trained model

The trained model is saved as:

```text
models/satellite_classifier.keras
```

---

# 📊 Evaluate the Model

Run:

```powershell
python -m src.evaluate
```

This evaluates the trained classifier on the evaluation data and provides classification performance information.

---

# 🔮 Make a Prediction

You can use the prediction script to classify an individual image.

```powershell
python -m src.predict path\to\image.jpg
```

Example:

```powershell
python -m src.predict sample.jpg
```

The prediction returns the predicted land-cover category and its confidence.

---

# 🌐 Run the Streamlit Application

Start the web application with:

```powershell
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

### Web App Workflow

```text
Upload Satellite Image
          │
          ▼
   Image Preprocessing
          │
          ▼
     MobileNetV2
          │
          ▼
   Model Prediction
          │
          ▼
 ┌──────────────────────┐
 │ Predicted Class      │
 │ Confidence Score     │
 │ Class Probabilities  │
 └──────────────────────┘
```

The application supports:

* JPG
* JPEG
* PNG

---

# 🧪 Testing

The project includes an application test file:

```text
test_app.py
```

It is located beside `app.py` in the project root.

Run the tests using:

```powershell
pytest
```

You can also verify that Streamlit is installed:

```powershell
python -c "import streamlit; print('Streamlit OK')"
```

And verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* MobileNetV2
* Transfer Learning

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Web Application

* Streamlit

### Testing

* Pytest

---

# 📦 Requirements

Typical dependencies include:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
streamlit
pytest
pillow
```

Install them with:

```powershell
pip install -r requirements.txt
```

---

# 🔍 Example Prediction

A satellite image is provided to the application:

```text
        Satellite Image
              │
              ▼
       Image Preprocessing
              │
              ▼
          MobileNetV2
              │
              ▼
        Model Prediction
              │
              ▼
      ┌─────────────────┐
      │ Predicted Class │
      │                 │
      │     Forest      │
      │                 │
      │ Confidence: XX% │
      └─────────────────┘
```

The application also displays the probability distribution across the available classes.

---

# 📈 Future Improvements

Possible improvements include:

* 🔧 Fine-tuning the MobileNetV2 base layers
* 🧠 Experimenting with EfficientNet and ResNet
* 📊 Adding confusion matrix visualization
* 📈 Adding training/validation accuracy graphs
* 🚀 Deploying the Streamlit application
* ⚡ GPU-accelerated training
* 🔍 Adding Grad-CAM explainability
* 📦 Improving model optimization and inference speed
* 🌍 Adding support for additional satellite datasets

---

# 🎯 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Convolutional Neural Networks
* Transfer Learning
* Image Classification
* Data Augmentation
* Model Evaluation
* TensorFlow/Keras
* MobileNetV2
* Streamlit Application Development
* Machine Learning Model Deployment
* Automated Testing

---

# 👩‍💻 Author

**Swati Keshri**

B.Tech Computer Science Engineering Student

### Skills Demonstrated

```text
Python
TensorFlow
Keras
Deep Learning
Computer Vision
Machine Learning
Streamlit
Model Deployment
```

---

## ⭐ Project Highlights

> **Satellite Image Classification using MobileNetV2 Transfer Learning**

A complete end-to-end deep learning project covering:

**Dataset → Preprocessing → Augmentation → Transfer Learning → Training → Evaluation → Prediction → Streamlit Deployment → Testing**

---

## 📄 License

This project is intended for educational and portfolio purposes.
