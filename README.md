# AI Galaxy Classification
### Overview

AI Galaxy Classification is a deep learning–based project that classifies galaxy images into four major morphological categories: Spiral, Elliptical, Irregular, and Lenticular.
The system uses a Convolutional Neural Network (CNN) trained on astronomical image data and exposes predictions through a Flask-based REST API. A Next.js frontend is included for future UI integration and deployment.

### Features

Galaxy image classification using CNN

Supports four galaxy classes

- Spiral

- Elliptical

- Irregular

- Lenticular

Image preprocessing and dataset pipeline

Model training with early stopping and learning rate scheduling

REST API for image classification

Modular and scalable project structure

### Project Structure
AI-Galaxy-Classification/
│
├── api/
│   └── classify.py          # Flask API for inference
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── train/               # Processed training data
│   └── test/                # Test images
│
├── model/
│   ├── model_utils.py       # Model architecture & utilities
│   ├── prepare_dataset.py   # Dataset preprocessing
│   ├── train_model.py       # Training pipeline
│   └── galaxy_classifier.h5 # Trained model
│
├── pages/                   # Next.js frontend pages
├── styles/                  # Global CSS
│
├── requirements.txt         # Python dependencies
├── package.json             # Frontend dependencies
├── next.config.js           # Next.js configuration
├── vercel.json              # Deployment configuration
└── README.md

### Tech Stack

#### Backend / ML

Python

TensorFlow / Keras

NumPy

Pandas

Pillow

Flask

Flask-CORS

#### Frontend

Next.js

React

Tailwind CSS

### Installation
1. Clone the Repository
git clone https://github.com/Ishika-guptaa25/AI-Galaxy-Classification.git
cd AI-Galaxy-Classification

2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3. Install Python Dependencies
pip install -r requirements.txt

4. Install Frontend Dependencies
npm install

Dataset Preparation

Download a galaxy image dataset (Galaxy Zoo or similar).

Place images inside:

data/raw/Train_images/
data/raw/train_label.csv


### Prepare dataset:

python model/prepare_dataset.py

### Model Training
python model/train_model.py


After training, the model will be saved as:

model/galaxy_classifier.h5

### Running the API
python api/classify.py


API will be available at:

http://localhost:5000/api/classify

### API Usage

Method: POST

Body: Form-data

image: Image file

### Current Limitations

Limited dataset size affects accuracy

Model trained on resized images only

No real-time frontend upload integration yet

### Future Enhancements

Increase dataset size using Galaxy Zoo full dataset

Use transfer learning (ResNet, EfficientNet, MobileNet)

Improve preprocessing with noise reduction and augmentation

Add confidence calibration and explainability (Grad-CAM)

Integrate frontend image upload and result visualization

Deploy model on cloud (Vercel / AWS / GCP)

Add authentication and rate limiting to API

Support batch image classification

Improve accuracy with ensemble models

### Use Cases

Astronomical image analysis

Educational AI/ML projects

Research experiments in galaxy morphology

Computer vision portfolio project
