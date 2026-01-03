"""
Utility functions for galaxy classification model
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os

IMG_SIZE = 224
CLASS_NAMES = ['Elliptical', 'Irregular', 'Lenticular', 'Spiral']


def load_model(model_path='model/galaxy_classifier.h5'):
    """Load trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def load_metadata(metadata_path='model/galaxy_classifier_metadata.json'):
    """Load model metadata including class names"""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    return None


def preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Preprocess image for prediction
    Args:
        image_path: Path to image file or file object
        img_size: Target size for image
    Returns:
        Preprocessed image array
    """
    try:
        # Handle both file paths and file objects
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = Image.open(image_path)

        # Convert to RGB (handle grayscale or RGBA)
        img = img.convert('RGB')

        # Resize image
        img = img.resize((img_size, img_size), Image.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


def predict_galaxy(model, image_path, class_names=None):
    """
    Predict galaxy type from image
    Args:
        model: Trained Keras model
        image_path: Path to image or file object
        class_names: List of class names (optional)
    Returns:
        Dictionary with prediction results
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Preprocess image
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]

    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])

    # Create all predictions list
    all_predictions = [
        {
            'class': class_names[i],
            'probability': float(predictions[i])
        }
        for i in range(len(class_names))
    ]

    # Sort by probability
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'galaxy_type': predicted_class,
        'confidence': confidence,
        'all_predictions': all_predictions
    }


def batch_predict(model, image_paths, class_names=None):
    """
    Predict galaxy types for multiple images
    Args:
        model: Trained Keras model
        image_paths: List of image paths
        class_names: List of class names
    Returns:
        List of prediction dictionaries
    """
    results = []
    for img_path in image_paths:
        try:
            result = predict_galaxy(model, img_path, class_names)
            result['image_path'] = img_path
            results.append(result)
        except Exception as e:
            results.append({
                'image_path': img_path,
                'error': str(e)
            })

    return results


def get_model_summary(model):
    """Get readable model summary"""
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate various accuracy metrics
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist()
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing model utilities...")

    # Load model
    try:
        model = load_model('model/galaxy_classifier.h5')
        print("✓ Model loaded successfully")

        # Load metadata
        metadata = load_metadata()
        if metadata:
            print(f"✓ Metadata loaded: {metadata}")

        print("\nModel Summary:")
        print(get_model_summary(model))

    except Exception as e:
        print(f"✗ Error: {e}")