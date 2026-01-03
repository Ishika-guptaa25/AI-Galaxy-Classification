"""
Vercel Serverless Function for Galaxy Classification
This file handles the API endpoint for classifying galaxy images
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global variables
model = None
IMG_SIZE = 224
CLASS_NAMES = ['Elliptical', 'Irregular', 'Lenticular', 'Spiral']

# Galaxy descriptions
GALAXY_INFO = {
    'Spiral': {
        'description': 'Spiral galaxies have rotating disks with spiral arms extending from a central bulge. They contain gas, dust, and active star formation.',
        'examples': 'Milky Way, Andromeda (M31), Whirlpool Galaxy (M51)'
    },
    'Elliptical': {
        'description': 'Elliptical galaxies are smooth, featureless light distributions with elliptical shapes. They contain older stars and little gas.',
        'examples': 'M87, M49, NGC 4472'
    },
    'Irregular': {
        'description': 'Irregular galaxies lack a distinct regular shape and have chaotic appearances. Often result from gravitational interactions.',
        'examples': 'Large Magellanic Cloud, Small Magellanic Cloud, NGC 1427A'
    },
    'Lenticular': {
        'description': 'Lenticular galaxies are disk galaxies without prominent spiral arms. They are intermediate between spiral and elliptical galaxies.',
        'examples': 'NGC 5866, NGC 2787, Spindle Galaxy'
    }
}


def load_model_once():
    """Load model once and cache it in memory"""
    global model

    if model is None:
        try:
            model_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'model',
                'galaxy_classifier.h5'
            )

            print(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    return model


def load_class_names():
    """Load class names from metadata if available"""
    try:
        metadata_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'model',
            'galaxy_classifier_metadata.json'
        )

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('class_names', CLASS_NAMES)
    except:
        pass

    return CLASS_NAMES


def preprocess_image(image_file):
    """
    Preprocess uploaded image for prediction
    Args:
        image_file: File object from request
    Returns:
        Preprocessed numpy array
    """
    try:
        # Read image
        img = Image.open(image_file).convert('RGB')

        # Resize
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")


@app.route('/api/classify', methods=['POST', 'OPTIONS'])
def classify():
    """
    Main classification endpoint
    Accepts: POST request with 'image' file
    Returns: JSON with classification results
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please upload an image file'
            }), 400

        image_file = request.files['image']

        # Check file
        if image_file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'Please select a valid image file'
            }), 400

        # Load model
        model = load_model_once()
        class_names = load_class_names()

        # Preprocess image
        img_array = preprocess_image(image_file)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]

        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # Create all predictions list
        all_predictions = [
            {
                'class': class_names[i],
                'probability': float(predictions[i]),
                'percentage': f"{float(predictions[i]) * 100:.2f}%"
            }
            for i in range(len(class_names))
        ]

        # Sort by probability
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Prepare response
        response = {
            'success': True,
            'galaxy_type': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'all_predictions': all_predictions,
            'galaxy_info': GALAXY_INFO.get(predicted_class, {}),
            'timestamp': str(np.datetime64('now'))
        }

        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({
            'error': 'Invalid image',
            'message': str(ve)
        }), 400

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({
            'error': 'Classification failed',
            'message': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_loaded = model is not None
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'version': '1.0.0'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        class_names = load_class_names()
        return jsonify({
            'classes': class_names,
            'num_classes': len(class_names),
            'input_size': IMG_SIZE,
            'galaxy_types': GALAXY_INFO
        }), 200
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# For local testing
if __name__ == '__main__':
    print("Starting Flask development server...")
    print("API will be available at: http://localhost:5000/api/classify")
    app.run(debug=True, host='0.0.0.0', port=5000)


# For Vercel
def handler(environ, start_response):
    """WSGI handler for Vercel"""
    return app(environ, start_response)