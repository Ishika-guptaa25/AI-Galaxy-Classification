"""
Model Testing and Evaluation Script
Tests the trained model and generates performance metrics
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from model.model_utils import load_model, predict_galaxy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
MODEL_PATH = 'model/galaxy_classifier.h5'
TEST_DATA_DIR = 'data/test'
IMG_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['Elliptical', 'Irregular', 'Lenticular', 'Spiral']


class ModelTester:
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_model(model_path)
        self.test_dir = Path(TEST_DATA_DIR)
        self.class_names = CLASS_NAMES

    def test_single_image(self, image_path):
        """Test model on a single image"""
        print(f"\nTesting image: {image_path}")
        print("-" * 50)

        result = predict_galaxy(self.model, image_path)

        print(f"Predicted Galaxy Type: {result['galaxy_type']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("\nAll Predictions:")
        for pred in result['all_predictions']:
            print(f"  {pred['class']:12} - {pred['probability'] * 100:.2f}%")

        return result

    def test_directory(self, directory_path):
        """Test model on all images in a directory"""
        print(f"\nTesting directory: {directory_path}")
        print("=" * 50)

        image_paths = list(Path(directory_path).glob('*.jpg'))
        results = []

        for img_path in image_paths:
            try:
                result = predict_galaxy(self.model, str(img_path))
                result['image_path'] = str(img_path)
                results.append(result)
                print(f"✓ {img_path.name:30} -> {result['galaxy_type']:12} ({result['confidence'] * 100:.1f}%)")
            except Exception as e:
                print(f"✗ {img_path.name:30} -> Error: {e}")

        return results

    def evaluate_test_set(self):
        """Evaluate model on entire test set"""
        print("\n" + "=" * 50)
        print("EVALUATING ON TEST SET")
        print("=" * 50)

        if not self.test_dir.exists():
            print(f"Error: Test directory '{self.test_dir}' not found!")
            return None

        # Prepare test data generator
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        # Get predictions
        print("\nGenerating predictions...")
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # True labels
        true_classes = test_generator.classes

        # Calculate metrics
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names,
            digits=4
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        return {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'confusion_matrix': cm,
            'report': report
        }

    def plot_confusion_matrix(self, cm, save_path='model/confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to {save_path}")
        plt.close()

    def plot_sample_predictions(self, num_samples=12, save_path='model/sample_predictions.png'):
        """Plot sample predictions with images"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()

        sample_count = 0

        for class_name in self.class_names:
            class_dir = self.test_dir / class_name.lower()
            if not class_dir.exists():
                continue

            images = list(class_dir.glob('*.jpg'))[:3]  # Get 3 images per class

            for img_path in images:
                if sample_count >= num_samples:
                    break

                # Load and display image
                img = Image.open(img_path)
                axes[sample_count].imshow(img)

                # Make prediction
                result = predict_galaxy(self.model, str(img_path))

                # Set title
                title = f"True: {class_name}\nPred: {result['galaxy_type']}"
                title += f"\nConf: {result['confidence'] * 100:.1f}%"

                color = 'green' if result['galaxy_type'] == class_name else 'red'
                axes[sample_count].set_title(title, fontsize=10, color=color, fontweight='bold')
                axes[sample_count].axis('off')

                sample_count += 1

        # Hide unused subplots
        for i in range(sample_count, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sample predictions saved to {save_path}")
        plt.close()

    def analyze_misclassifications(self, results):
        """Analyze and report misclassifications"""
        print("\n" + "=" * 50)
        print("MISCLASSIFICATION ANALYSIS")
        print("=" * 50)

        true_classes = results['true_classes']
        predicted_classes = results['predicted_classes']

        misclassified = true_classes != predicted_classes
        num_misclassified = np.sum(misclassified)

        print(f"\nTotal misclassifications: {num_misclassified}")

        # Find patterns
        for i, class_name in enumerate(self.class_names):
            class_mask = true_classes == i
            class_misclassified = np.sum(misclassified & class_mask)
            class_total = np.sum(class_mask)

            if class_total > 0:
                error_rate = class_misclassified / class_total * 100
                print(f"{class_name:12} - {class_misclassified}/{class_total} misclassified ({error_rate:.2f}%)")


def main():
    """Main testing pipeline"""
    print("=" * 50)
    print("GALAXY CLASSIFIER - MODEL TESTING")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Error: Model file not found at '{MODEL_PATH}'")
        print("Please train the model first using: python model/train_model.py")
        return

    # Initialize tester
    tester = ModelTester(MODEL_PATH)

    # Menu
    while True:
        print("\n" + "=" * 50)
        print("TEST OPTIONS:")
        print("=" * 50)
        print("1. Test single image")
        print("2. Test directory")
        print("3. Evaluate full test set")
        print("4. Plot confusion matrix")
        print("5. Plot sample predictions")
        print("6. Run full evaluation")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                tester.test_single_image(img_path)
            else:
                print("✗ Image not found!")

        elif choice == '2':
            dir_path = input("Enter directory path: ").strip()
            if os.path.exists(dir_path):
                tester.test_directory(dir_path)
            else:
                print("✗ Directory not found!")

        elif choice == '3':
            results = tester.evaluate_test_set()
            if results:
                tester.analyze_misclassifications(results)

        elif choice == '4':
            results = tester.evaluate_test_set()
            if results:
                tester.plot_confusion_matrix(results['confusion_matrix'])

        elif choice == '5':
            tester.plot_sample_predictions()

        elif choice == '6':
            print("\nRunning full evaluation...")
            results = tester.evaluate_test_set()
            if results:
                tester.plot_confusion_matrix(results['confusion_matrix'])
                tester.plot_sample_predictions()
                tester.analyze_misclassifications(results)
                print("\n✓ Evaluation complete!")

        elif choice == '0':
            print("\nExiting...")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()