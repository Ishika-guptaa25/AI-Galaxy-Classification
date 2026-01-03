"""
Galaxy Classifier Model Training Script
Trains a CNN model to classify galaxies into 4 categories
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 4
DATA_DIR = 'data/train'
MODEL_SAVE_PATH = 'model/galaxy_classifier.h5'


class GalaxyClassifierModel:
    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = []

    def build_model(self, model_type='mobilenet'):
        """
        Build CNN model for galaxy classification
        Options: 'mobilenet', 'resnet', 'efficientnet', 'custom'
        """
        print(f"Building {model_type} model...")

        if model_type == 'mobilenet':
            base_model = keras.applications.MobileNetV2(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif model_type == 'resnet':
            base_model = keras.applications.ResNet50(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        elif model_type == 'efficientnet':
            base_model = keras.applications.EfficientNetB0(
                input_shape=(self.img_size, self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
        else:  # custom
            base_model = self._build_custom_cnn()

        if model_type != 'custom':
            base_model.trainable = False

            self.model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            self.model = base_model

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        print(f"Model built successfully!")
        self.model.summary()
        return self.model

    def _build_custom_cnn(self):
        """Build custom CNN from scratch"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu',
                          input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def prepare_data(self, data_dir, validation_split=0.2):
        """Prepare training and validation data with augmentation"""
        print("Preparing data generators...")

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        val_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=validation_split
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes found: {self.class_names}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")

        return train_generator, val_generator

    def train(self, train_gen, val_gen, epochs=EPOCHS):
        """Train the model with callbacks"""
        print(f"Starting training for {epochs} epochs...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'model/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        print("Training completed!")
        return self.history

    def evaluate(self, test_gen):
        """Evaluate model on test data"""
        print("Evaluating model...")
        results = self.model.evaluate(test_gen, verbose=1)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        return results

    def plot_training_history(self, save_path='model/training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.close()

    def save_model(self, path=MODEL_SAVE_PATH):
        """Save trained model and metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

        # Save class names
        metadata = {
            'class_names': self.class_names,
            'img_size': self.img_size,
            'num_classes': self.num_classes
        }

        with open(path.replace('.h5', '_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print("Metadata saved")


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("Galaxy Classifier Training Pipeline")
    print("=" * 50)

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please organize your data as:")
        print(f"  {DATA_DIR}/")
        print("    ├── spiral/")
        print("    ├── elliptical/")
        print("    ├── irregular/")
        print("    └── lenticular/")
        return

    # Initialize classifier
    classifier = GalaxyClassifierModel()

    # Build model (choose: 'mobilenet', 'resnet', 'efficientnet', 'custom')
    classifier.build_model(model_type='mobilenet')

    # Prepare data
    train_gen, val_gen = classifier.prepare_data(DATA_DIR)

    # Train model
    classifier.train(train_gen, val_gen, epochs=EPOCHS)

    # Plot training history
    classifier.plot_training_history()

    # Save model
    classifier.save_model()

    print("=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()