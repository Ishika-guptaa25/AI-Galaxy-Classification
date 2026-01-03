"""
Dataset Preparation Script
Downloads and prepares galaxy images for training
"""

import os
import requests
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
BASE_DIR = Path('data')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR = BASE_DIR / 'test'
RAW_DIR = BASE_DIR / 'raw'

CLASSES = ['spiral', 'elliptical', 'irregular', 'lenticular']
TARGET_SIZE = (224, 224)
MIN_IMAGES_PER_CLASS = 100


class DatasetPreparation:
    def __init__(self):
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        print("Setting up directories...")

        for split in [TRAIN_DIR, TEST_DIR]:
            for class_name in CLASSES:
                (split / class_name).mkdir(parents=True, exist_ok=True)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print("✓ Directories created")

    def download_sample_data(self):
        """
        Download sample galaxy images
        Note: Replace with actual data sources
        """
        print("\n" + "=" * 50)
        print("DOWNLOADING SAMPLE DATA")
        print("=" * 50)

        # Sample URLs (replace with actual galaxy image sources)
        sample_urls = {
            'spiral': [
                'https://example.com/spiral1.jpg',
                'https://example.com/spiral2.jpg',
            ],
            'elliptical': [
                'https://example.com/elliptical1.jpg',
            ],
            'irregular': [
                'https://example.com/irregular1.jpg',
            ],
            'lenticular': [
                'https://example.com/lenticular1.jpg',
            ]
        }

        print("\n⚠️  NOTE: This is a template!")
        print("Please add your own image sources or download from:")
        print("  - Galaxy Zoo: https://www.galaxyzoo.org/")
        print("  - NASA Archives: https://hubblesite.org/")
        print("  - SDSS: http://skyserver.sdss.org/")
        print("  - Kaggle Datasets: https://www.kaggle.com/")

        # Placeholder for actual download logic
        # for class_name, urls in sample_urls.items():
        #     for i, url in enumerate(urls):
        #         try:
        #             response = requests.get(url, timeout=10)
        #             if response.status_code == 200:
        #                 save_path = RAW_DIR / class_name / f"{class_name}_{i:04d}.jpg"
        #                 save_path.parent.mkdir(parents=True, exist_ok=True)
        #                 with open(save_path, 'wb') as f:
        #                     f.write(response.content)
        #         except Exception as e:
        #             print(f"Error downloading {url}: {e}")

    def process_images(self, source_dir, quality=95):
        """
        Process images: resize, convert format, validate
        """
        print("\n" + "=" * 50)
        print("PROCESSING IMAGES")
        print("=" * 50)

        stats = {cls: {'processed': 0, 'failed': 0} for cls in CLASSES}

        for class_name in CLASSES:
            class_dir = Path(source_dir) / class_name

            if not class_dir.exists():
                print(f"⚠️  Directory not found: {class_dir}")
                continue

            image_files = list(class_dir.glob('*.*'))
            print(f"\n📁 Processing {class_name}: {len(image_files)} images")

            for img_path in tqdm(image_files, desc=class_name):
                try:
                    # Open and validate image
                    img = Image.open(img_path)

                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)

                    # Save processed image
                    save_path = TRAIN_DIR / class_name / f"{class_name}_{stats[class_name]['processed']:04d}.jpg"
                    img.save(save_path, 'JPEG', quality=quality)

                    stats[class_name]['processed'] += 1

                except Exception as e:
                    stats[class_name]['failed'] += 1
                    print(f"  ✗ Failed to process {img_path.name}: {e}")

        # Print statistics
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        for class_name, stat in stats.items():
            print(f"{class_name:12} - Processed: {stat['processed']:4d} | Failed: {stat['failed']:4d}")

        return stats

    def split_data(self, train_ratio=0.8):
        """Split data into train and test sets"""
        print("\n" + "=" * 50)
        print("SPLITTING DATA")
        print("=" * 50)

        for class_name in CLASSES:
            source_dir = TRAIN_DIR / class_name
            dest_dir = TEST_DIR / class_name

            if not source_dir.exists():
                continue

            images = list(source_dir.glob('*.jpg'))
            num_train = int(len(images) * train_ratio)

            # Shuffle images
            np.random.shuffle(images)

            # Move test images
            test_images = images[num_train:]
            for img_path in test_images:
                shutil.move(str(img_path), str(dest_dir / img_path.name))

            print(f"{class_name:12} - Train: {num_train:4d} | Test: {len(test_images):4d}")

    def validate_dataset(self):
        """Validate the prepared dataset"""
        print("\n" + "=" * 50)
        print("VALIDATING DATASET")
        print("=" * 50)

        valid = True

        for split_name, split_dir in [('Train', TRAIN_DIR), ('Test', TEST_DIR)]:
            print(f"\n{split_name} Set:")

            for class_name in CLASSES:
                class_dir = split_dir / class_name

                if not class_dir.exists():
                    print(f"  ✗ {class_name:12} - Directory not found")
                    valid = False
                    continue

                images = list(class_dir.glob('*.jpg'))
                num_images = len(images)

                if num_images < MIN_IMAGES_PER_CLASS:
                    print(f"  ⚠️  {class_name:12} - Only {num_images:4d} images (minimum: {MIN_IMAGES_PER_CLASS})")
                    valid = False
                else:
                    print(f"  ✓ {class_name:12} - {num_images:4d} images")

        if valid:
            print("\n✓ Dataset is ready for training!")
        else:
            print("\n✗ Dataset needs more images")

        return valid

    def create_sample_images(self):
        """Create sample colored images for testing (when no data available)"""
        print("\n" + "=" * 50)
        print("CREATING SAMPLE IMAGES")
        print("=" * 50)
        print("⚠️  Creating dummy colored images for testing only!")

        colors = {
            'spiral': (100, 150, 255),  # Blue
            'elliptical': (255, 150, 100),  # Orange
            'irregular': (150, 255, 100),  # Green
            'lenticular': (255, 255, 100)  # Yellow
        }

        for class_name in CLASSES:
            for i in range(10):  # Create 10 sample images per class
                # Create colored image with random noise
                img_array = np.ones((224, 224, 3), dtype=np.uint8)
                img_array[:, :] = colors[class_name]

                # Add random noise
                noise = np.random.randint(-30, 30, (224, 224, 3))
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

                # Save
                img = Image.fromarray(img_array)
                save_path = TRAIN_DIR / class_name / f"sample_{i:04d}.jpg"
                img.save(save_path, 'JPEG')

            print(f"  ✓ Created 10 sample images for {class_name}")

        print("\n⚠️  Remember to replace with real galaxy images!")


def main():
    """Main execution"""
    print("=" * 50)
    print("GALAXY DATASET PREPARATION")
    print("=" * 50)

    prep = DatasetPreparation()

    # Menu
    while True:
        print("\n" + "=" * 50)
        print("OPTIONS:")
        print("=" * 50)
        print("1. Download sample data (placeholder)")
        print("2. Process existing images")
        print("3. Split into train/test")
        print("4. Validate dataset")
        print("5. Create dummy sample images (for testing)")
        print("6. Run full pipeline")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == '1':
            prep.download_sample_data()
        elif choice == '2':
            source = input("Enter source directory (default: data/raw): ").strip() or 'data/raw'
            prep.process_images(source)
        elif choice == '3':
            prep.split_data()
        elif choice == '4':
            prep.validate_dataset()
        elif choice == '5':
            prep.create_sample_images()
        elif choice == '6':
            print("\nRunning full pipeline...")
            # prep.download_sample_data()
            prep.create_sample_images()  # Using dummy images for demo
            prep.split_data(train_ratio=0.8)
            prep.validate_dataset()
            print("\n✓ Pipeline complete!")
        elif choice == '0':
            print("\nExiting...")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()