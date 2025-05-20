import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from tqdm import tqdm
import cv2
from PIL import Image
import random

class DataBalancer:
    def __init__(self, data_dir, target_dir, target_size=(150, 150)):
        """
        Initialize the DataBalancer.
        
        Args:
            data_dir (str): Path to the original dataset directory
            target_dir (str): Path to save the balanced dataset
            target_size (tuple): Target image size (height, width)
        """
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.target_size = target_size
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_class_distribution(self):
        """Analyze the current class distribution in the dataset."""
        class_counts = {}
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
            else:
                class_counts[class_name] = 0
                
        self.logger.info("Current class distribution:")
        for class_name, count in class_counts.items():
            self.logger.info(f"{class_name}: {count} images")
            
        return class_counts
    
    def create_balanced_dataset(self, target_samples_per_class=None, validation_split=0.2):
        """
        Create a balanced dataset with equal number of samples per class.
        
        Args:
            target_samples_per_class (int): Target number of samples per class
            validation_split (float): Ratio of validation split
        """
        # Create target directories
        os.makedirs(self.target_dir, exist_ok=True)
        for split in ['train', 'val']:
            for class_name in self.classes:
                os.makedirs(os.path.join(self.target_dir, split, class_name), exist_ok=True)
        
        # Analyze current distribution
        class_counts = self.analyze_class_distribution()
        
        # Determine target samples per class
        if target_samples_per_class is None:
            target_samples_per_class = min(class_counts.values())
        
        self.logger.info(f"Target samples per class: {target_samples_per_class}")
        
        # Process each class
        for class_name in self.classes:
            self.logger.info(f"Processing class: {class_name}")
            
            # Get all images for the class
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            images = os.listdir(class_path)
            if len(images) < target_samples_per_class:
                # Need to augment
                self._augment_class(class_name, images, target_samples_per_class)
            else:
                # Need to downsample
                selected_images = random.sample(images, target_samples_per_class)
                self._copy_images(class_name, selected_images)
            
            # Split into train and validation
            self._split_train_val(class_name, validation_split)
    
    def _augment_class(self, class_name, images, target_count):
        """
        Augment images for a class to reach target count.
        
        Args:
            class_name (str): Name of the class
            images (list): List of image filenames
            target_count (int): Target number of images
        """
        current_count = len(images)
        augmentations_needed = target_count - current_count
        
        # Calculate augmentation factor
        aug_factor = augmentations_needed // current_count + 1
        
        # Create augmented images
        for img_name in tqdm(images, desc=f"Augmenting {class_name}"):
            img_path = os.path.join(self.data_dir, class_name, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save original image
            self._save_image(img, class_name, f"orig_{img_name}")
            
            # Create augmented versions
            for i in range(aug_factor):
                aug_img = self._apply_augmentation(img)
                self._save_image(aug_img, class_name, f"aug_{i}_{img_name}")
    
    def _apply_augmentation(self, image):
        """
        Apply random augmentation to an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Augmented image
        """
        # Random rotation
        angle = random.uniform(-30, 30)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random contrast adjustment
        contrast = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        return image
    
    def _save_image(self, image, class_name, filename):
        """
        Save an image to the target directory.
        
        Args:
            image (numpy.ndarray): Image to save
            class_name (str): Name of the class
            filename (str): Name of the file
        """
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Save to temporary directory
        temp_path = os.path.join(self.target_dir, 'temp', class_name)
        os.makedirs(temp_path, exist_ok=True)
        cv2.imwrite(os.path.join(temp_path, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def _copy_images(self, class_name, images):
        """
        Copy selected images to the target directory.
        
        Args:
            class_name (str): Name of the class
            images (list): List of image filenames
        """
        for img_name in tqdm(images, desc=f"Copying {class_name}"):
            src_path = os.path.join(self.data_dir, class_name, img_name)
            dst_path = os.path.join(self.target_dir, 'temp', class_name, img_name)
            shutil.copy2(src_path, dst_path)
    
    def _split_train_val(self, class_name, validation_split):
        """
        Split images into training and validation sets.
        
        Args:
            class_name (str): Name of the class
            validation_split (float): Ratio of validation split
        """
        temp_path = os.path.join(self.target_dir, 'temp', class_name)
        if not os.path.exists(temp_path):
            return
            
        images = os.listdir(temp_path)
        train_images, val_images = train_test_split(
            images, test_size=validation_split, random_state=42
        )
        
        # Move images to respective directories
        for img_name in train_images:
            src = os.path.join(temp_path, img_name)
            dst = os.path.join(self.target_dir, 'train', class_name, img_name)
            shutil.move(src, dst)
            
        for img_name in val_images:
            src = os.path.join(temp_path, img_name)
            dst = os.path.join(self.target_dir, 'val', class_name, img_name)
            shutil.move(src, dst)
        
        # Clean up temporary directory
        shutil.rmtree(temp_path)
    
    def verify_balanced_dataset(self):
        """Verify the final distribution of the balanced dataset."""
        self.logger.info("\nVerifying balanced dataset distribution:")
        
        for split in ['train', 'val']:
            self.logger.info(f"\n{split.capitalize()} set distribution:")
            split_path = os.path.join(self.target_dir, split)
            
            for class_name in self.classes:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len(os.listdir(class_path))
                    self.logger.info(f"{class_name}: {count} images") 