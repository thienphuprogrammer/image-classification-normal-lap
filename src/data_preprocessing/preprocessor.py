import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(150, 150), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
    def preprocess_image(self, image):
        """Apply additional preprocessing to individual images"""
        # Convert to float32
        image = tf.cast(image, tf.float32)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl,a,b))
            image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        return image
    
    def create_data_generators(self):
        """Create enhanced data generators with advanced augmentation"""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=0.2,
            preprocessing_function=self.preprocess_image,
            brightness_range=[0.8, 1.2],
            channel_shift_range=50.0,
            featurewise_center=True,
            featurewise_std_normalization=True
        )
        
        # Validation and test data augmentation (only basic preprocessing)
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=self.preprocess_image,
            featurewise_center=True,
            featurewise_std_normalization=True
        )
        
        # Calculate mean and std for featurewise normalization
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Fit the featurewise normalization
        train_datagen.fit(train_generator)
        test_datagen.fit(train_generator)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        self.validation_generator = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return self.train_generator, self.validation_generator, self.test_generator
    
    def visualize_data_distribution(self):
        """Enhanced visualization of dataset distribution"""
        train_counts = []
        test_counts = []
        
        for class_name in self.class_names:
            train_path = self.data_dir / 'train' / class_name
            test_path = self.data_dir / 'test' / class_name
            train_count = len(list(train_path.glob('*.jpg')))
            test_count = len(list(test_path.glob('*.jpg')))
            train_counts.append(train_count)
            test_counts.append(test_count)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training distribution
        sns.barplot(x=self.class_names, y=train_counts, ax=ax1)
        ax1.set_title('Training Set Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot test distribution
        sns.barplot(x=self.class_names, y=test_counts, ax=ax2)
        ax2.set_title('Test Set Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Images')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print class imbalance statistics
        print("\nClass Distribution Statistics:")
        print("Training Set:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {train_counts[i]} images ({train_counts[i]/sum(train_counts)*100:.1f}%)")
        print("\nTest Set:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: {test_counts[i]} images ({test_counts[i]/sum(test_counts)*100:.1f}%)")
    
    def visualize_sample_images(self, num_samples=5):
        """Enhanced visualization of sample images with augmentations"""
        # Get a batch of training images
        images, labels = next(self.train_generator)
        
        # Create figure for original and augmented images
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f'Original - {self.class_names[np.argmax(labels[i])]}')
            axes[i, 0].axis('off')
            
            # Augmented image
            aug_img = self.train_generator.preprocessing_function(images[i])
            axes[i, 1].imshow(aug_img)
            axes[i, 1].set_title(f'Augmented - {self.class_names[np.argmax(labels[i])]}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def analyze_image_statistics(self):
        """Analyze and visualize image statistics"""
        # Collect statistics
        means = []
        stds = []
        for class_name in self.class_names:
            class_path = self.data_dir / 'train' / class_name
            images = list(class_path.glob('*.jpg'))[:100]  # Sample 100 images per class
            
            class_means = []
            class_stds = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                class_means.append(np.mean(img, axis=(0, 1)))
                class_stds.append(np.std(img, axis=(0, 1)))
            
            means.append(np.mean(class_means, axis=0))
            stds.append(np.mean(class_stds, axis=0))
        
        # Plot statistics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot mean RGB values
        x = np.arange(len(self.class_names))
        width = 0.25
        ax1.bar(x - width, [m[0] for m in means], width, label='R', color='red', alpha=0.7)
        ax1.bar(x, [m[1] for m in means], width, label='G', color='green', alpha=0.7)
        ax1.bar(x + width, [m[2] for m in means], width, label='B', color='blue', alpha=0.7)
        ax1.set_title('Mean RGB Values by Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.legend()
        
        # Plot standard deviations
        ax2.bar(x - width, [s[0] for s in stds], width, label='R', color='red', alpha=0.7)
        ax2.bar(x, [s[1] for s in stds], width, label='G', color='green', alpha=0.7)
        ax2.bar(x + width, [s[2] for s in stds], width, label='B', color='blue', alpha=0.7)
        ax2.set_title('RGB Standard Deviations by Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.show() 