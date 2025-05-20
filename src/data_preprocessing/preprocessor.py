import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

class DataPreprocessor:
    """Professional data preprocessor for image classification tasks.
    
    This class handles all aspects of data preprocessing including:
    - Advanced data augmentation
    - Normalization and standardization
    - Class weighting for imbalanced datasets
    - Data caching and prefetching for optimal performance
    - Data visualization and analysis
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, cache_data=True, mixup_alpha=0.2):
        """Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing the dataset
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            cache_data: Whether to cache the dataset in memory
            mixup_alpha: Alpha parameter for mixup augmentation (0 to disable)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.cache_data = cache_data
        self.mixup_alpha = mixup_alpha
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.mean = None  # Will be computed from training data
        self.std = None   # Will be computed from training data
        
        # Set up logging
        self._setup_logging()
        
        # Create more sophisticated augmentation pipeline
        self.train_augmentor = A.Compose([
            # Spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RandomGamma(),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8))
            ], p=0.5),
            
            # Noise and blur
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.GaussNoise(var_limit=(10, 50)),
                A.MotionBlur(blur_limit=3)
            ], p=0.2),
            
            # Weather simulations for outdoor scenes
            A.OneOf([
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255)),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None),
            ], p=0.1),
            
            # Compression and quality
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),
            
            # Normalization - will be handled separately
        ])
        
        # Validation augmentation (only essential preprocessing)
        self.val_augmentor = A.Compose([
            # No augmentation for validation, only resizing if needed
        ])

    def _setup_logging(self):
        """Set up logging for the preprocessor."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Get logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Only add handlers if they don't exist already
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Create file handler
            file_handler = logging.FileHandler(log_dir / 'preprocessing.log', mode='a')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def compute_statistics(self, dataset=None):
        """Compute mean and standard deviation from the training dataset.
        
        Args:
            dataset: Optional dataset to use for computing statistics
                    If None, will create a new dataset from the training split
        """
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
            
        self.logger.info("Computing dataset statistics...")
        
        # Create dataset if not provided
        if dataset is None:
            dataset = tf.keras.utils.image_dataset_from_directory(
                self.data_dir / 'train',
                image_size=self.img_size,
                batch_size=self.batch_size,
                shuffle=False,
                label_mode=None  # We only need images for statistics
            )
            
        # Compute mean and std across all images
        mean_sum = np.zeros(3)  # RGB channels
        std_sum = np.zeros(3)
        n_images = 0
        
        # First pass for mean
        for batch in tqdm(dataset, desc="Computing mean"):
            batch = batch.numpy() / 255.0  # Normalize to [0,1]
            mean_sum += np.sum(batch, axis=(0, 1, 2))
            n_images += batch.shape[0] * batch.shape[1] * batch.shape[2]
            
        # Calculate mean
        self.mean = mean_sum / n_images
        
        # Second pass for std
        for batch in tqdm(dataset, desc="Computing std"):
            batch = batch.numpy() / 255.0  # Normalize to [0,1]
            std_sum += np.sum(np.square(batch - self.mean), axis=(0, 1, 2))
            
        # Calculate std
        self.std = np.sqrt(std_sum / n_images)
        
        self.logger.info(f"Dataset statistics - Mean: {self.mean}, Std: {self.std}")
        return self.mean, self.std
    
    def compute_class_weights(self):
        """Compute class weights to handle class imbalance.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        self.logger.info("Computing class weights...")
        
        # Get all class directories
        train_dir = self.data_dir / 'train'
        class_counts = {}
        
        # Count images in each class
        for i, cls in enumerate(self.class_names):
            cls_dir = train_dir / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob('*.jpg')))
                class_counts[i] = count
            else:
                class_counts[i] = 0
                
        # Calculate class weights using sklearn's balanced method
        total_samples = sum(class_counts.values())
        n_classes = len(self.class_names)
        class_weights = {}
        
        for cls_idx, count in class_counts.items():
            if count > 0:
                class_weights[cls_idx] = total_samples / (n_classes * count)
            else:
                class_weights[cls_idx] = 1.0
                
        self.logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    def tf_augment(self, image, label, training=True):
        """Apply augmentation using TensorFlow py_function.
        
        Args:
            image: Input image tensor
            label: Input label tensor
            training: Whether this is for training (True) or validation (False)
            
        Returns:
            Tuple of (augmented image, label)
        """
        def aug_fn(img, is_training):
            # Convert to numpy
            img_np = img.numpy()
            
            # Apply augmentation based on mode
            if is_training:
                result = self.train_augmentor(image=img_np)['image']
            else:
                result = self.val_augmentor(image=img_np)['image']
                
            # Apply normalization if statistics are available
            if self.mean is not None and self.std is not None:
                result = (result / 255.0 - self.mean) / self.std
            else:
                result = result / 255.0  # Simple normalization
                
            return result.astype(np.float32)

        # Apply augmentation
        aug_img = tf.py_function(
            lambda x: aug_fn(x, training), 
            [image], 
            Tout=tf.float32
        )
        
        # Ensure shape is preserved
        aug_img.set_shape([None, None, 3])
        
        return aug_img, label
        
    def apply_mixup(self, images, labels, alpha=0.2):
        """Apply mixup augmentation to a batch of images.
        
        Args:
            images: Batch of images
            labels: Batch of one-hot encoded labels
            alpha: Mixup interpolation coefficient
            
        Returns:
            Tuple of mixed images and labels
        """
        if alpha <= 0:
            return images, labels
            
        # Generate mixup coefficient
        batch_size = tf.shape(images)[0]
        lam = tf.random.beta(alpha, alpha, shape=[batch_size, 1, 1, 1])
        
        # Create shuffled indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Shuffle images and labels
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        # Reshape lambda for labels
        lam_labels = tf.reshape(lam, [batch_size, 1])
        
        # Mix labels
        mixed_labels = lam_labels * labels + (1 - lam_labels) * shuffled_labels
        
        return mixed_images, mixed_labels

    def create_tf_dataset(self, split='train', shuffle=True, augment=True):
        """Create a TensorFlow dataset for the specified split.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow dataset
        """
        self.logger.info(f"Creating {split} dataset...")
        
        # Check if directory exists
        path = self.data_dir / split
        if not path.exists():
            self.logger.error(f"Directory not found: {path}")
            raise ValueError(f"Directory not found: {path}")
            
        # Compute dataset statistics if not already done
        if split == 'train' and (self.mean is None or self.std is None):
            self.compute_statistics()
            
        # Create base dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=shuffle,
            seed=42  # For reproducibility
        )
        
        # Get dataset size for logging
        dataset_size = sum(1 for _ in path.glob('*/*'))
        self.logger.info(f"Created {split} dataset with {dataset_size} images")
        
        # Apply augmentation if requested
        is_training = split == 'train'
        if augment:
            dataset = dataset.map(
                lambda x, y: self.tf_augment(x, y, training=is_training),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # Just normalize without augmentation
            dataset = dataset.map(
                lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        # Apply mixup for training if enabled
        if is_training and self.mixup_alpha > 0 and augment:
            dataset = dataset.map(
                lambda x, y: self.apply_mixup(x, y, self.mixup_alpha),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
        # Cache if requested
        if self.cache_data:
            dataset = dataset.cache()
            
        # Shuffle, prefetch, etc.
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(dataset_size, 10000))
            
        # Always prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def create_pipeline(self, include_test=True):
        """Create a complete data pipeline for training, validation, and optionally test.
        
        Args:
            include_test: Whether to include the test dataset
            
        Returns:
            Dictionary containing datasets and class weights
        """
        self.logger.info("Creating complete data pipeline...")
        
        # Create datasets
        train_ds = self.create_tf_dataset('train', shuffle=True, augment=True)
        val_ds = self.create_tf_dataset('val', shuffle=False, augment=False)
        
        # Create test dataset if requested
        test_ds = None
        if include_test:
            test_ds = self.create_tf_dataset('test', shuffle=False, augment=False)
            
        # Compute class weights
        class_weights = self.compute_class_weights()
        
        # Return everything in a dictionary
        return {
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds,
            'class_weights': class_weights
        }
        
    def visualize_augmentations(self, num_samples=5, save_path=None):
        """Visualize augmentations applied to sample images.
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization, or None to display
        """
        # Create a dataset with a few samples
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir / 'train',
            image_size=self.img_size,
            batch_size=num_samples,
            shuffle=True,
            label_mode='categorical',
            seed=42
        )
        
        # Get a batch of images
        for images, labels in dataset.take(1):
            # Convert to numpy
            images = images.numpy()
            labels = labels.numpy()
            
            # Create figure
            fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
            
            # For each sample
            for i in range(num_samples):
                # Original image
                axes[i, 0].imshow(images[i].astype('uint8'))
                axes[i, 0].set_title(f"Original: {self.class_names[np.argmax(labels[i])]}")
                
                # Generate 4 different augmentations
                for j in range(1, 5):
                    # Apply augmentation
                    aug_img = self.train_augmentor(image=images[i])['image']
                    
                    # Display
                    axes[i, j].imshow(aug_img.astype('uint8'))
                    axes[i, j].set_title(f"Augmentation {j}")
                    
                # Remove axes
                for j in range(5):
                    axes[i, j].axis('off')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()
                self.logger.info(f"Saved augmentation visualization to {save_path}")
            else:
                plt.show()
                
    def analyze_dataset(self, save_dir=None):
        """Analyze the dataset and generate statistics.
        
        Args:
            save_dir: Directory to save analysis results, or None to display
        """
        self.logger.info("Analyzing dataset...")
        
        # Create save directory if needed
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        # Analyze class distribution
        class_distribution = {}
        
        # For each split
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
                
            class_distribution[split] = {}
            
            # Count images in each class
            for cls in self.class_names:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    count = len(list(cls_dir.glob('*.jpg')))
                    class_distribution[split][cls] = count
                else:
                    class_distribution[split][cls] = 0
        
        # Plot class distribution
        plt.figure(figsize=(12, 6))
        
        # Get all splits
        splits = list(class_distribution.keys())
        
        # Set width of bars
        bar_width = 0.8 / len(splits)
        
        # Set positions of bars on x-axis
        r = np.arange(len(self.class_names))
        
        # Plot bars
        for i, split in enumerate(splits):
            counts = [class_distribution[split][cls] for cls in self.class_names]
            plt.bar(r + i * bar_width, counts, width=bar_width, label=split.capitalize())
            
        # Add labels and legend
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution Across Splits')
        plt.xticks(r + bar_width * (len(splits) - 1) / 2, self.class_names)
        plt.legend()
        
        # Save or show
        if save_dir:
            plt.savefig(save_dir / 'class_distribution.png')
            plt.close()
            self.logger.info(f"Saved class distribution plot to {save_dir / 'class_distribution.png'}")
        else:
            plt.show()
            
        # Return analysis results
        return {
            'class_distribution': class_distribution
        }
