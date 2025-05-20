import shutil
from pathlib import Path
import os
import numpy as np
from albumentations import Compose, HorizontalFlip, RandomRotate90, RandomBrightnessContrast
import albumentations as A
import logging
import cv2
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

class DataBalancer:
    """Class for balancing image datasets by augmenting underrepresented classes."""
    
    def __init__(self, data_dir, target_dir, img_size=(150, 150), max_workers=4):
        """Initialize the DataBalancer with source and target directories.
        
        Args:
            data_dir: Source directory containing the original dataset
            target_dir: Target directory for the balanced dataset
            img_size: Tuple of (width, height) for resizing images
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.img_size = img_size
        self.max_workers = max_workers
        
        # Create a more diverse set of augmentations for better generalization
        self.augmentor = Compose([
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.OneOf([
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1)
            ], p=0.3)
        ])
        
        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the DataBalancer."""
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
            file_handler = logging.FileHandler(log_dir / 'data_balancing.log', mode='a')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def augment_image(self, image):
        """Apply augmentation to a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image as numpy array
        """
        # Check if image is valid
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided for augmentation")
            
        # Apply augmentation
        return self.augmentor(image=image)['image']

    def balance_class(self, source_dir, target_dir, cls, target_count):
        """Balance a specific class in a dataset split.
        
        Args:
            source_dir: Source directory containing the original class images
            target_dir: Target directory for the balanced class
            cls: Class name
            target_count: Target number of images for the class
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info(f"Balancing class '{cls}' to {target_count} images...")
            
            src_path = source_dir / cls
            tgt_path = target_dir / cls
            
            # Check if source directory exists
            if not src_path.exists():
                self.logger.error(f"Source directory not found: {src_path}")
                return False
                
            # Create target directory
            tgt_path.mkdir(parents=True, exist_ok=True)
            
            # Get existing images
            existing_images = list(src_path.glob('*.jpg'))
            existing_count = len(existing_images)
            
            if existing_count == 0:
                self.logger.warning(f"No images found in {src_path}, skipping class")
                return False
                
            self.logger.info(f"Found {existing_count} existing images for class '{cls}'.")
            
            # Copy existing images (up to target_count)
            copy_count = min(existing_count, target_count)
            for img_path in tqdm(existing_images[:copy_count], desc=f"Copying {cls} images"):
                shutil.copy2(img_path, tgt_path / img_path.name)
            
            # Augment if needed
            if existing_count < target_count:
                augment_count = target_count - existing_count
                self.logger.info(f"Generating {augment_count} augmented images for class '{cls}'...")
                
                success = self._augment_class(
                    src_path=src_path,
                    tgt_path=tgt_path,
                    target_count=augment_count
                )
                
                if not success:
                    self.logger.warning(f"Failed to generate all required augmented images for '{cls}'.")
            
            # Log completion
            elapsed_time = time.time() - start_time
            final_count = len(list(tgt_path.glob('*.jpg')))
            self.logger.info(f"Balanced class '{cls}' to {final_count} images in {elapsed_time:.2f} seconds.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error balancing class '{cls}': {str(e)}")
            return False

    def _augment_class(self, src_path, tgt_path, target_count):
        """Generate augmented images using parallel processing.
        
        Args:
            src_path: Source directory containing original images
            tgt_path: Target directory for augmented images
            target_count: Number of augmented images to generate
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get list of existing images
            existing_images = list(src_path.glob('*.jpg'))
            
            # Check if there are any images to augment
            if not existing_images:
                self.logger.error(f"No images found in {src_path} for augmentation")
                return False
                
            # Create batches for parallel processing
            batch_size = min(100, target_count)  # Process in batches to avoid memory issues
            num_batches = (target_count + batch_size - 1) // batch_size  # Ceiling division
            
            # Process each batch
            total_generated = 0
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, target_count)
                batch_count = end_idx - start_idx
                
                # Generate augmented images in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for i in range(batch_count):
                        futures.append(executor.submit(
                            self._generate_augmented_image,
                            i + start_idx,
                            existing_images,
                            tgt_path
                        ))
                    
                    # Wait for all tasks to complete with progress bar
                    for future in tqdm(futures, desc=f"Generating augmented images (batch {batch+1}/{num_batches})"):
                        if future.result():
                            total_generated += 1
            
            self.logger.info(f"Generated {total_generated} augmented images")
            return total_generated == target_count
            
        except Exception as e:
            self.logger.error(f"Error generating augmented images: {str(e)}")
            return False
            
    def _generate_augmented_image(self, index, source_images, target_path):
        """Generate a single augmented image.
        
        Args:
            index: Index for the filename
            source_images: List of source image paths
            target_path: Directory to save the augmented image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Select a random source image
            source_img = random.choice(source_images)
            
            # Read the image
            img = cv2.imread(str(source_img))
            if img is None:
                self.logger.warning(f"Failed to read image: {source_img}")
                return False
                
            # Resize image if needed
            if self.img_size is not None:
                img = cv2.resize(img, self.img_size)
                
            # Apply multiple augmentations for more diversity
            # We'll create 2-3 different augmentations and randomly select one
            augmented_versions = []
            for _ in range(random.randint(2, 3)):
                augmented = self.augmentor(image=img)['image']
                augmented_versions.append(augmented)
                
            # Select one augmented version randomly
            final_augmented = random.choice(augmented_versions)
            
            # Generate a unique filename
            timestamp = int(time.time() * 1000)  # Use timestamp for uniqueness
            filename = f"aug_{index}_{timestamp}_{source_img.name}"
            
            # Save the augmented image
            output_path = target_path / filename
            cv2.imwrite(str(output_path), final_augmented)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating augmented image: {str(e)}")
            return False
            
    def balance_dataset(self, splits=['train', 'val', 'test'], target_counts=None):
        """Balance all classes across all splits to the same count.
        
        Args:
            splits: List of dataset splits to balance
            target_counts: Dictionary mapping splits to target counts, or None to use max class count
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info("Starting dataset balancing...")
            
            # Process each split
            for split in splits:
                split_dir = self.data_dir / f"seg_{split}"
                
                # Handle nested directory structure for train and test
                if split in ['train', 'test']:
                    nested_dir = split_dir / f"seg_{split}"
                    if nested_dir.exists():
                        split_dir = nested_dir
                
                # Check if split directory exists
                if not split_dir.exists():
                    self.logger.warning(f"Split directory not found: {split_dir}, skipping")
                    continue
                    
                # Get class directories
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                if not class_dirs:
                    self.logger.warning(f"No class directories found in {split_dir}, skipping")
                    continue
                    
                # Count images in each class
                class_counts = {}
                for class_dir in class_dirs:
                    cls = class_dir.name
                    count = len(list(class_dir.glob('*.jpg')))
                    class_counts[cls] = count
                
                # Determine target count for this split
                if target_counts and split in target_counts:
                    target_count = target_counts[split]
                else:
                    # Use the maximum class count as the target
                    target_count = max(class_counts.values()) if class_counts else 0
                
                if target_count == 0:
                    self.logger.warning(f"No images found in {split} split, skipping")
                    continue
                    
                # Log class distribution before balancing
                self.logger.info(f"Class distribution in {split} split before balancing:")
                for cls, count in class_counts.items():
                    self.logger.info(f"  - {cls}: {count} images")
                self.logger.info(f"Target count for {split} split: {target_count} images per class")
                
                # Balance each class
                target_split_dir = self.target_dir / f"balanced_{split}"
                for cls in class_counts.keys():
                    self.balance_class(
                        source_dir=split_dir,
                        target_dir=target_split_dir,
                        cls=cls,
                        target_count=target_count
                    )
                
                # Log completion for this split
                self.logger.info(f"Completed balancing {split} split")
            
            # Log overall completion
            elapsed_time = time.time() - start_time
            self.logger.info(f"Dataset balancing completed in {elapsed_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Error balancing dataset: {str(e)}")
            return False