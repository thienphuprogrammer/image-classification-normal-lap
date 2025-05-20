import kaggle
from pathlib import Path
import shutil
import logging
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm
import time

class DataLoader:
    """Handles downloading, verifying, and preparing image classification datasets from Kaggle."""
    
    def __init__(self, dataset_path, kaggle_dataset="puneet6060/intel-image-classification", max_workers=4):
        """Initialize the DataLoader with dataset path and source.
        
        Args:
            dataset_path: Path to store the dataset
            kaggle_dataset: Kaggle dataset identifier
            max_workers: Maximum number of worker threads for parallel operations
        """
        self.logger = None
        self.dataset_path = Path(dataset_path)
        self.kaggle_dataset = kaggle_dataset
        self.required_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.max_workers = max_workers
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging with both console and file output."""
        log_file = Path('logs')
        log_file.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file / 'dataset.log', mode='a')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def download_dataset(self):
        """Download dataset from Kaggle if it doesn't already exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if dataset already exists to avoid redundant downloads
            if (self.dataset_path / 'seg_train').exists():
                self.logger.info("Dataset already exists, skipping download.")
                return True

            # Create the dataset directory if it doesn't exist
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Verify Kaggle API credentials
            self._verify_kaggle_credentials()
            
            start_time = time.time()
            self.logger.info(f"Downloading dataset '{self.kaggle_dataset}' from Kaggle...")
            
            # Download the dataset
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=self.dataset_path,
                unzip=True,
                quiet=False
            )
            
            download_time = time.time() - start_time
            self.logger.info(f"Download completed in {download_time:.2f} seconds")
            return True
            
        except kaggle.rest.ApiException as api_e:
            self.logger.error(f"Kaggle API error: {api_e}")
            self.logger.info("Please check your Kaggle API credentials in ~/.kaggle/kaggle.json")
            return False
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            return False
            
    def _verify_kaggle_credentials(self):
        """Verify that Kaggle API credentials are properly set up."""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            self.logger.warning("Kaggle credentials not found at ~/.kaggle/kaggle.json")
            self.logger.info("Please set up your Kaggle API credentials")
            
            # Check if credentials exist in the project directory
            project_kaggle_json = Path('kaggle.json')
            if project_kaggle_json.exists():
                # Create the .kaggle directory if it doesn't exist
                kaggle_dir.mkdir(exist_ok=True)
                
                # Copy the credentials file
                shutil.copy(project_kaggle_json, kaggle_json)
                os.chmod(kaggle_json, 0o600)  # Set proper permissions
                self.logger.info("Copied Kaggle credentials from project directory")

    @lru_cache(maxsize=1)
    def verify_dataset_structure(self):
        """Verify that the dataset has the expected directory structure and classes.
        
        Returns:
            bool: True if the structure is valid, False otherwise
        """
        try:
            # Check for required top-level directories
            required_dirs = {'seg_train', 'seg_test', 'seg_pred'}
            existing_dirs = {d.name for d in self.dataset_path.iterdir() if d.is_dir()}

            if not required_dirs.issubset(existing_dirs):
                missing = required_dirs - existing_dirs
                self.logger.error(f"Missing directories: {missing}")
                return False

            # Check for required class directories in each split
            valid = True
            for split in ['train', 'test']:
                split_path = self.dataset_path / f'seg_{split}' / f'seg_{split}'
                if not split_path.exists():
                    self.logger.error(f"Directory not found: {split_path}")
                    valid = False
                    continue
                    
                present_classes = {d.name for d in split_path.iterdir() if d.is_dir()}
                if not set(self.required_classes).issubset(present_classes):
                    missing = set(self.required_classes) - present_classes
                    self.logger.error(f"Missing classes in {split}: {missing}")
                    valid = False
                    
                # Check that each class directory contains images
                for cls in present_classes:
                    class_dir = split_path / cls
                    image_count = len(list(class_dir.glob('*.jpg')))
                    if image_count == 0:
                        self.logger.warning(f"No images found in {split}/{cls}")

            return valid
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False

    def create_validation_split(self, validation_ratio=0.2):
        """Create a validation split from the training data using parallel processing.
        
        Args:
            validation_ratio: Fraction of training data to use for validation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if validation split already exists
            val_dir = self.dataset_path / 'seg_val'
            if val_dir.exists() and any(val_dir.iterdir()):
                self.logger.info("Validation split already exists, skipping creation.")
                return True
                
            val_dir.mkdir(exist_ok=True)
            train_dir = self.dataset_path / 'seg_train' / 'seg_train'
            
            if not train_dir.exists():
                self.logger.error(f"Training directory not found: {train_dir}")
                return False
            
            # Process each class in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for cls in self.required_classes:
                    futures.append(executor.submit(
                        self._split_class, cls, train_dir, val_dir, validation_ratio
                    ))
                
                # Wait for all tasks to complete and check results
                for future in futures:
                    if not future.result():
                        return False
            
            # Verify the validation split was created correctly
            self._verify_validation_split(val_dir)
            return True
            
        except Exception as e:
            self.logger.error(f"Validation split failed: {str(e)}")
            return False
            
    def _split_class(self, cls, train_dir, val_dir, validation_ratio):
        """Split a single class into training and validation sets.
        
        Args:
            cls: Class name
            train_dir: Source directory for training data
            val_dir: Target directory for validation data
            validation_ratio: Fraction of data to use for validation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            src = train_dir / cls
            if not src.exists():
                self.logger.warning(f"Class directory not found: {src}")
                return True  # Not a critical error, continue with other classes
                
            # Create the validation class directory
            val_cls = val_dir / cls
            val_cls.mkdir(exist_ok=True)
            
            # Get all image files and shuffle them
            files = list(src.glob('*.jpg'))
            if not files:
                self.logger.warning(f"No images found in {src}")
                return True
                
            # Set a fixed random seed for reproducibility
            np.random.seed(42)
            np.random.shuffle(files)
            
            # Calculate split index
            split_idx = int(len(files) * validation_ratio)
            if split_idx == 0:
                self.logger.warning(f"Too few images in {cls} to create validation split")
                return True
            
            # Move files to validation directory with progress bar
            for f in tqdm(files[:split_idx], desc=f"Creating validation split for {cls}"):
                shutil.copy2(f, val_cls / f.name)  # Use copy instead of move to preserve original data
            
            self.logger.info(f"Created validation split for {cls}: {split_idx} images")
            return True
            
        except Exception as e:
            self.logger.error(f"Error splitting class {cls}: {str(e)}")
            return False
            
    def _verify_validation_split(self, val_dir):
        """Verify that the validation split was created correctly."""
        class_counts = {}
        for cls in self.required_classes:
            cls_dir = val_dir / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob('*.jpg')))
                class_counts[cls] = count
                
        total = sum(class_counts.values())
        self.logger.info(f"Validation split created with {total} images:")
        for cls, count in class_counts.items():
            self.logger.info(f"  - {cls}: {count} images")
    
    def cleanup_dataset(self):
        """Clean up temporary files and organize dataset.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Track what was cleaned up
            cleanup_stats = {
                'zip_files': 0,
                'temp_dirs': 0,
                'other_files': 0
            }
            
            # Remove any zip files
            for zip_file in self.dataset_path.glob('*.zip'):
                zip_file.unlink()
                cleanup_stats['zip_files'] += 1
            
            # Remove any temporary directories
            temp_dirs = [d for d in self.dataset_path.iterdir() 
                        if d.is_dir() and (d.name.startswith('temp_') or d.name == '__MACOSX')]
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)
                cleanup_stats['temp_dirs'] += 1
            
            # Remove any other temporary files
            for temp_file in self.dataset_path.glob('._*'):
                temp_file.unlink()
                cleanup_stats['other_files'] += 1
            
            self.logger.info(f"Dataset cleanup completed: removed {cleanup_stats['zip_files']} zip files, "
                           f"{cleanup_stats['temp_dirs']} temporary directories, and "
                           f"{cleanup_stats['other_files']} other temporary files")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up dataset: {str(e)}")
            return False
    
    def prepare_dataset(self):
        """Prepare the complete dataset by downloading, verifying, and creating splits.
        
        Returns:
            bool: True if all preparation steps completed successfully, False otherwise
        """
        start_time = time.time()
        self.logger.info("Starting dataset preparation...")
        
        # Ensure the dataset directory exists
        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True)
        
        # Define preparation steps
        steps = [
            ("Downloading dataset", self.download_dataset),
            ("Verifying dataset structure", self.verify_dataset_structure),
            ("Creating validation split", self.create_validation_split),
            ("Cleaning up temporary files", self.cleanup_dataset)
        ]
        
        # Execute each step
        for step_name, step_func in steps:
            self.logger.info(f"Step: {step_name}")
            step_start = time.time()
            
            if not step_func():
                self.logger.error(f"Dataset preparation failed at step: {step_name}")
                return False
                
            step_time = time.time() - step_start
            self.logger.info(f"Completed {step_name} in {step_time:.2f} seconds")
        
        # Calculate dataset statistics
        self._log_dataset_statistics()
        
        total_time = time.time() - start_time
        self.logger.info(f"Dataset preparation completed successfully in {total_time:.2f} seconds!")
        return True
        
    def _log_dataset_statistics(self):
        """Log statistics about the prepared dataset."""
        try:
            stats = {}
            
            # Count images in each split and class
            for split in ['train', 'test', 'val']:
                split_dir = self.dataset_path / f'seg_{split}'
                if not split_dir.exists():
                    continue
                    
                # Handle the nested directory structure in train and test
                if split in ['train', 'test']:
                    split_dir = split_dir / f'seg_{split}'
                    if not split_dir.exists():
                        continue
                
                total_images = 0
                class_stats = {}
                
                for cls in self.required_classes:
                    cls_dir = split_dir / cls
                    if cls_dir.exists():
                        image_count = len(list(cls_dir.glob('*.jpg')))
                        class_stats[cls] = image_count
                        total_images += image_count
                
                stats[split] = {
                    'total': total_images,
                    'classes': class_stats
                }
            
            # Log the statistics
            self.logger.info("Dataset statistics:")
            for split, split_stats in stats.items():
                self.logger.info(f"  {split.capitalize()} set: {split_stats['total']} images")
                for cls, count in split_stats['classes'].items():
                    self.logger.info(f"    - {cls}: {count} images ({count/split_stats['total']*100:.1f}%)")
                    
        except Exception as e:
            self.logger.error(f"Error calculating dataset statistics: {str(e)}") 