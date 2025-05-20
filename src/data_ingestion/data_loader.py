import os
import kaggle
import zipfile
from pathlib import Path
import shutil
import logging
from tqdm import tqdm

class DataLoader:
    def __init__(self, dataset_path, kaggle_dataset="puneet6060/intel-image-classification"):
        self.dataset_path = Path(dataset_path)
        self.kaggle_dataset = kaggle_dataset
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_dataset(self):
        """Download dataset from Kaggle"""
        try:
            self.logger.info("Downloading dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=self.dataset_path,
                unzip=True
            )
            self.logger.info("Dataset downloaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Error downloading dataset: {str(e)}")
            return False
    
    def verify_dataset_structure(self):
        """Verify the dataset structure and integrity"""
        required_dirs = ['train', 'test']
        required_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        
        try:
            # Check if main directories exist
            for dir_name in required_dirs:
                dir_path = self.dataset_path / dir_name
                if not dir_path.exists():
                    self.logger.error(f"Missing directory: {dir_name}")
                    return False
                
                # Check if all class directories exist
                for class_name in required_classes:
                    class_path = dir_path / class_name
                    if not class_path.exists():
                        self.logger.error(f"Missing class directory: {class_name} in {dir_name}")
                        return False
                    
                    # Check if directory contains images
                    images = list(class_path.glob('*.jpg'))
                    if not images:
                        self.logger.error(f"No images found in {class_path}")
                        return False
                    
                    self.logger.info(f"Found {len(images)} images in {class_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error verifying dataset structure: {str(e)}")
            return False
    
    def create_validation_split(self, validation_ratio=0.2):
        """Create validation split from training data"""
        try:
            train_dir = self.dataset_path / 'train'
            val_dir = self.dataset_path / 'validation'
            
            # Create validation directory if it doesn't exist
            val_dir.mkdir(exist_ok=True)
            
            for class_name in ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']:
                # Create class directory in validation
                (val_dir / class_name).mkdir(exist_ok=True)
                
                # Get all images for this class
                images = list((train_dir / class_name).glob('*.jpg'))
                num_val = int(len(images) * validation_ratio)
                
                # Move validation images
                for img in tqdm(images[:num_val], desc=f"Moving validation images for {class_name}"):
                    shutil.move(str(img), str(val_dir / class_name / img.name))
                
                self.logger.info(f"Created validation split for {class_name}: {num_val} images")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating validation split: {str(e)}")
            return False
    
    def cleanup_dataset(self):
        """Clean up temporary files and organize dataset"""
        try:
            # Remove any zip files
            for zip_file in self.dataset_path.glob('*.zip'):
                zip_file.unlink()
            
            # Remove any temporary directories
            temp_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir() and d.name.startswith('temp_')]
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)
            
            self.logger.info("Dataset cleanup completed successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up dataset: {str(e)}")
            return False
    
    def prepare_dataset(self):
        """Prepare the complete dataset"""
        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True)
        
        # Download dataset
        if not self.download_dataset():
            return False
        
        # Verify dataset structure
        if not self.verify_dataset_structure():
            return False
        
        # Create validation split
        if not self.create_validation_split():
            return False
        
        # Cleanup
        if not self.cleanup_dataset():
            return False
        
        self.logger.info("Dataset preparation completed successfully!")
        return True 