import os
import shutil
import random
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.model_selection import train_test_split

class DataDistributor:
    """Professional data redistribution for optimizing training datasets.
    
    This class handles intelligent redistribution of data across train/val/test splits:
    - Stratified sampling to maintain class distributions
    - Cross-validation fold creation
    - Handling of class imbalance through intelligent redistribution
    - Data analysis and visualization
    """
    
    def __init__(self, source_dir, target_dir, class_names=None, max_workers=4):
        """Initialize the DataDistributor.
        
        Args:
            source_dir: Source directory containing the dataset
            target_dir: Target directory for the redistributed dataset
            class_names: List of class names (if None, will be inferred)
            max_workers: Maximum number of worker threads for parallel operations
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.max_workers = max_workers
        
        # Set up logging
        self._setup_logging()
        
        # Infer class names if not provided
        if class_names is None:
            self._infer_class_names()
        else:
            self.class_names = class_names
            
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Set up logging for the distributor."""
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
            file_handler = logging.FileHandler(log_dir / 'data_distribution.log', mode='a')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _infer_class_names(self):
        """Infer class names from the directory structure."""
        self.logger.info("Inferring class names from directory structure...")
        
        # Look for class directories in any of the standard splits
        for split in ['train', 'val', 'test', 'seg_train/seg_train', 'seg_test/seg_test']:
            split_dir = self.source_dir / split
            if split_dir.exists() and split_dir.is_dir():
                # Get all subdirectories (classes)
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                if class_dirs:
                    self.class_names = [d.name for d in class_dirs]
                    self.logger.info(f"Inferred {len(self.class_names)} classes: {self.class_names}")
                    return
        
        # If no classes found, use default
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.logger.warning(f"Could not infer class names, using default: {self.class_names}")
    
    def analyze_source_data(self):
        """Analyze the source data distribution.
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing source data distribution...")
        
        # Initialize results
        results = {
            'class_counts': {},
            'split_counts': {},
            'total_images': 0
        }
        
        # Find all image files in the source directory
        image_paths = []
        class_counts = {cls: 0 for cls in self.class_names}
        split_counts = {}
        
        # Check different directory structures
        for split in ['train', 'val', 'test']:
            # Check for flat structure
            split_dir = self.source_dir / split
            if split_dir.exists() and split_dir.is_dir():
                split_counts[split] = 0
                for cls in self.class_names:
                    cls_dir = split_dir / cls
                    if cls_dir.exists() and cls_dir.is_dir():
                        images = list(cls_dir.glob('*.jpg'))
                        class_counts[cls] += len(images)
                        split_counts[split] += len(images)
                        image_paths.extend(images)
            
            # Check for nested structure (seg_train/seg_train)
            nested_dir = self.source_dir / f'seg_{split}' / f'seg_{split}'
            if nested_dir.exists() and nested_dir.is_dir():
                split_counts[f'seg_{split}'] = 0
                for cls in self.class_names:
                    cls_dir = nested_dir / cls
                    if cls_dir.exists() and cls_dir.is_dir():
                        images = list(cls_dir.glob('*.jpg'))
                        class_counts[cls] += len(images)
                        split_counts[f'seg_{split}'] += len(images)
                        image_paths.extend(images)
        
        # Update results
        results['class_counts'] = class_counts
        results['split_counts'] = split_counts
        results['total_images'] = len(image_paths)
        
        # Log results
        self.logger.info(f"Found {results['total_images']} images in total")
        self.logger.info(f"Class distribution: {class_counts}")
        self.logger.info(f"Split distribution: {split_counts}")
        
        return results
    
    def redistribute_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify=True, seed=42):
        """Redistribute data into train/val/test splits with stratified sampling.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            stratify: Whether to use stratified sampling
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with redistribution statistics
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
        
        self.logger.info(f"Redistributing data with ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Collect all images by class
        class_images = {cls: [] for cls in self.class_names}
        
        # Find all image files in the source directory
        for split in ['train', 'val', 'test']:
            # Check for flat structure
            split_dir = self.source_dir / split
            if split_dir.exists() and split_dir.is_dir():
                for cls in self.class_names:
                    cls_dir = split_dir / cls
                    if cls_dir.exists() and cls_dir.is_dir():
                        images = list(cls_dir.glob('*.jpg'))
                        class_images[cls].extend(images)
            
            # Check for nested structure (seg_train/seg_train)
            nested_dir = self.source_dir / f'seg_{split}' / f'seg_{split}'
            if nested_dir.exists() and nested_dir.is_dir():
                for cls in self.class_names:
                    cls_dir = nested_dir / cls
                    if cls_dir.exists() and cls_dir.is_dir():
                        images = list(cls_dir.glob('*.jpg'))
                        class_images[cls].extend(images)
        
        # Create target directories
        for split in ['train', 'val', 'test']:
            split_dir = self.target_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for cls in self.class_names:
                cls_dir = split_dir / cls
                cls_dir.mkdir(exist_ok=True)
        
        # Redistribute data for each class
        stats = {'train': {}, 'val': {}, 'test': {}}
        
        for cls in self.class_names:
            images = class_images[cls]
            if not images:
                self.logger.warning(f"No images found for class {cls}")
                continue
                
            # Shuffle images
            random.shuffle(images)
            
            # Split into train/val/test
            if stratify:
                # First split into train and temp (val+test)
                train_images, temp_images = train_test_split(
                    images, 
                    train_size=train_ratio,
                    random_state=seed
                )
                
                # Then split temp into val and test
                val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
                val_images, test_images = train_test_split(
                    temp_images,
                    train_size=val_ratio_adjusted,
                    random_state=seed
                )
            else:
                # Simple proportional split
                n_train = int(len(images) * train_ratio)
                n_val = int(len(images) * val_ratio)
                
                train_images = images[:n_train]
                val_images = images[n_train:n_train+n_val]
                test_images = images[n_train+n_val:]
            
            # Copy images to target directories
            self._copy_images(train_images, self.target_dir / 'train' / cls)
            self._copy_images(val_images, self.target_dir / 'val' / cls)
            self._copy_images(test_images, self.target_dir / 'test' / cls)
            
            # Update statistics
            stats['train'][cls] = len(train_images)
            stats['val'][cls] = len(val_images)
            stats['test'][cls] = len(test_images)
        
        # Log statistics
        for split in ['train', 'val', 'test']:
            total = sum(stats[split].values())
            self.logger.info(f"{split.capitalize()} split: {total} images")
            for cls, count in stats[split].items():
                self.logger.info(f"  - {cls}: {count} images ({count/total*100:.1f}%)")
        
        return stats
    
    def _copy_images(self, images, target_dir):
        """Copy images to target directory with progress bar.
        
        Args:
            images: List of image paths
            target_dir: Target directory
        """
        for img_path in tqdm(images, desc=f"Copying to {target_dir.name}", leave=False):
            target_path = target_dir / img_path.name
            shutil.copy2(img_path, target_path)
    
    def create_cross_validation_folds(self, n_folds=5, seed=42):
        """Create cross-validation folds from the training data.
        
        Args:
            n_folds: Number of folds to create
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with fold statistics
        """
        self.logger.info(f"Creating {n_folds} cross-validation folds...")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Create folds directory
        folds_dir = self.target_dir / 'cv_folds'
        folds_dir.mkdir(exist_ok=True)
        
        # Collect all training images by class
        train_dir = self.target_dir / 'train'
        class_images = {cls: [] for cls in self.class_names}
        
        for cls in self.class_names:
            cls_dir = train_dir / cls
            if cls_dir.exists() and cls_dir.is_dir():
                images = list(cls_dir.glob('*.jpg'))
                class_images[cls].extend(images)
        
        # Create fold directories
        for fold in range(n_folds):
            fold_dir = folds_dir / f'fold_{fold}'
            fold_dir.mkdir(exist_ok=True)
            
            for split in ['train', 'val']:
                split_dir = fold_dir / split
                split_dir.mkdir(exist_ok=True)
                
                for cls in self.class_names:
                    cls_dir = split_dir / cls
                    cls_dir.mkdir(exist_ok=True)
        
        # Create folds for each class
        stats = {f'fold_{fold}': {'train': {}, 'val': {}} for fold in range(n_folds)}
        
        for cls in self.class_names:
            images = class_images[cls]
            if not images:
                self.logger.warning(f"No images found for class {cls}")
                continue
                
            # Shuffle images
            random.shuffle(images)
            
            # Split into n_folds
            fold_size = len(images) // n_folds
            folds = []
            
            for fold in range(n_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else len(images)
                folds.append(images[start_idx:end_idx])
            
            # For each fold, use it as validation and the rest as training
            for fold in range(n_folds):
                val_images = folds[fold]
                train_images = [img for i, fold_imgs in enumerate(folds) if i != fold for img in fold_imgs]
                
                # Copy images to fold directories
                self._copy_images(train_images, folds_dir / f'fold_{fold}' / 'train' / cls)
                self._copy_images(val_images, folds_dir / f'fold_{fold}' / 'val' / cls)
                
                # Update statistics
                stats[f'fold_{fold}']['train'][cls] = len(train_images)
                stats[f'fold_{fold}']['val'][cls] = len(val_images)
        
        # Log statistics
        for fold in range(n_folds):
            fold_name = f'fold_{fold}'
            self.logger.info(f"Fold {fold}: ")
            for split in ['train', 'val']:
                total = sum(stats[fold_name][split].values())
                self.logger.info(f"  {split.capitalize()}: {total} images")
        
        return stats
    
    def visualize_distribution(self, save_path=None):
        """Visualize the data distribution across splits.
        
        Args:
            save_path: Path to save the visualization, or None to display
            
        Returns:
            Path to saved visualization if save_path is provided
        """
        self.logger.info("Visualizing data distribution...")
        
        # Collect statistics
        stats = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.target_dir / split
            if not split_dir.exists():
                continue
                
            stats[split] = {}
            for cls in self.class_names:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    stats[split][cls] = len(list(cls_dir.glob('*.jpg')))
                else:
                    stats[split][cls] = 0
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.8 / len(stats)
        
        # Set positions of bars on x-axis
        r = np.arange(len(self.class_names))
        
        # Plot bars
        for i, (split, counts) in enumerate(stats.items()):
            values = [counts.get(cls, 0) for cls in self.class_names]
            plt.bar(r + i * bar_width, values, width=bar_width, label=split.capitalize())
        
        # Add labels and legend
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Data Distribution Across Splits')
        plt.xticks(r + bar_width * (len(stats) - 1) / 2, self.class_names)
        plt.legend()
        
        # Save or show
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Saved distribution visualization to {save_path}")
            return save_path
        else:
            plt.show()
            return None
