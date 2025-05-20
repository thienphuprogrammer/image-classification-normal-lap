import os
import logging
from pathlib import Path
from data_ingestion.data_loader import DataLoader
from data_engineering.data_balancer import DataBalancer
from data_preprocessing.preprocessor import DataPreprocessor
from data_engineering.feature_engineering import FeatureEngineer

class DataPipeline:
    def __init__(self, config):
        """
        Initialize the data pipeline.
        
        Args:
            config (dict): Configuration dictionary containing:
                - dataset_name: Name of the Kaggle dataset
                - data_dir: Base directory for data
                - target_size: Target image size (height, width)
                - batch_size: Batch size for data generators
                - target_samples: Target samples per class
                - validation_split: Validation split ratio
        """
        self.config = config
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """Create necessary directories for the pipeline."""
        self.raw_data_dir = os.path.join(self.config['data_dir'], 'raw')
        self.balanced_data_dir = os.path.join(self.config['data_dir'], 'balanced')
        self.feature_data_dir = os.path.join(self.config['data_dir'], 'features')
        
        for directory in [self.raw_data_dir, self.balanced_data_dir, self.feature_data_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Execute the complete data pipeline."""
        try:
            # Step 1: Data Loading
            self.logger.info("Step 1: Loading dataset...")
            self.load_data()
            
            # Step 2: Data Balancing
            self.logger.info("Step 2: Balancing dataset...")
            self.balance_data()
            
            # Step 3: Data Preprocessing
            self.logger.info("Step 3: Preprocessing data...")
            self.preprocess_data()
            
            # Step 4: Feature Engineering
            self.logger.info("Step 4: Engineering features...")
            self.engineer_features()
            
            self.logger.info("Data pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data pipeline: {str(e)}")
            return False
    
    def load_data(self):
        """Load and prepare the raw dataset."""
        data_loader = DataLoader(
            dataset_name=self.config['dataset_name'],
            data_dir=self.raw_data_dir
        )
        
        if not data_loader.prepare_dataset():
            raise Exception("Failed to prepare dataset")
        
        self.logger.info("Dataset loaded successfully")
    
    def balance_data(self):
        """Balance the dataset using DataBalancer."""
        balancer = DataBalancer(
            data_dir=os.path.join(self.raw_data_dir, 'seg_train/seg_train'),
            target_dir=self.balanced_data_dir,
            target_size=self.config['target_size']
        )
        
        # Analyze original distribution
        original_distribution = balancer.analyze_class_distribution()
        self.logger.info("Original class distribution:")
        for class_name, count in original_distribution.items():
            self.logger.info(f"{class_name}: {count} images")
        
        # Create balanced dataset
        balancer.create_balanced_dataset(
            target_samples_per_class=self.config['target_samples'],
            validation_split=self.config['validation_split']
        )
        
        # Verify balanced distribution
        balancer.verify_balanced_dataset()
    
    def preprocess_data(self):
        """Preprocess the balanced dataset."""
        preprocessor = DataPreprocessor(
            data_dir=self.balanced_data_dir,
            img_size=self.config['target_size'],
            batch_size=self.config['batch_size']
        )
        
        # Analyze dataset statistics
        preprocessor.analyze_image_statistics()
        
        # Visualize data distribution
        preprocessor.visualize_data_distribution()
        
        # Visualize sample images
        preprocessor.visualize_sample_images()
        
        # Create data generators
        self.train_generator = preprocessor.create_data_generator('train')
        self.val_generator = preprocessor.create_data_generator('val')
        self.test_generator = preprocessor.create_data_generator('test')
    
    def engineer_features(self):
        """Extract and save features from the dataset."""
        feature_engineer = FeatureEngineer(
            data_dir=self.balanced_data_dir,
            img_size=self.config['target_size']
        )
        
        # Prepare feature datasets
        feature_engineer.prepare_feature_datasets()
        
        # Save feature statistics
        self.save_feature_statistics(feature_engineer)
    
    def save_feature_statistics(self, feature_engineer):
        """Save feature statistics to a file."""
        stats_file = os.path.join(self.feature_data_dir, 'feature_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Feature Statistics\n")
            f.write("=================\n\n")
            
            # Add feature statistics here
            f.write("Feature extraction completed successfully\n")
    
    def get_data_generators(self):
        """Return the data generators for training."""
        return {
            'train': self.train_generator,
            'val': self.val_generator,
            'test': self.test_generator
        }

def main():
    # Configuration for the pipeline
    config = {
        'dataset_name': 'puneet6060/intel-image-classification',
        'data_dir': 'data',
        'target_size': (150, 150),
        'batch_size': 32,
        'target_samples': 2000,  # Adjust based on your needs
        'validation_split': 0.2
    }
    
    # Create and run the pipeline
    pipeline = DataPipeline(config)
    success = pipeline.run()
    
    if success:
        print("Data pipeline completed successfully!")
    else:
        print("Data pipeline failed!")

if __name__ == "__main__":
    main()
