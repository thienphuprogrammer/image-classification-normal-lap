import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class FeatureEngineer:
    def __init__(self, data_dir, img_size=(150, 150)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_color_features(self, image):
        """Extract color-based features from image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate color statistics
        features = []
        for color_space in [image, hsv, lab]:
            for channel in cv2.split(color_space):
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.median(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])
        
        return np.array(features)
    
    def extract_texture_features(self, image):
        """Extract texture features using GLCM"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate GLCM
        glcm = self._calculate_glcm(gray)
        
        # Extract texture features
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.append(self._calculate_glcm_property(glcm, prop))
        
        return np.array(features)
    
    def _calculate_glcm(self, gray_image):
        """Calculate Gray-Level Co-occurrence Matrix"""
        # Quantize image to 8 levels
        gray_quantized = (gray_image / 32).astype(np.uint8)
        
        # Calculate GLCM
        glcm = np.zeros((8, 8), dtype=np.uint32)
        for i in range(gray_quantized.shape[0]-1):
            for j in range(gray_quantized.shape[1]-1):
                glcm[gray_quantized[i,j], gray_quantized[i+1,j]] += 1
                glcm[gray_quantized[i,j], gray_quantized[i,j+1]] += 1
        
        return glcm
    
    def _calculate_glcm_property(self, glcm, prop):
        """Calculate GLCM properties"""
        if prop == 'contrast':
            return np.sum((np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1]))**2 * glcm)
        elif prop == 'dissimilarity':
            return np.sum(np.abs(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])) * glcm)
        elif prop == 'homogeneity':
            return np.sum(glcm / (1 + (np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1]))**2))
        elif prop == 'energy':
            return np.sqrt(np.sum(glcm**2))
        elif prop == 'correlation':
            mean_i = np.sum(np.sum(glcm, axis=1) * np.arange(glcm.shape[0]))
            mean_j = np.sum(np.sum(glcm, axis=0) * np.arange(glcm.shape[1]))
            std_i = np.sqrt(np.sum(np.sum(glcm, axis=1) * (np.arange(glcm.shape[0]) - mean_i)**2))
            std_j = np.sqrt(np.sum(np.sum(glcm, axis=0) * (np.arange(glcm.shape[1]) - mean_j)**2))
            return np.sum((np.arange(glcm.shape[0])[:, None] - mean_i) * 
                         (np.arange(glcm.shape[1]) - mean_j) * glcm) / (std_i * std_j)
    
    def extract_shape_features(self, image):
        """Extract shape-based features"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate shape features
        features = []
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate features
            features.extend([
                cv2.contourArea(largest_contour),
                cv2.arcLength(largest_contour, True),
                cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1])
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def process_image(self, image_path):
        """Process a single image and extract all features"""
        try:
            # Read and resize image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            
            # Extract features
            color_features = self.extract_color_features(image)
            texture_features = self.extract_texture_features(image)
            shape_features = self.extract_shape_features(image)
            
            # Combine all features
            features = np.concatenate([color_features, texture_features, shape_features])
            
            return features
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def extract_features_batch(self, image_paths, batch_size=32):
        """Process a batch of images in parallel"""
        features_list = []
        
        with ThreadPoolExecutor() as executor:
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
                batch_paths = image_paths[i:i + batch_size]
                batch_features = list(executor.map(self.process_image, batch_paths))
                features_list.extend([f for f in batch_features if f is not None])
        
        return np.array(features_list)
    
    def normalize_features(self, features):
        """Normalize features using StandardScaler"""
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    
    def create_feature_dataset(self, split='train'):
        """Create feature dataset for a specific split"""
        split_dir = self.data_dir / split
        all_features = []
        all_labels = []
        
        for class_idx, class_name in enumerate(['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']):
            class_dir = split_dir / class_name
            image_paths = list(class_dir.glob('*.jpg'))
            
            self.logger.info(f"Processing {len(image_paths)} images for class {class_name}")
            features = self.extract_features_batch(image_paths)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend([class_idx] * len(features))
        
        if all_features:
            X = np.vstack(all_features)
            y = np.array(all_labels)
            
            # Normalize features
            X = self.normalize_features(X)
            
            return X, y
        else:
            self.logger.error(f"No features extracted for split {split}")
            return None, None
    
    def save_features(self, X, y, split):
        """Save extracted features to disk"""
        output_dir = self.data_dir / 'features'
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / f'X_{split}.npy', X)
        np.save(output_dir / f'y_{split}.npy', y)
        
        self.logger.info(f"Saved features for {split} split")
    
    def prepare_feature_datasets(self):
        """Prepare feature datasets for all splits"""
        for split in ['train', 'validation', 'test']:
            self.logger.info(f"Preparing features for {split} split")
            X, y = self.create_feature_dataset(split)
            
            if X is not None and y is not None:
                self.save_features(X, y, split)
            else:
                self.logger.error(f"Failed to prepare features for {split} split") 