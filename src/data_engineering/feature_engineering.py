import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import logging
from tqdm import tqdm


class FeatureEngineer:
    def __init__(self, img_size=(150, 150)):
        self.img_size = img_size
        self.glcms_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self, image_path):
        try:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            # Color features
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            color_features = np.concatenate([
                img.mean(axis=(0, 1)), img.std(axis=(0, 1)),
                hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))
            ])

            # GLCM features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            texture_features = np.concatenate([graycoprops(glcm, prop).ravel() for prop in self.glcms_props])

            return np.concatenate([color_features, texture_features])
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None

    def parallel_feature_extraction(self, image_paths, workers=8):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(tqdm(executor.map(self.extract_features, image_paths), total=len(image_paths)))
