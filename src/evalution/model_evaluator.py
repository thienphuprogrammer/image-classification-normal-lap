import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from time import time
from typing import Tuple, Dict, Any
import os


class ModelEvaluator:
    def __init__(self, model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list):
        """
        Optimized model evaluation class using tf.data.Dataset

        Args:
            model: Trained Keras model
            dataset: tf.data.Dataset yielding (images, labels)
            class_names: List of class names for visualization
        """
        self.model = model
        self.dataset = dataset
        self.class_names = class_names
        self._y_true = None
        self._y_pred = None
        self._cached_predictions = None

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cache predictions to avoid recomputation"""
        if self._cached_predictions is None:
            start_time = time()
            self._y_true = np.concatenate([y for _, y in self.dataset], axis=0)
            self._y_pred = self.model.predict(self.dataset, verbose=0)
            self._cached_predictions = (self._y_true, self._y_pred)
            print(f"Prediction time: {time() - start_time:.2f} seconds")

        return self._cached_predictions

    def plot_training_history(self, history: tf.keras.callbacks.History,
                              save_path: str = None) -> None:
        """Enhanced training history visualization with smoothing"""
        plt.figure(figsize=(12, 5))

        # Smoothing function
        def smooth(scalars: list, weight: float = 0.6) -> list:
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(smooth(history.history['accuracy']), label='Train')
        plt.plot(smooth(history.history['val_accuracy']), label='Validation')
        plt.title('Accuracy Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(smooth(history.history['loss']), label='Train')
        plt.plot(smooth(history.history['val_loss']), label='Validation')
        plt.title('Loss Evolution')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(self, normalize: bool = True,
                              save_path: str = None) -> None:
        """Enhanced confusion matrix with multiple view options"""
        y_true, y_pred = self._get_predictions()
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='viridis', xticklabels=self.class_names,
                    yticklabels=self.class_names, cbar=False)

        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_classification_report(self) -> Dict[str, Any]:
        """Enhanced classification report with additional metrics"""
        y_true, y_pred = self._get_predictions()
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        report = classification_report(y_true, y_pred,
                                       target_names=self.class_names,
                                       output_dict=True)

        # Add additional metrics
        report['macro_avg']['fpr'] = np.mean(
            [report[cls]['fpr'] for cls in self.class_names]
        )
        return report

    def visualize_predictions(self, num_samples: int = 5,
                              save_path: str = None) -> None:
        """Enhanced prediction visualization with confidence"""
        y_true, y_pred = self._get_predictions()
        indices = np.random.choice(len(y_true), num_samples)

        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(indices):
            image = self.dataset.unbatch().skip(idx).take(1).get_single_element()[0]
            true_label = self.class_names[np.argmax(y_true[idx])]
            pred_label = self.class_names[np.argmax(y_pred[idx])]
            confidence = np.max(y_pred[idx])

            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image.numpy().astype('uint8'))
            title_color = 'green' if true_label == pred_label else 'red'
            plt.title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.2f})',
                      color=title_color)
            plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def evaluate_performance(self) -> Dict[str, float]:
        """Comprehensive performance evaluation"""
        start_time = time()
        loss, accuracy = self.model.evaluate(self.dataset, verbose=0)
        inference_time = time() - start_time

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'inference_time': inference_time,
            'throughput': len(self.dataset) / inference_time
        }

    def full_evaluation_pipeline(self, save_dir: str = None) -> Dict[str, Any]:
        """Complete evaluation pipeline"""
        results = {}

        # Performance metrics
        results.update(self.evaluate_performance())

        # Classification report
        results['classification_report'] = self.generate_classification_report()

        # Visualization
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_training_history(save_path=os.path.join(save_dir, 'training_history.png'))
            self.plot_confusion_matrix(save_path=os.path.join(save_dir, 'confusion_matrix.png'))
            self.visualize_predictions(save_path=os.path.join(save_dir, 'predictions.png'))

        return results