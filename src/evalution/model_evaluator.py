import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model, test_generator, class_names):
        self.model = model
        self.test_generator = test_generator
        self.class_names = class_names
        
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self):
        """Create and plot confusion matrix"""
        # Get predictions
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
        
    def print_classification_metrics(self):
        """Print precision, recall, and F1-score for each class"""
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
    def evaluate_model(self):
        """Evaluate model and print test accuracy"""
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
    def visualize_predictions(self, num_samples=5):
        """Visualize sample predictions"""
        # Get a batch of test images
        test_images, test_labels = next(self.test_generator)
        
        # Make predictions
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Plot sample predictions
        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(test_images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(test_images[i])
            plt.title(f'True: {self.class_names[true_classes[i]]}\nPred: {self.class_names[predicted_classes[i]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show() 