from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

class BaseCNN:
    """Base class with common functionality for CNN models.
    
    This class provides common methods and properties used by all CNN models
    in the project, including callbacks configuration, visualization tools,
    and utility methods.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=6, model_name=None):
        """Initialize the base CNN model.
        
        Args:
            input_shape: Tuple of (height, width, channels) for input images
            num_classes: Number of output classes
            model_name: Optional name for the model (used for saving)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = 1e-4
        self.model_name = model_name or self.__class__.__name__
        self.model = None
        self._configure_callbacks()

    def _configure_callbacks(self):
        """Configure common callbacks for model training.
        
        Sets up callbacks for early stopping, learning rate reduction,
        model checkpointing, and TensorBoard logging.
        """
        # Create directories for logs and model checkpoints
        self.log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.checkpoint_dir = os.path.join('models', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Configure callbacks
        self.callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # More patience for complex models
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when training plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More gradual reduction
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Save model checkpoints
            ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f'{self.model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,  # Log histograms of weights
                profile_batch='500,520',
                update_freq='epoch'
            )
        ]

    def _conv_block(self, x, filters, kernel_size=3, stride=1, use_se=False, activation='swish'):
        """Create an optimized convolution block with optional Squeeze-Excitation.
        
        Args:
            x: Input tensor
            filters: Number of filters in the convolution
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            use_se: Whether to use Squeeze-Excitation
            activation: Activation function to use ('relu' or 'swish')
            
        Returns:
            Output tensor after applying convolutions and optional SE
        """
        # Store input for potential residual connection
        input_tensor = x
        
        # Depthwise Separable Convolution
        x = layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding='same',
            depthwise_regularizer=l2(self.weight_decay),
            use_bias=False  # No bias needed before BatchNorm
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        # Pointwise Convolution
        x = layers.Conv2D(
            filters,
            1,
            padding='same',
            kernel_regularizer=l2(self.weight_decay),
            use_bias=False  # No bias needed before BatchNorm
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Apply Squeeze-Excitation if requested
        if use_se:
            # Squeeze-Excitation with improved ratio
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Dense(
                max(filters // 16, 8),  # Ensure minimum width
                activation=activation,
                kernel_regularizer=l2(self.weight_decay)
            )(se)
            se = layers.Dense(
                filters,
                activation='sigmoid',
                kernel_regularizer=l2(self.weight_decay)
            )(se)
            x = layers.multiply([x, se])
            
        # Apply activation after SE
        x = layers.Activation(activation)(x)

        return x
        
    def visualize_training_history(self, history):
        """Visualize the training history with plots.
        
        Args:
            history: History object returned by model.fit()
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')
        
        # Save the figure
        plt.tight_layout()
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(f'visualizations/{self.model_name}_training_history.png')
        plt.close()
        
    def evaluate_model(self, test_data, verbose=1):
        """Evaluate the model on test data and print detailed metrics.
        
        Args:
            test_data: Test dataset
            verbose: Verbosity mode
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            print("Error: Model has not been built yet.")
            return None
            
        # Evaluate the model
        results = self.model.evaluate(test_data, verbose=verbose)
        metrics = dict(zip(self.model.metrics_names, results))
        
        # Print detailed results
        print("\n=== Model Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Classes: {self.num_classes}")
        print("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
            
        return metrics
        
    def save_model(self, filepath=None):
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model, or None to use default path
        
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            print("Error: Model has not been built yet.")
            return None
            
        # Create default filepath if not provided
        if filepath is None:
            os.makedirs('models/saved', exist_ok=True)
            filepath = f'models/saved/{self.model_name}.h5'
            
        # Save the model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        return filepath
        
    def load_model(self, filepath):
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False