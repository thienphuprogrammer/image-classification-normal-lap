from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.mixed_precision import Policy
from src.models.base_cnn import BaseCNN

class EfficientCNN(BaseCNN):
    """Optimized CNN with modern architectural features and transfer learning"""

    def __init__(self, *args, use_mixed_precision=True, use_transfer_learning=True, **kwargs):
        """Initialize the EfficientCNN model.
        
        Args:
            use_mixed_precision: Whether to use mixed precision training for faster performance
            use_transfer_learning: Whether to use transfer learning from EfficientNet
            *args, **kwargs: Arguments passed to the parent class
        """
        super().__init__(*args, **kwargs)
        self.use_mixed_precision = use_mixed_precision
        self.use_transfer_learning = use_transfer_learning
        
        # Enable mixed precision if requested (speeds up training on compatible GPUs)
        if self.use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self):
        """Build the model architecture.
        
        Returns:
            A Keras Model instance
        """
        if self.use_transfer_learning:
            # Use EfficientNet as base model with transfer learning
            return self._build_transfer_model()
        else:
            # Use custom architecture
            return self._build_custom_model()
            
    def _build_transfer_model(self):
        """Build a model using transfer learning from EfficientNet.
        
        Returns:
            A Keras Model instance
        """
        # Create base model from EfficientNetB0 with pre-trained weights
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create new model on top
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (inline during training)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocessing required by EfficientNet
        x = layers.Rescaling(1./255)(x)  # Normalize to [0,1]
        x = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
        
        # Pass through the base model
        x = base_model(x, training=False)
        
        # Add custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Add a few dense layers with dropout
        x = layers.Dense(512, kernel_regularizer=l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)  # EfficientNet uses swish activation
        x = layers.Dropout(0.5)(x)
        
        # Add classification layer
        # Always use float32 for the final layer for numerical stability
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        # Assemble the model
        model = models.Model(inputs, outputs)
        
        return model
        
    def _build_custom_model(self):
        """Build a custom CNN architecture without transfer learning.
        
        Returns:
            A Keras Model instance
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (inline during training)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Normalize inputs
        x = layers.Rescaling(1./255)(x)
        
        # Initial stem with larger filters
        x = layers.Conv2D(64, 3, strides=2, padding='same', kernel_regularizer=l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)  # Using swish activation for better performance
        
        # Efficient blocks with residual connections
        # Block 1
        residual = x
        x = self._conv_block(x, 128, stride=2, use_se=True)
        x = layers.Dropout(0.2)(x)
        
        # Block 2
        residual = layers.Conv2D(256, 1, strides=2, padding='same')(residual)  # Match dimensions
        x = self._conv_block(x, 256, stride=2, use_se=True)
        x = layers.add([x, residual])  # Add residual connection
        x = layers.Activation('swish')(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 3
        residual = x
        x = self._conv_block(x, 512, stride=2, use_se=True)
        x = layers.add([x, residual])  # Add residual connection
        x = layers.Activation('swish')(x)
        x = layers.Dropout(0.4)(x)
        
        # Head with attention mechanism
        x = layers.GlobalAveragePooling2D()(x)
        
        # Self-attention mechanism
        attention = layers.Dense(512, activation='tanh', kernel_regularizer=l2(self.weight_decay))(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        x = layers.multiply([x, attention])
        
        # Final classification layers
        x = layers.Dense(1024, kernel_regularizer=l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        x = layers.Dropout(0.5)(x)
        
        # Always use float32 for the final layer for numerical stability
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        return models.Model(inputs, outputs)

    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        # Use a cosine decay learning rate schedule that's compatible with callbacks
        initial_learning_rate = 0.001
        
        # Use different optimizers based on whether we're using transfer learning
        if self.use_transfer_learning:
            optimizer = optimizers.AdamW(
                learning_rate=initial_learning_rate,
                weight_decay=1e-5,
                clipnorm=1.0  # Gradient clipping for stability
            )
        else:
            # For custom model, use a more aggressive learning rate
            optimizer = optimizers.AdamW(
                learning_rate=initial_learning_rate,
                weight_decay=2e-5,
                clipnorm=1.0
            )
            
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
    def fine_tune(self, train_data, val_data, epochs=10, unfreeze_layers=30):
        """Fine-tune the model after initial training.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs for fine-tuning
            unfreeze_layers: Number of layers to unfreeze from the base model
            
        Returns:
            Training history
        """
        if not self.use_transfer_learning:
            return None
            
        # Get the base model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.models.Model):
                base_model = layer
                break
                
        if base_model is None:
            return None
            
        # Unfreeze the top layers of the base model
        base_model.trainable = True
        
        # Freeze all the layers except the last unfreeze_layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
            
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-6),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Add early stopping specific for fine-tuning
        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train with fine-tuning
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=self.callbacks + fine_tune_callbacks
        )
