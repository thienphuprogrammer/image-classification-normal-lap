import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.regularizers import l2
from src.models.base_cnn import BaseCNN

class ResNetImproved(BaseCNN):
    """Optimized ResNet implementation with performance improvements and transfer learning"""

    def __init__(self, *args, use_transfer_learning=True, resnet_version=50, **kwargs):
        """Initialize the ResNetImproved model.
        
        Args:
            use_transfer_learning: Whether to use transfer learning from pre-trained ResNet
            resnet_version: ResNet version to use (50, 101, or 152)
            *args, **kwargs: Arguments passed to the parent class
        """
        super().__init__(*args, model_name=f"ResNet{resnet_version}Improved", **kwargs)
        self.use_transfer_learning = use_transfer_learning
        self.resnet_version = resnet_version
        self.model = self._build_model()
        self._compile_model()

    def _residual_block(self, x, filters, stride=1, activation='relu'):
        """Bottleneck residual block with improved regularization.
        
        Args:
            x: Input tensor
            filters: Number of filters for the first two convolutions
            stride: Stride for the first convolution
            activation: Activation function to use
            
        Returns:
            Output tensor after applying the residual block
        """
        shortcut = x

        # Bottleneck path
        x = layers.Conv2D(
            filters, 1, 
            strides=stride, 
            padding='same',
            kernel_regularizer=l2(self.weight_decay),
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            filters, 3, 
            padding='same',
            kernel_regularizer=l2(self.weight_decay),
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        x = layers.Conv2D(
            filters * 4, 1, 
            padding='same',
            kernel_regularizer=l2(self.weight_decay),
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)

        # Project shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters * 4:
            shortcut = layers.Conv2D(
                filters * 4, 1, 
                strides=stride,
                kernel_regularizer=l2(self.weight_decay),
                use_bias=False
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add shortcut connection
        x = layers.add([x, shortcut])
        return layers.Activation(activation)(x)

    def _build_model(self):
        """Build the model architecture.
        
        Returns:
            A Keras Model instance
        """
        if self.use_transfer_learning:
            return self._build_transfer_model()
        else:
            return self._build_custom_model()
            
    def _build_transfer_model(self):
        """Build a model using transfer learning from pre-trained ResNet.
        
        Returns:
            A Keras Model instance
        """
        # Select the appropriate ResNet version
        if self.resnet_version == 50:
            base_model = applications.ResNet50V2(
                weights='imagenet', include_top=False, input_shape=self.input_shape
            )
        elif self.resnet_version == 101:
            base_model = applications.ResNet101V2(
                weights='imagenet', include_top=False, input_shape=self.input_shape
            )
        elif self.resnet_version == 152:
            base_model = applications.ResNet152V2(
                weights='imagenet', include_top=False, input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported ResNet version: {self.resnet_version}")
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create new model on top
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (inline during training)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Preprocessing required by ResNet
        x = layers.Rescaling(1./255)(x)
        x = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
        
        # Pass through the base model
        x = base_model(x, training=False)
        
        # Add custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Add a few dense layers with dropout
        x = layers.Dense(1024, kernel_regularizer=l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Add classification layer
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        # Assemble the model
        model = models.Model(inputs, outputs)
        
        return model
        
    def _build_custom_model(self):
        """Build a custom ResNet architecture without transfer learning.
        
        Returns:
            A Keras Model instance
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers (inline during training)
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        
        # Normalize inputs
        x = layers.Rescaling(1./255)(x)
        
        # Initial layers with improved regularization
        x = layers.Conv2D(
            64, 7, strides=2, padding='same', 
            kernel_regularizer=l2(self.weight_decay)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Residual stages - more blocks for deeper network
        # Stage 1
        x = self._residual_block(x, 64, stride=1)
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)

        # Stage 2
        x = self._residual_block(x, 128, stride=2)
        x = self._residual_block(x, 128)
        x = self._residual_block(x, 128)
        x = self._residual_block(x, 128)

        # Stage 3
        x = self._residual_block(x, 256, stride=2)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)

        # Stage 4
        x = self._residual_block(x, 512, stride=2)
        x = self._residual_block(x, 512)
        x = self._residual_block(x, 512)

        # Final layers with dropout for regularization
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            kernel_regularizer=l2(self.weight_decay),
            dtype='float32'
        )(x)

        return models.Model(inputs, outputs)

    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        # Use different optimizers based on whether we're using transfer learning
        if self.use_transfer_learning:
            # For transfer learning, use Adam with lower learning rate
            optimizer = optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0  # Gradient clipping for stability
            )
        else:
            # For custom model, use SGD with momentum and nesterov
            optimizer = optimizers.SGD(
                learning_rate=0.01,  # Start with higher learning rate
                momentum=0.9,
                nesterov=True
            )
            
        # Compile the model with additional metrics
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

    def train(self, train_data, val_data, epochs=100, use_scheduler=True):
        """Enhanced training method with gradual learning rate and mixed precision.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs for training
            use_scheduler: Whether to use learning rate scheduler
            
        Returns:
            Training history
        """
        # Define learning rate scheduler if requested
        if use_scheduler and not self.use_transfer_learning:
            def lr_scheduler(epoch):
                if epoch < 5: return 0.01  # Warm-up phase
                if epoch < 30: return 0.1   # Main training phase
                if epoch < 60: return 0.01  # First decay
                return 0.001               # Final fine-tuning
                
            # Add scheduler to callbacks
            train_callbacks = self.callbacks + [
                tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
            ]
        else:
            train_callbacks = self.callbacks
            
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=train_callbacks,
            verbose=1
        )
        
        # Visualize training history
        self.visualize_training_history(history)
        
        return history
        
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
            print("Fine-tuning is only applicable for transfer learning models.")
            return None
            
        # Get the base model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.models.Model):
                base_model = layer
                break
                
        if base_model is None:
            print("Could not find base model for fine-tuning.")
            return None
            
        # Unfreeze the top layers of the base model
        base_model.trainable = True
        
        # Freeze all the layers except the last unfreeze_layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
            
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # Add callbacks specific for fine-tuning
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
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=self.callbacks + fine_tune_callbacks
        )
        
        # Visualize fine-tuning history
        self.visualize_training_history(history)
        
        return history