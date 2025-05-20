import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

class CustomLoss:
    """Custom loss function combining categorical crossentropy with focal loss"""
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        
    def focal_loss(self, y_true, y_pred):
        """Focal loss implementation"""
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_loss = -K.mean(self.alpha * K.pow(1. - pt, self.gamma) * K.log(pt))
        
        return focal_loss

class ResNetBlock(layers.Layer):
    """Residual block implementation"""
    def __init__(self, filters, kernel_size=3, stride=1, use_bias=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias)
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection
        self.shortcut = models.Sequential()
        if stride != 1 or filters != filters:
            self.shortcut = models.Sequential([
                layers.Conv2D(filters, 1, strides=stride, use_bias=use_bias),
                layers.BatchNormalization()
            ])
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        shortcut = self.shortcut(inputs)
        
        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)
        
        return x

class AdvancedCNNModel:
    def __init__(self, input_shape=(150, 150, 3), num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.custom_loss = CustomLoss()
        
    def create_resnet_model(self):
        """Create a ResNet-based model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = self._make_layer(x, 64, 2, stride=1)
        x = self._make_layer(x, 128, 2, stride=2)
        x = self._make_layer(x, 256, 2, stride=2)
        x = self._make_layer(x, 512, 2, stride=2)
        
        # Global average pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        # Compile with different optimizers
        optimizers_dict = {
            'adam': optimizers.Adam(learning_rate=0.001),
            'sgd': optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
            'rmsprop': optimizers.RMSprop(learning_rate=0.001, rho=0.9)
        }
        
        # Create three models with different optimizers
        models_dict = {}
        for opt_name, optimizer in optimizers_dict.items():
            model_copy = models.clone_model(model)
            model_copy.compile(
                optimizer=optimizer,
                loss=self.custom_loss.focal_loss,
                metrics=['accuracy']
            )
            models_dict[opt_name] = model_copy
        
        return models_dict
    
    def _make_layer(self, x, filters, blocks, stride=1):
        """Helper function to create a layer of residual blocks"""
        x = ResNetBlock(filters, stride=stride)(x)
        for _ in range(1, blocks):
            x = ResNetBlock(filters)(x)
        return x
    
    def get_callbacks(self):
        """Get callbacks for training"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ] 