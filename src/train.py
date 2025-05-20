#!/usr/bin/env python3

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import our optimized components
from src.data_engineering.data_distributor import DataDistributor
from src.data_preprocessing.preprocessor import DataPreprocessor
from src.models import EfficientCNN, ResNetImproved

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train image classification models')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory for processed dataset')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--redistribute', action='store_true',
                        help='Redistribute the dataset before training')
    
    # Model arguments
    parser.add_argument('--model', type=str, choices=['efficient', 'resnet'], default='efficient',
                        help='Model architecture to use')
    parser.add_argument('--transfer-learning', action='store_true',
                        help='Use transfer learning')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune the model after initial training')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--fine-tune-lr', type=float, default=0.0001,
                        help='Learning rate for fine-tuning')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_gpu(gpu_id):
    """Set up GPU for training."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Only use specified GPU
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.warning("No GPUs found, using CPU")

def redistribute_data(args):
    """Redistribute the dataset for optimal training."""
    logger.info("Redistributing dataset...")
    
    # Initialize data distributor
    distributor = DataDistributor(
        source_dir=args.data_dir,
        target_dir=args.processed_dir,
    )
    
    # Analyze source data
    analysis = distributor.analyze_source_data()
    logger.info(f"Source data analysis: {analysis['total_images']} total images")
    
    # Redistribute data
    stats = distributor.redistribute_data(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify=True,
        seed=args.seed
    )
    
    # Create cross-validation folds
    cv_stats = distributor.create_cross_validation_folds(n_folds=5, seed=args.seed)
    
    # Visualize distribution
    distributor.visualize_distribution(save_path='logs/data_distribution.png')
    
    logger.info("Data redistribution complete")
    return stats

def train_model(args, data_stats=None):
    """Train the model with the specified architecture."""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(
        data_dir=args.processed_dir if args.redistribute else args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Create datasets
    train_ds = preprocessor.create_tf_dataset(split='train', shuffle=True, augment=True)
    val_ds = preprocessor.create_tf_dataset(split='val', shuffle=False, augment=False)
    test_ds = preprocessor.create_tf_dataset(split='test', shuffle=False, augment=False)
    
    # Get class weights if available
    class_weights = preprocessor.compute_class_weights() if hasattr(preprocessor, 'compute_class_weights') else None
    
    # Create model
    num_classes = preprocessor.num_classes
    input_shape = (args.img_size, args.img_size, 3)
    
    if args.model == 'efficient':
        model = EfficientCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            use_transfer_learning=args.transfer_learning,
            use_mixed_precision=args.mixed_precision,
            learning_rate=args.lr
        )
    else:  # resnet
        model = ResNetImproved(
            input_shape=input_shape,
            num_classes=num_classes,
            version=50,  # Default to ResNet50
            use_transfer_learning=args.transfer_learning,
            learning_rate=args.lr
        )
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        logger.info("Enabling mixed precision training")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # Compile and build the model
    model.compile()
    model.build()
    
    # Train the model
    logger.info(f"Training {args.model} model for {args.epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights
    )
    
    # Fine-tune if requested
    if args.fine_tune and args.transfer_learning:
        logger.info(f"Fine-tuning model for {args.fine_tune_epochs} epochs with LR={args.fine_tune_lr}")
        if hasattr(model, 'fine_tune'):
            fine_tune_history = model.fine_tune(
                train_ds,
                validation_data=val_ds,
                epochs=args.fine_tune_epochs,
                learning_rate=args.fine_tune_lr,
                class_weight=class_weights
            )
            # Merge histories
            for key in fine_tune_history.history:
                history.history[key].extend(fine_tune_history.history[key])
    
    # Evaluate the model
    logger.info("Evaluating model on test set")
    test_results = model.evaluate(test_ds)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = Path(f"models/{args.model}_{timestamp}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(model_dir / "model.h5")
    logger.info(f"Model saved to {model_dir / 'model.h5'}")
    
    # Save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(model_dir / "training_history.png")
    
    # Save test results
    with open(model_dir / "test_results.txt", "w") as f:
        for metric, value in zip(model.model.metrics_names, test_results):
            f.write(f"{metric}: {value}\n")
            logger.info(f"Test {metric}: {value}")
    
    return history, test_results

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    set_seed(args.seed)
    setup_gpu(args.gpu)
    
    # Redistribute data if requested
    data_stats = None
    if args.redistribute:
        data_stats = redistribute_data(args)
    
    # Train model
    history, test_results = train_model(args, data_stats)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
