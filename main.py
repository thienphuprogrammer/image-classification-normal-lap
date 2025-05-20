import os
import tensorflow as tf
from src.data_preprocessing.preprocessor import DataPreprocessor
from src.data_ingestion.data_loader import DataLoader
from src.data_engineering.feature_engineering import FeatureEngineer
from src.models.cnn_model import CNNModel
from src.models.advanced_models import AdvancedCNNModel
from src.evalution.model_evaluator import ModelEvaluator
from src.data_engineering.data_balancer import DataBalancer
import matplotlib.pyplot as plt
import numpy as np 

def plot_optimizer_comparison(histories, optimizers):
    """Plot comparison of different optimizers"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for opt_name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{opt_name} - Training')
        plt.plot(history.history['val_accuracy'], label=f'{opt_name} - Validation')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for opt_name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{opt_name} - Training')
        plt.plot(history.history['val_loss'], label=f'{opt_name} - Validation')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Initialize data loader and prepare dataset
    data_loader = DataLoader(
        dataset_name="puneet6060/intel-image-classification",
        data_dir="data"
    )
    
    try:
        data_loader.prepare_dataset()
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return
    
    # Initialize data balancer
    balancer = DataBalancer(
        data_dir="data/seg_train/seg_train",
        target_dir="data/balanced_dataset",
        target_size=(150, 150)
    )
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    balancer.create_balanced_dataset(target_samples_per_class=2000)  # Adjust this number based on your needs
    balancer.verify_balanced_dataset()
    
    # Initialize data preprocessor with balanced dataset
    preprocessor = DataPreprocessor(
        data_dir="data/balanced_dataset",
        img_size=(150, 150),
        batch_size=32
    )
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(
        data_dir="data/balanced_dataset",
        img_size=(150, 150)
    )
    
    # Prepare feature datasets
    print("\nPreparing feature datasets...")
    feature_engineer.prepare_feature_datasets()
    
    # Enhanced data analysis and visualization
    print("\nAnalyzing dataset statistics...")
    preprocessor.analyze_image_statistics()
    preprocessor.visualize_data_distribution()
    preprocessor.visualize_sample_images()
    
    # Create data generators
    train_generator = preprocessor.create_data_generator('train')
    val_generator = preprocessor.create_data_generator('val')
    test_generator = preprocessor.create_data_generator('test')
    
    # Initialize models
    basic_cnn = CNNModel(num_classes=6)
    improved_cnn = CNNModel(num_classes=6, improved=True)
    
    # Train basic CNN
    print("\nTraining basic CNN...")
    basic_history = basic_cnn.train(
        train_generator,
        val_generator,
        epochs=20
    )
    
    # Train improved CNN
    print("\nTraining improved CNN...")
    improved_history = improved_cnn.train(
        train_generator,
        val_generator,
        epochs=30
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluator.plot_training_history(basic_history, "Basic CNN")
    evaluator.plot_training_history(improved_history, "Improved CNN")
    
    # Evaluate on test set
    basic_metrics = evaluator.evaluate_model(basic_cnn.model, test_generator, "Basic CNN")
    improved_metrics = evaluator.evaluate_model(improved_cnn.model, test_generator, "Improved CNN")
    
    # Train and evaluate advanced models
    print("\nTraining advanced models...")
    advanced_models = AdvancedCNNModel(num_classes=6)
    
    # Train models with different optimizers
    histories = {}
    for optimizer_name in ['adam', 'rmsprop', 'sgd']:
        print(f"\nTraining with {optimizer_name} optimizer...")
        history = advanced_models.train(
            train_generator,
            val_generator,
            optimizer_name=optimizer_name,
            epochs=30
        )
        histories[optimizer_name] = history
    
    # Plot optimizer comparison
    evaluator.plot_optimizer_comparison(histories)
    
    # Evaluate advanced models
    for optimizer_name in ['adam', 'rmsprop', 'sgd']:
        metrics = evaluator.evaluate_model(
            advanced_models.models[optimizer_name],
            test_generator,
            f"Advanced CNN ({optimizer_name})"
        )

if __name__ == "__main__":
    main() 