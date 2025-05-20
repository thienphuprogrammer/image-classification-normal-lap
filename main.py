import tensorflow as tf
from pathlib import Path
import os

# Configuration
CONFIG = {
    "data_dir": Path("data/intel_images"),
    "balanced_dir": Path("data/balanced"),
    "img_size": (224, 224),
    "batch_size": 64,
    "num_classes": 6,
    "class_names": ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'],
    "epochs": 50,
    "eval_save_dir": "evaluation_results"
}


def main():
    # --------------------------
    # 1. Data Preparation Pipeline
    # --------------------------
    print("\n=== Data Preparation ===")
    from src.data_ingestion.data_loader import DataLoader
    from src.data_engineering.data_balancer import DataBalancer

    # Initialize and prepare dataset
    loader = DataLoader(CONFIG['data_dir'])
    if not loader.prepare_dataset():
        raise RuntimeError("Data preparation failed")

    # Balance dataset
    balancer = DataBalancer(
        data_dir=CONFIG['data_dir'],  # Point to original dataset
        target_dir=CONFIG['balanced_dir'],
        img_size=CONFIG['img_size']
    )

    # Balance all splits (train, val, test)
    for split in ['train', 'val', 'test']:
        for cls in CONFIG['class_names']:
            balancer.balance_class(
                source_dir=(CONFIG['data_dir'] / f'seg_{split}' if split == 'val' else CONFIG['data_dir'] / f'seg_{split}' / f'seg_{split}'),
                target_dir=CONFIG['balanced_dir'] / split,
                cls=cls,
                target_count=2000 if split == 'train' else 400  # Adjust counts
            )
            print(f"Balanced {cls} in {split} split.")
    print("Dataset balancing completed.")
    print(f"Balanced dataset saved to {CONFIG['balanced_dir']}")

    # --------------------------
    # 2. Data Preprocessing Pipeline
    # --------------------------
    print("\n=== Data Preprocessing ===")
    from src.data_preprocessing.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(
        data_dir=CONFIG['balanced_dir'],  # Point to balanced dataset
        img_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size']
    )

    # Create datasets for all splits
    train_ds = preprocessor.create_tf_dataset('train')
    val_ds = preprocessor.create_tf_dataset('val')
    test_ds = preprocessor.create_tf_dataset('test')

    # --------------------------
    # 3. Model Training Pipeline
    # --------------------------
    print("\n=== Model Training ===")
    from src.models import EfficientCNN, ResNetImproved

    # Initialize model
    model = ResNetImproved(
        input_shape=CONFIG['img_size'] + (3,),
        num_classes=CONFIG['num_classes']
    )

    # Train model
    history = model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG['epochs'],
        callbacks=model.callbacks
    )

    # Save model
    model_path = "saved_models/best_model"
    model.model.save(model_path)
    print(f"Model saved to {model_path}")

    # --------------------------
    # 4. Model Evaluation Pipeline
    # --------------------------
    print("\n=== Model Evaluation ===")
    from src.evalution.model_evaluator import ModelEvaluator

    # Create test dataset
    test_ds = preprocessor.create_tf_dataset('test')
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Load best model for evaluation
    best_model = tf.keras.models.load_model(model_path)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=best_model,
        dataset=test_ds,
        class_names=CONFIG['class_names']
    )

    # Run full evaluation
    results = evaluator.full_evaluation_pipeline(
        save_dir=CONFIG['eval_save_dir']
    )

    # Print key metrics
    print("\n=== Final Metrics ===")
    print(f"Test Accuracy: {results['accuracy']:.2%}")
    print(f"Inference Throughput: {results['throughput']:.1f} samples/sec")
    print(f"Macro F1-Score: {results['classification_report']['macro avg']['f1-score']:.2%}")

    print(f"Evaluation results saved to {CONFIG['eval_save_dir']}/evaluation_results.json")

    # save training history
    evaluator.plot_training_history(
        history,
        save_path=os.path.join(CONFIG['eval_save_dir'], "training_history.png")
    )

    history.models.save(
        os.path.join(CONFIG['eval_save_dir'], "training_history.json"),
        overwrite=True,
        include_optimizer=False
    )


if __name__ == "__main__":
    # Create directory structure
    for d in [CONFIG['data_dir'], CONFIG['balanced_dir'], CONFIG['eval_save_dir']]:
        os.makedirs(d, exist_ok=True)

    # Run pipeline
    main()