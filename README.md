# Image Classification with Convolutional Neural Networks

This project implements a comprehensive image classification system using Convolutional Neural Networks (CNNs) for the Intel Image Classification dataset. The implementation includes both basic and advanced architectures, along with extensive data preprocessing and evaluation capabilities.

## Project Structure

```
src/
├── data_engineering/
│   └── feature_engineering.py    # Advanced feature extraction
├── data_ingestion/
│   └── data_loader.py           # Dataset loading and preparation
├── data_preprocessing/
│   └── preprocessor.py          # Data preprocessing and augmentation
├── models/
│   ├── cnn_model.py            # Basic and improved CNN architectures
│   └── advanced_models.py      # ResNet and custom loss implementations
├── evaluation/
│   └── model_evaluator.py      # Model evaluation and visualization
└── main.py                     # Main execution script
```

## Implementation Details

### Part 1: Core Requirements

#### Task 1.1: Dataset Exploration and Preprocessing
- **Loading and Preprocessing**
  - Automatic dataset download from Kaggle
  - Image resizing and normalization
  - Validation split creation
  - Dataset structure verification

- **Data Augmentation Techniques**
  - Random rotation and flipping
  - Brightness and contrast adjustments
  - Zoom and shear transformations
  - Feature-wise normalization

- **Dataset Visualization and Analysis**
  - Class distribution analysis
  - Image statistics visualization
  - Sample image display with augmentations
  - Feature distribution analysis

#### Task 1.2: Custom CNN Architecture
- **Initial CNN Architecture**
  - Basic CNN with 3 convolutional layers
  - MaxPooling and Dropout for regularization
  - Dense layers for classification
  - Initial configuration with Adam optimizer

- **Model Architecture Improvement**
  - Increased network depth
  - Additional convolutional layers
  - Batch normalization
  - Improved regularization techniques

- **Hyperparameter Tuning**
  - Learning rate optimization
  - Batch size selection
  - Dropout rate adjustment
  - Early stopping implementation

#### Task 1.3: Performance Analysis
- **Training/Validation Analysis**
  - Accuracy and loss plots
  - Learning curve analysis
  - Overfitting detection
  - Model convergence monitoring

- **Model Evaluation**
  - Confusion matrix visualization
  - Precision, recall, and F1-score calculation
  - Per-class performance analysis
  - Error analysis and visualization

### Part 2: Advanced Challenges

#### Advanced Architectures
- **ResNet Implementation**
  - Residual connections
  - Skip connections
  - Batch normalization
  - Global average pooling

- **Custom Loss Functions**
  - Focal loss implementation
  - Class-weighted loss
  - Custom metrics

#### Model Comparison
- Performance comparison between architectures
- Training time analysis
- Resource utilization
- Accuracy vs. complexity trade-off

## Requirements

```
tensorflow>=2.8.0
numpy>=1.19.5
matplotlib>=3.4.3
seaborn>=0.11.2
scikit-learn>=0.24.2
opencv-python>=4.5.3
kaggle>=1.5.12
tqdm>=4.62.3
pillow>=8.3.1
pandas>=1.3.3
```

## Setup and Execution

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Kaggle API credentials:
   - Create a Kaggle account
   - Generate API token
   - Place kaggle.json in ~/.kaggle/

3. Run the main script:
```bash
python src/main.py
```

## Report Structure

The accompanying report (2-6 pages) includes:

1. **Approach Documentation**
   - Detailed methodology for each task
   - Implementation decisions and rationale
   - Challenges faced and solutions

2. **Results Analysis**
   - Performance metrics and visualizations
   - Model comparison charts
   - Error analysis and insights

3. **Architecture Justification**
   - CNN architecture choices
   - Hyperparameter selection rationale
   - Advanced model implementation details

4. **Answers to Questions**
   - Q1: Analysis of model performance and improvements
   - Q2: Justification of advanced architecture choices

## Submission

The submission includes:
1. All Python scripts and Jupyter notebooks
2. Comprehensive report (2-6 pages)
3. Requirements.txt
4. README.md

Files can be submitted as:
- A single .zip or .rar file
- Google Drive link
- GitHub repository link

## Grading Breakdown

### Part 1: Core Requirements (8 points)
- Task 1.1: Dataset Exploration and Preprocessing (2 points)
- Task 1.2: Custom CNN Architecture (4 points)
- Task 1.3: Performance Analysis (2 points)

### Part 2: Advanced Challenges (2 points)
- Advanced architectures and techniques (1 point)
- Evaluation and comparison (0.5 point)
- Explanation and justification (0.5 point)

Note: Failure to submit a report may result in up to 30% deduction in points.