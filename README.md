# ML-Assignment_1
# Complete Machine Learning Assignment Series - MT25081

## Assignment Overview
This repository contains my complete submission for the comprehensive Machine Learning assignment series covering data processing, algorithm implementation, model development, evaluation, and advanced ML techniques using the Crema-D emotion recognition dataset.

## Assignment Structure

### Q1: Emotion Recognition Dataset, Random Number Generator, and Normalization

#### q1a: Crema-D Dataset Loading
- `load_csv(path_to_csv)`: Loads CSV metadata using pandas
- `load_image(path_to_image_file)`: Loads images using OpenCV with RGB conversion  
- `load_audio(path_to_audio_file)`: Loads audio files using librosa with original sampling rate

#### q1b: Custom Random Number Generator
- Linear Congruential Generator (LCG) implementation
- Parameters: multiplier=1664525, increment=1013904223, modulus=2^32
- Supports 1D and multi-dimensional output shapes

#### q1c: Min-Max Scaling Normalization
- Scales numerical data to range [0, 1]
- Handles 1D, 2D, and 3D tensors/arrays
- Numerical stability with epsilon=1e-8

### Q2: Data Preprocessing and Feature Engineering

#### q2a: Data Cleaning and Validation
- Missing value imputation strategies
- Outlier detection and handling
- Data type validation and conversion
- Dataset consistency checks

#### q2b: Audio Feature Extraction
- MFCC (Mel-Frequency Cepstral Coefficients) extraction
- Spectral features (centroid, bandwidth, rolloff)
- Chroma features and RMS energy
- Zero-crossing rate and temporal features

#### q2c: Image Feature Extraction
- HOG (Histogram of Oriented Gradients)
- Color histograms in multiple color spaces
- LBP (Local Binary Patterns)
- CNN feature extraction from pre-trained models

#### q2d: Text Feature Extraction (if applicable)
- TF-IDF vectorization
- Word embeddings (Word2Vec, GloVe)
- Sentiment analysis features
- Linguistic features from transcriptions

#### q2e: Feature Selection
- Correlation-based feature selection
- Recursive Feature Elimination (RFE)
- Principal Component Analysis (PCA)
- Mutual information-based selection

#### q2f: Data Augmentation
- Audio: pitch shifting, time stretching, noise addition
- Image: rotation, flipping, cropping, color adjustments
- SMOTE for handling class imbalance
- Generative data augmentation techniques

#### q2g: Data Pipeline Implementation
- End-to-end preprocessing pipeline
- Batch processing for large datasets
- Caching mechanisms for efficiency
- Real-time data transformation

### Q3: Machine Learning Model Development

#### q3a: Baseline Models
- Logistic Regression with regularization
- Naive Bayes classifier
- K-Nearest Neighbors (KNN)
- Decision Trees with pruning

#### q3b: Ensemble Methods
- Random Forest with hyperparameter tuning
- Gradient Boosting Machines (GBM)
- AdaBoost with different base estimators
- Voting and stacking classifiers

#### q3c: Support Vector Machines
- Linear SVM with different C parameters
- Kernel SVM (RBF, polynomial, sigmoid)
- Multi-class strategies (one-vs-rest, one-vs-one)
- Custom kernel implementations

#### q3d: Neural Network Architectures
- CNN for image-based emotion recognition
- LSTM/GRU for sequential audio data
- Multimodal fusion architectures
- Attention mechanisms and transformer layers

#### q3e: Model Training Strategies
- Custom training loops with early stopping
- Learning rate scheduling (step, cosine, exponential)
- Gradient clipping and optimization techniques
- Regularization methods (dropout, batch normalization)

#### q3f: Hyperparameter Optimization
- Grid search with cross-validation
- Random search for large parameter spaces
- Bayesian optimization techniques
- Genetic algorithms for architecture search

### Q4: Model Evaluation and Validation

#### q4a: Comprehensive Metrics
- Accuracy, Precision, Recall, F1-score (macro/micro/weighted)
- Confusion matrix analysis and visualization
- ROC curves and AUC scores per class
- Precision-Recall curves for imbalanced data

#### q4b: Cross-Validation Strategies
- k-Fold cross-validation (k=5,10)
- Stratified k-fold for imbalanced datasets
- Leave-one-out cross-validation
- Time-series aware validation splits

#### q4c: Statistical Significance Testing
- Paired t-tests between model performances
- McNemar's test for classifier comparison
- Confidence interval calculation
- Effect size analysis

### Q5: Advanced Machine Learning Techniques

#### q5a: Transfer Learning
- Fine-tuning pre-trained models (VGG, ResNet, BERT)
- Feature extraction from pre-trained networks
- Domain adaptation techniques
- Progressive neural networks

#### q5b: Unsupervised Learning
- K-means clustering for emotion discovery
- Gaussian Mixture Models (GMM)
- t-SNE and UMAP for visualization
- Autoencoders for feature learning

#### q5c: Semi-Supervised Learning
- Self-training algorithms
- Co-training with multiple views
- Label propagation techniques
- Pseudo-labeling strategies

### Q6: Model Interpretation and Explainability

#### q6a: Feature Importance Analysis
- Permutation importance
- SHAP (SHapley Additive exPlanations) values
- LIME (Local Interpretable Model-agnostic Explanations)
- Partial dependence plots

#### q6b: Model Debugging
- Error analysis by emotion categories
- Confidence calibration analysis
- Bias detection across demographic groups
- Failure case analysis

#### q6c: Visualization Techniques
- Activation maximization for neural networks
- Grad-CAM for CNN visualization
- Attention visualization for sequence models
- Decision boundary plots

### Q7: Deployment and Production Considerations

#### q7a: Model Optimization
- Model quantization for reduced size
- Pruning for computational efficiency
- Knowledge distillation for model compression
- Hardware-aware optimization

#### q7b: API Development
- RESTful API design for model serving
- Batch prediction endpoints
- Real-time streaming inference
- Authentication and rate limiting

#### q7c: Monitoring and Maintenance
- Model performance drift detection
- Data distribution shift monitoring
- A/B testing framework
- Continuous integration for model updates


