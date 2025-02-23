# Tumor Classification: Predicting Benign vs Malignant
This project aims to build a predictive model to classify tumors as **benign** or **malignant** based on clinical and diagnostic features. The goal is to provide an accurate and efficient classification system leveraging various machine learning models.
## Features of the Project
- **Data Processing**: Data is prepared and filtered to work seamlessly with classification algorithms. Preprocessing steps include scaling and encoding features.
- **Feature Visualization**: Scatter plots and other visualizations explore relationships between attributes.
- **Model Training**: Various models (e.g., Logistic Regression, Support Vector Machines, Decision Trees) are implemented with hyperparameter tuning.
- **User Interaction**: Offers options to explore and test different scaling methods, feature combinations, and classifiers.

## Key Components
### Data Preparation
- **Loading and Filtering**: Data is processed using functions like `load_data` and `filter_data`.
- **Scaling Methods**: Includes multiple scaling options (`scaler_options`) for better feature standardization.

### Machine Learning Models
- Supported models include:
    - **Logistic Regression:** Optimized with `logreg_param_grid`.
    - **Support Vector Machines (SVM):** Configured using `svc_param_grid`.
    - **Decision Trees/Classifiers:** Tuned using `clf_param_grid`.

- Functionality provided for training and evaluation through `log_reg`, `svm_svc`, and `clf` functions.

### Visualization
- Visual tools like `scatter_plot` to explore feature relationships, class distinctions, and model effectiveness.

## Usage
1. **Load Data**: Use the `load_data` function to import and inspect data.
2. **Scale Features**: Choose scaling options via `scale_data` for model input preparation.
3. **Train Models**: Select models and adjust hyperparameters via `create_model`.
4. **Visualize Results**: Leverage `scatter_plot` for insights into data distribution.

## Requirements
Ensure these prerequisites are installed:
- Python (>= 3.8)
- Key libraries: `numpy`, `pandas`, `scikit-learn`, and `matplotlib`
