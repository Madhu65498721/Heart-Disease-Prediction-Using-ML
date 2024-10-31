# Heart Disease Prediction Using Machine Learning

A predictive model to analyze patient data and predict the likelihood of heart disease using Machine Learning (ML) techniques. This project aligns with **United Nations Sustainable Development Goal (SDG) 3: Good Health and Well-being**, aiming to improve preventive care and patient outcomes by leveraging data science.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
---

## Objective
This project seeks to:
1. Develop an ML model to predict heart disease.
2. Provide actionable insights through personalized health recommendations.
3. Contribute towards SDG 3 by helping to reduce premature mortality from chronic diseases.

## Dataset
The dataset used for this project (`hdp_data.csv`) includes patient information with various health indicators relevant to heart disease. Each row represents a patient’s data and includes features like cholesterol levels, age, blood pressure, and more. The target variable indicates the presence or absence of heart disease.

## Technologies Used
- **Python**: For scripting and model development
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine Learning library for training models
- **Logistic Regression, K-Nearest Neighbors, Support Vector Machine (SVM)**: ML models used for prediction

## Project Structure
```
.
├── hdp_data.csv               # Dataset
├── README.md                  # Project documentation
├── heart_disease_prediction.py # Main script for training and evaluating models
└── requirements.txt           # Dependencies
```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Madhu65498721/Heart-Disease-Prediction-Using-ML.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the Main Script**:
   Run `heart_disease_prediction.py` to preprocess data, train models, and view results. The script will load the dataset, perform data preprocessing, and train three models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)

   ```bash
   python heart_disease_prediction.py
   ```

2. **Evaluate Model Performance**:
   After running the script, you’ll see:
   - Classification reports for each model
   - Confusion matrices and ROC curves
   - Comparison of model accuracies through a bar chart

## Model Performance
- **Logistic Regression**: Accuracy of **85%**
- **K-Nearest Neighbors (KNN)**: Accuracy of **92%**
- **Support Vector Machine (SVM)**: Accuracy of **82%**

The **K-Nearest Neighbors model** demonstrated the highest accuracy for this dataset, making it the preferred model for this heart disease prediction task.

### Sample Visualization
Below are examples of visualizations included in the project:
- **Confusion Matrix**: Display of actual vs. predicted cases
- **ROC Curve**: Evaluation of model performance across thresholds
- **Accuracy Comparison Chart**: A bar chart comparing the accuracy of all models

## Future Enhancements
- **Advanced Models**: Implement additional models like Random Forest or Gradient Boosting.
- **Explainability**: Use SHAP or LIME for feature importance analysis to better interpret individual predictions.
- **Expanded Dataset**: Incorporate more features, including genetic predispositions and lifestyle factors, to improve the model's generalizability.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions.
