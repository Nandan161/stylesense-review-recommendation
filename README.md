# README Template

# Data Science Pipeline for Product Recommendation
## Project Description
### This project builds a machine learning pipeline to predict whether a customer recommends a product based on their review. The pipeline processes numerical, categorical, and text data, performs feature engineering, trains multiple machine learning models (Random Forest and Logistic Regression), and evaluates their performance using cross-validation.

## Getting Started
Instructions for setting up the project locally.

## Dependencies
The following dependencies are required to run this project:

nginx
Copy
Edit
pandas
scikit-learn
numpy
matplotlib
seaborn
nltk
Installation
Follow the steps below to set up a development environment:

## Clone the repository:

git clone https://github.com/Nandan161/stylesense-review-recommendation.git
cd stylesense-review-recommendation


## Install the required dependencies:
pip install -r requirements.txt

## Start Jupyter Notebook:

Open and run the notebook starter.ipynb.

## Testing
The project includes cross-validation and model evaluation using accuracy scores for different models.

## Break Down Tests
Cross-Validation Scores: Each model is evaluated using 5-fold cross-validation. The models' performance is compared to determine the best-performing model.

## Model Evaluation: After training, the model is evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score.

# Example to evaluate the Random Forest model
rf_score = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f'Random Forest CV Accuracy: {rf_score:.4f}')
Project Instructions
Preprocessing Steps: Clean and process the data (handle numerical, categorical, and text features).

## Feature Engineering: Create new features from the text (e.g., review sentiment, word count).

## Model Training: Train both a Random Forest and Logistic Regression model.

## Fine-tuning: Use cross-validation to find the best hyperparameters for the models.

## Model Evaluation: Evaluate the models on the test set.

## Built With
pandas - Data manipulation library used to load and preprocess the dataset.

scikit-learn - Used for building machine learning models and evaluation.

nltk - Natural Language Processing library for text processing and feature engineering.

matplotlib - Used for data visualization.

## License
Udacity

Author: Nandan Choudhary

