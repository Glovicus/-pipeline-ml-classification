# -common-pipeline-ml-classification
A general-purpose machine learning pipeline for tabular and text data classification. Features include preprocessing of numerical, categorical, and text features, hyperparameter tuning with GridSearchCV and prediction on new data. Libraries: scikit-learn and pandas.

## Features

- Handles **numerical**, **categorical**, and **text** features.  
- Performs **missing value imputation** and **scaling/encoding**.  
- Uses **TF-IDF vectorization** for text features.  
- Integrates a **classifier** (RandomForest) into a unified **pipeline**.  
- Supports **hyperparameter tuning** with `GridSearchCV`.  
- Outputs predictions for unseen test data in CSV format.

## Requirements

- Python 3.8+  
- pandas  
- scikit-learn  

Install dependencies with:

```bash
pip install pandas scikit-learn
```

## **Usage**

Place your training and test datasets as train.csv and test.csv.

Update the feature lists (numeric_features, categorical_features, text_features) and the target column name in the code.

Run the pipeline:

```bash
python pipeline.py
```
The script will output:

- Validation accuracy in the console.

- Predictions for the test set saved as predictions.csv.


## **Customization**

You can replace the classifier (RandomForestClassifier) with any scikit-learn compatible model.

Add additional text features using FeatureUnion or separate transformers.

Adjust hyperparameter grids in param_grid for GridSearchCV.
