TELECOM CUSTOMER CHURN PREDICTION WITH LOGISTIC REGRESSION

This project analyzes telecommunications customer data to predict churn (customer attrition) using logistic regression.

1.Features
   ● Data Loading and Preprocessing:
   ●Loads data from a CSV file using pandas.
   ●Selects 'Partner' and 'Dependents' as features.
   ●Encodes categorical features ('Partner', 'Dependents') using LabelEncoder.

●Model Training:

   ●Splits the data into training and testing sets.
   ●Scales the features using StandardScaler.
   ●Trains a logistic regression model using scikit-learn.

●Model Evaluation:
   ●Evaluates model performance using a confusion matrix.

●Visualization:
   ●Visualizes the decision boundary of the logistic regression model on both the training and test sets.

2.Dependencies
  ●numpy
  ●pandas
  ●matplotlib
  ●scikit-learn
