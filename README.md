FINANCIAL EXPENSE CATEGORIZATION WITH LOGISTIC REGRESSION

This project categorizes financial expenses using logistic regression.

1.Features
    ●Data Loading and Preprocessing:
    ●Loads financial data from a CSV file using pandas.
    ●Removes the 'Date' column.
    ●One-hot encodes the 'Description' column.
    ●Splits the data into training and testing sets.
    ●Scales the numerical features using StandardScaler.

●Model Training:
    ●Trains a logistic regression model (using the 'liblinear' solver and 'auto' multi_class setting) with scikit-learn.
    ●Model Evaluation:
    ●Evaluates model performance using accuracy score and a classification report.
    ●Performs 5-fold cross-validation to assess model generalization.

●Visualization
    ●Visualizes the distribution of expense categories using a count plot

2.Dependencies
    ●pandas
    ●scikit-learn
    ●matplotlib
    ●seaborn
