AIR QUALITY PREDICTION

This project uses a simple linear regression model to predict PM2.5 levels based on the concentration of Ozone (O3), Nitrogen Dioxide (NO2), and Sulfur Dioxide (SO2).

1. Features

●Data Loading: Loads air quality data from a CSV file using pandas.

●Data Preprocessing: Handles missing values by dropping rows.

●Feature Selection: Selects 'O3', 'NO2', and 'SO2' as features.

●Model Training: Trains a linear regression model using scikit-learn.

●Model Evaluation: Evaluates the model using Mean Squared Error (MSE).

●Visualization: Visualizes actual vs. predicted PM2.5 values using matplotlib.

●Future Prediction: Predicts PM2.5 levels for new data.

2.Dependencies

  ●pandas

  ●scikit-learn

  ●matplotlib
