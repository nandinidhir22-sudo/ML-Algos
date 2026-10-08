MULTIPLE LINEAR REGRESSION -CALIFORNIA HOUSING

PROJECT OVERVIEW:
This project demonstrates multiple linear regression using the California Housing dataset from scikit-learn.
The notebook loads and inspects the data, separates the input features from the value to predict, splits the data into training and test sets, and fits a regression model.
Each row represents a California area, not an individual house.

WHAT THE MODEL PREDICTS:
The target is MedHouseVal: the median house value for an area.
The target is measured in hundreds of thousands of dollars.

The dataset contains 20,640 rows, eight input features, and one target column.


INPUT FEATURES:
MedInc       Median household income in the area
HouseAge     Median age of houses in the area
AveRooms     Average number of rooms per household
AveBedrms    Average number of bedrooms per household
Population   Population of the area
AveOccup     Average number of people per household
Latitude     Latitude of the area
Longitude    Longitude of the area

WHAT IS MULTIPLE LINEAR REGRESSION?
Regression is used when the value we want to estimate is a number.
Multiple linear regression uses several input features to estimate that number. It multiplies each feature by a learned coefficient, then adds the results together.

The model's equation is:
predicted MedHouseVal = b0 + b1 * MedInc + b2 * HouseAge + b3 * AveRooms + b4 * AveBedrms + b5 * Population + b6 * AveOccup + b7 * Latitude + b8 * Longitude

b0 is the intercept. It is the model's baseline prediction when all feature values are zero.

b1 through b8 are coefficients learned from the training data. Each coefficient determines how much its feature contributes to the prediction.


MATHEMATICAL INTUITION: HOW THE MODEL LEARNS
For each training example, the model makes a prediction and compares it with the known target.
    residual = actual value - predicted value

A positive residual means the model predicted too low.
A negative residual means the model predicted too high.

Ordinary least squares chooses coefficients that minimize the sum of squared residuals:
    SSE = sum((actual value - predicted value)^2)


(# Squaring the residuals has two useful effects:
1. Positive and negative errors cannot cancel each other out.
2. Large errors count more heavily than small errors.

The model learns one set of coefficients that gives a good overall fit to the training data. It does not memorize a separate equation for every area.)


UNDERSTANDING THE COEFFICIENTS
A coefficient describes how the model's prediction changes when its feature increases by one unit, while the other included features stay fixed.
For example, the MedInc coefficient describes the change in the model's predicted target for a one-unit increase in MedInc, with the other features held constant.
The features use different units, so coefficient sizes should not be compared without considering those units.
A coefficient describes a pattern in the data. It does not, by itself, prove that changing a feature causes house values to change.


NOTEBOOK WORKFLOW
1. Import the libraries.
   pandas is used to work with tables.
   scikit-learn provides the dataset loader, data split, and regression model.

2. Load the dataset.
   fetch_california_housing(as_frame=True) loads the data in a pandas-friendly format.
   The first run may download the dataset and may require internet access.

3. Inspect the data.
   housing.frame contains the input features and target.
   head(10) previews the first ten rows.
   shape reports the number of rows and columns.
   info() reports column names, data types, and non-missing values.

4. Export and read a CSV.
   The notebook writes the dataset to california_housing.csv and reads it into df for inspection.
   The model's feature table and target are taken from housing.frame.

5. Separate the features and target.
   X contains the eight input feature columns.
   y contains MedHouseVal, the value to predict.
   The target must not be included in X, because it is the answer the model is meant to predict.

6. Split the examples.
   train_test_split holds back some rows for evaluation.
   test_size=0.2 assigns 20 percent of the rows to the test set.
   random_state=42 makes the random split repeatable.

7. Fit the model.
   LinearRegression() creates the model.
   model.fit(X_train, y_train) learns the intercept and coefficients from the training examples.

8. Evaluation.
MSE — Mean Squared Error
    The average squared difference between actual and predicted values.
    Larger errors count more because the differences are squared.

RMSE — Root Mean Squared Error
    The square root of MSE.
    It is expressed in the target's units. Multiply RMSE by 100,000 to express it approximately in dollars.

R-squared — R2
    Compares the model with a baseline that always predicts the average target.
    A score of 1 means a perfect fit.
    A score of 0 means about as good as that baseline.
    A negative score means worse than the baseline on the evaluated data.
    R2 is not the percentage of predictions that are correct.