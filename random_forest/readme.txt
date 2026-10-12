RANDOM FOREST - HEART CLASSIFICATION AND PHONE ADDICTION REGRESSION

PROJECT OVERVIEW:
This project demonstrates Random Forest for two different machine-learning tasks:
1. Classification on the Heart dataset.
   The model predicts whether a record is labeled as having heart disease.
2. Regression on the Teen Phone Addiction dataset.
   The model predicts the numerical Addiction_Level score.
A Random Forest combines many decision trees. For classification, the trees vote for a class. For regression, their numerical predictions are averaged.

PROJECT FILES:
The project uses these datasets:
    Heart_dataset
        Heart dataset used for classification.
    teen_phone_addiction_dataset.csv
        Teen phone addiction dataset used for regression.
The notebook code reads each dataset using its file path. Update the paths if the files are stored elsewhere.

WHAT IS A RANDOM FOREST?
A Random Forest is an ensemble model made from many decision trees.
Each tree is trained using a sample of the training data and considers randomly selected features when making splits. The trees make predictions independently, then the forest combines them:
    Classification:
        The final class is selected by majority vote.
    Regression:
        The final numerical prediction is the average of the tree predictions.
The code uses 300 trees through n_estimators=300.

NOTEBOOK 1: RANDOM FOREST CLASSIFICATION - HEART DATASET

WHAT THE MODEL PREDICTS:
The target is HeartDisease.

    0 = No heart disease
    1 = Heart disease

The model uses the patient measurements and recorded test results as inputs, then predicts one of these two class labels.

INPUT FEATURES:
The model uses all columns except HeartDisease.

The dataset features include:
Age              Age of the patient
Sex              Sex category recorded in the dataset
ChestPainType    Type of chest pain
RestingBP        Resting blood pressure
Cholesterol      Cholesterol measurement
FastingBS        Fasting blood sugar indicator
RestingECG       Resting electrocardiogram result
MaxHR            Maximum heart rate
ExerciseAngina   Whether exercise-related angina is recorded
Oldpeak          ST depression measurement
ST_Slope         Slope of the peak exercise ST segment

CLASSIFICATION WORKFLOW:
1. Load the data.
   pandas read_csv() loads Heart_dataset into a DataFrame.
2. Separate features and target.
   X contains all columns except HeartDisease.
   y contains HeartDisease, the class to predict.
3. Split the data.
   train_test_split separates the data into training and test sets.
   test_size=0.2 reserves 20 percent of the rows for testing.
   random_state=42 makes the split repeatable.
   stratify=y helps preserve the class proportions in both sets.
4. Handle missing values.
   Numeric columns are filled with the training-set median.
   Categorical columns are filled with the most frequent training value.
   These same fill values are applied to the test set.
5. Encode categories.
   pd.get_dummies() converts text categories into numeric indicator columns.
  Test columns are aligned with the training columns before prediction.
6. Train the classifier.
   RandomForestClassifier(n_estimators=300) creates the model with 300 decision trees.
   model.fit(X_train, y_train) trains the forest.
7. Predict and evaluate.
   model.predict(X_test) predicts heart-disease labels.
   The notebook reports accuracy, confusion matrix, and classification report.
8. Visualize results.
   A confusion matrix plot shows correct and incorrect predictions for each class.
   A feature-importance plot can show which input features contributed most to tree splits.

CLASSIFICATION METRICS:
Accuracy
    Fraction of test examples classified correctly.
Confusion Matrix
    Counts correct and incorrect predictions by actual and predicted class.
    The lower-left cell represents actual heart-disease cases predicted as no heart disease.
Precision
    Of the records predicted to belong to a class, the fraction that actually belongs to that class.
Recall
    Of the records that actually belong to a class, the fraction the model correctly identifies.
F1-score
    A combined measure of precision and recall.
Support
    Number of actual test examples belonging to each class.

NOTEBOOK 2: RANDOM FOREST REGRESSION - TEEN PHONE ADDICTION DATASET

WHAT THE MODEL PREDICTS:
The target is Addiction_Level, a numerical score.
The model predicts a score rather than a class label.
INPUT FEATURES:
The model uses the dataset's feature columns except Addiction_Level and these identifying columns:
    ID
    Name
    Location
Other included features describe age, phone usage, sleep, wellbeing, school grade, and communication.

REGRESSION WORKFLOW:
1. Load the data.
   pandas read_csv() loads teen_phone_addiction_dataset.csv into a DataFrame.
2. Separate features and target.
   y contains Addiction_Level, the numerical score to predict.
   X contains the selected input features.
   ID, Name, and Location are excluded because they identify records or places.
3. Split the data.
   train_test_split separates the data into training and test sets.
   test_size=0.2 reserves 20 percent of the rows for testing.
   random_state=42 makes the split repeatable.
4. Handle missing values.
   Numeric columns are filled with the training-set median.
   Categorical columns are filled with the most frequent training value.
   These same fill values are applied to the test set.
5. Encode categories.
   pd.get_dummies() converts categorical text columns into numeric indicator columns.
   Test columns are aligned with the training columns.
6. Train the regressor.
   RandomForestRegressor(n_estimators=300) creates the model with 300 decision trees.
   min_samples_leaf=2 requires at least two training examples in each leaf.
   model.fit(X_train, y_train) trains the forest.
7. Predict and evaluate.
   model.predict(X_test) predicts Addiction_Level scores.
   The notebook reports MAE, RMSE, and R-squared.
8. Visualize results.
   An actual-versus-predicted scatter plot compares test scores with predictions.
   The red diagonal line represents perfect predictions; points closer to it are more accurate.
  A feature-importance plot can show which inputs contributed most to tree splits.


REGRESSION METRICS:
MAE — Mean Absolute Error
    Average absolute difference between actual and predicted Addiction_Level scores.
    It is measured in score units.
RMSE — Root Mean Squared Error
    Square root of the average squared prediction error.
    It is also in score units and gives larger errors more weight.
R-squared — R2
    Compares predictions with a baseline that always predicts the average target.
    A score of 1 indicates a perfect fit.
    A score of 0 indicates performance similar to the average-value baseline.
    A negative score indicates worse performance than that baseline on the test set.
    R2 is not the percentage of predictions that are correct.

FEATURE IMPORTANCE:
Random Forest provides feature_importances_ values that summarize how much each feature contributed to the trees' splits.
A feature-importance bar chart can display the most influential features in the fitted model. These values describe model behavior; they do not show that a feature causes the target to change.
