NAIVE BAYES CLASSIFICATION - HEART DATASET

PROJECT OVERVIEW:
This project demonstrates Naive Bayes classification using the Heart dataset.
The model uses patient measurements and recorded test results to predict whether a row is labeled as having heart disease.
The dataset contains 918 rows, 11 input features, and one target column. This is a classification task because the target contains two class labels.

WHAT THE MODEL PREDICTS:
The target is HeartDisease.
    0 = No heart disease
    1 = Heart disease
The model predicts one of these labels for each input row. These predictions are for machine-learning practice and are not medical diagnoses.

INPUT FEATURES:
Age              Age of the patient
Sex              Sex category recorded in the dataset
ChestPainType    Type of chest pain
RestingBP        Resting blood pressure
Cholesterol      Cholesterol measurement
FastingBS        Whether fasting blood sugar is above the dataset threshold
RestingECG       Resting electrocardiogram result
MaxHR            Maximum heart rate
ExerciseAngina   Whether exercise-related angina is recorded
Oldpeak          ST depression measurement
ST_Slope         Slope of the peak exercise ST segment
HeartDisease is the target and is not included among the input features.

WHAT IS NAIVE BAYES?
Naive Bayes is a classification algorithm based on probability. It estimates which class is most likely given the observed input features, then predicts that class.
Bayes' theorem is:
    P(class | features) = P(features | class) * P(class) / P(features)

Where:
    P(class | features)
        The probability of a class after observing the input features.
    P(features | class)
        The probability of observing those features for that class.
    P(class)
        The prior probability of the class before considering the features.
    P(features)
        The overall probability of observing those features.
The word "naive" refers to the simplifying assumption that features are conditionally independent given the class. In practice, features may be related, but Naive Bayes can still be useful and fast.

GAUSSIAN NAIVE BAYES:
This project uses GaussianNB, a Naive Bayes model suited to numerical features.
GaussianNB assumes each numerical feature follows an approximate bell-shaped, or Gaussian, distribution within each class. It estimates the mean and variance of each feature for each class, then uses those estimates to calculate class probabilities.
The model predicts the class with the highest estimated probability.


NOTEBOOK WORKFLOW:
1. Import the libraries.
   pandas loads and displays the dataset.
   scikit-learn provides the data split, encoder, Naive Bayes model, and evaluation metrics.

2. Load the dataset.
   pd.read_csv() loads Heart_dataset into a pandas DataFrame.
   The file path can be changed if the dataset is stored elsewhere.

3. Separate features and target.
   X contains all columns except HeartDisease.
   y contains HeartDisease, the class the model should predict.
   The target must not be included in X because it is the answer being predicted.

4. Identify feature types.
   Text and category columns are sent to OneHotEncoder.
   Numeric columns are passed through unchanged.

5. Split the examples.
   train_test_split divides the data into training and test sets.
   test_size=0.2 assigns 20 percent of the rows to the test set.
   random_state=42 makes the split repeatable.
   stratify=y helps preserve the proportion of each target class in both sets.

6. Prepare the features.
   ColumnTransformer applies one-hot encoding to categorical columns and passes numeric columns through.
   The Pipeline applies the same preparation steps during training and prediction.

7. Train the model.
   GaussianNB() creates the Gaussian Naive Bayes classifier.
   model.fit(X_train, y_train) learns feature distributions and class probabilities from the training data.

8. Make predictions.
   model.predict(X_test) predicts a class label for each held-back test example.

9. Evaluate the predictions.
   The predictions are compared with y_test, the known labels for the test examples.


EVALUATION METRICS:
Accuracy
    The fraction of test examples classified correctly.
    Accuracy alone may hide poor performance on one class.

Confusion Matrix
    Shows how many examples were correctly and incorrectly classified.
    Rows represent actual classes; columns represent predicted classes.

Precision
    Of the examples predicted to belong to a class, the fraction that actually belongs to that class.

Recall
    Of the examples that actually belong to a class, the fraction the model correctly identifies.

F1-score
    A combined measure of precision and recall.

Support
    The number of actual test examples belonging to each class.

Evaluation results depend on the train-test split and dataset. The notebook should be run to obtain the current metric values.

