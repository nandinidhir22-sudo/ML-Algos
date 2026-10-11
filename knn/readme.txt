K-NEAREST NEIGHBORS (KNN) - CLASSIFICATION AND REGRESSION

PROJECT OVERVIEW:
This project demonstrates two uses of the K-Nearest Neighbors algorithm:
1. KNN classification on the Heart dataset.
   It predicts whether a record is labeled as having heart disease.
2. KNN regression on the Teen Phone Addiction dataset.
   It predicts the numerical Addiction_Level score.
Both notebooks scale numerical features, encode categorical features, split data into training and test sets, and use five neighbors (k=5).

PROJECT FILES:
knn_class.ipynb
    Notebook for KNN classification using the Heart dataset.
knn_reg.ipynb
    Notebook for KNN regression using the Teen Phone Addiction dataset.
Heart_dataset
    Dataset file used by knn_class.ipynb.
teen_phone_addiction_dataset.csv
    Dataset file used by knn_reg.ipynb.


WHAT IS K-NEAREST NEIGHBORS?
K-Nearest Neighbors (KNN) makes predictions by finding the k training examples most similar to a new example.
Similarity is calculated using distance between feature values. This project uses k=5, so each prediction is based on the five closest training examples.
For classification:
    The model predicts the most common class among the nearest neighbors.
For regression:
    The model predicts the average target value among the nearest neighbors.
The KNN algorithm stores the training examples and uses them when making predictions rather than learning a short equation during training.




KNN CLASSIFICATION - HEART DATASET

WHAT THE MODEL PREDICTS:
The target is HeartDisease.

    0 = No heart disease
    1 = Heart disease

The dataset contains 918 rows. Each row represents one record with patient measurements and test results.

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

CLASSIFICATION WORKFLOW:
1. Load the dataset with pandas.
   pd.read_csv("Heart_dataset") reads the file into a DataFrame.

2. Separate features and target.
   X contains all columns except HeartDisease.
   y contains HeartDisease, the class the model should predict.

3. Split the examples.
   train_test_split divides the rows into training and test sets.
   test_size=0.2 assigns 20 percent of the rows to the test set.
   random_state=42 makes the split repeatable.
   stratify=y helps keep the class proportions similar in both sets.

4. Prepare the features.
   StandardScaler scales the numeric columns.
   OneHotEncoder converts text categories into numeric indicator columns.
   ColumnTransformer applies the correct transformation to each column type.

5. Train the classifier.
   KNeighborsClassifier(n_neighbors=5) creates a KNN classifier using five neighbors.
   knn_classifier.fit(X_train, y_train) prepares the pipeline using the training examples.

6. Predict and evaluate.
   knn_classifier.predict(X_test) predicts a class for each test record.
   The notebook reports accuracy, a confusion matrix, and a classification report.
   It also plots the confusion matrix.


CLASSIFICATION EVALUATION:
Accuracy
    The fraction of test examples classified correctly.

Confusion Matrix
    Counts correct and incorrect predictions for each actual and predicted class.
    It helps show false positives and false negatives.

Classification Report
    Reports precision, recall, F1-score, and support for each class.

Confusion Matrix Plot
    Displays the confusion-matrix counts as a color-coded chart.


KNN REGRESSION - TEEN PHONE ADDICTION DATASET

WHAT THE MODEL PREDICTS:
The target is Addiction_Level, a numerical score.

The dataset contains 3,000 rows and 25 columns. Each row represents one record in the dataset.

INPUT FEATURES:
Age                       Age of the teen
Gender                    Gender category recorded in the dataset
School_Grade              School grade
Daily_Usage_Hours         Daily phone usage time
Sleep_Hours               Sleep duration
Academic_Performance      Academic performance measurement
Social_Interactions       Social interaction measurement
Exercise_Hours            Exercise time
Anxiety_Level             Anxiety score
Depression_Level          Depression score
Self_Esteem               Self-esteem score
Parental_Control          Parental control indicator
Screen_Time_Before_Bed    Phone screen time before bed
Phone_Checks_Per_Day      Number of daily phone checks
Apps_Used_Daily           Number of apps used daily
Time_on_Social_Media      Time spent on social media
Time_on_Gaming            Time spent gaming
Time_on_Education         Time spent using educational apps
Phone_Usage_Purpose       Main phone usage purpose
Family_Communication      Family communication measurement
Weekend_Usage_Hours       Phone usage on weekends

The notebook excludes ID, Name, and Location from the input features because they identify records or places. Addiction_Level is the target and is also excluded from the inputs.

REGRESSION WORKFLOW:
1. Load the dataset with pandas.
   pd.read_csv("teen_phone_addiction_dataset.csv") reads the file into a DataFrame.

2. Separate features and target.
   X contains the selected input features.
   y contains Addiction_Level, the score to predict.

3. Split the examples.
   train_test_split divides the rows into training and test sets.
   test_size=0.2 assigns 20 percent of the rows to the test set.
   random_state=42 makes the split repeatable.

4. Prepare the features.
   StandardScaler scales numeric columns.
   OneHotEncoder converts categorical columns into numeric indicator columns.
   ColumnTransformer applies the transformations to the appropriate columns.

5. Train the regressor.
   KNeighborsRegressor(n_neighbors=5) creates a KNN regressor using five neighbors.
   knn_regressor.fit(X_train, y_train) prepares the pipeline using the training examples.

6. Predict and evaluate.
   knn_regressor.predict(X_test) predicts Addiction_Level for each test record.
   The notebook reports MAE, RMSE, and R-squared.

7. Compare predictions.
   The notebook displays the first ten actual and predicted scores in a table.
   It also plots those values as grouped bars.


REGRESSION EVALUATION:
MAE — Mean Absolute Error
    The average absolute difference between actual and predicted scores.
    It is expressed in Addiction_Level score units.

RMSE — Root Mean Squared Error
    The square root of the average squared prediction error.
    It is expressed in score units and gives larger errors more weight.

R-squared — R2
    Compares model predictions with a baseline that always predicts the average target.
    A score of 1 means a perfect fit.
    A score of 0 means about as good as the average-value baseline.
    A negative score means worse than that baseline on the evaluated data.
    R2 is not the percentage of predictions that are correct.

Actual-versus-predicted table and bar graph
    The notebook displays and plots the first ten test examples.
    Similar actual and predicted values indicate closer predictions for those examples.
    The graph shows only ten rows; the evaluation metrics summarize the full test set.
