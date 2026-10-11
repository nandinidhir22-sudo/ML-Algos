SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION - HEART DATASET

PROJECT OVERVIEW:
This project demonstrates Support Vector Machine classification using the Heart dataset.
The model uses patient measurements and test results to predict whether the dataset labels a record as having heart disease.
The dataset contains 918 rows, 11 input features, and one target column.
This is a classification task because the target contains categories rather than a continuous numerical measurement.

WHAT THE MODEL PREDICTS:
The target is HeartDisease.
    0 = No heart disease
    1 = Heart disease
The model predicts one of these two class labels for each input record. These predictions are for learning purposes and are not medical diagnoses.

INPUT FEATURES:
Age              Age of the patient
Sex              Sex category recorded in the dataset
ChestPainType    Type of chest pain
RestingBP        Resting blood pressure
Cholesterol      Cholesterol measurement
FastingBS        Whether fasting blood sugar is above the dataset's threshold
RestingECG       Resting electrocardiogram result
MaxHR            Maximum heart rate
ExerciseAngina   Whether exercise-related angina is recorded
Oldpeak          ST depression measurement
ST_Slope         Slope of the peak exercise ST segment

WHAT IS SUPPORT VECTOR MACHINE CLASSIFICATION?
A Support Vector Machine (SVM) is a supervised learning algorithm used to classify examples into categories.
The SVM looks for a decision boundary that separates the classes. It tries to create the widest possible margin between that boundary and the closest training examples from each class.
The closest examples are called support vectors. They help determine the position of the boundary.
For a linear SVM, a simplified decision function is:
    decision value = b0+ b1*x1 + b2*x2 + ... + bn*xn 

The model assigns a class based on which side of the decision boundary the example falls on.
This notebook uses an RBF kernel:
    SVC(kernel="rbf")
The RBF kernel allows the SVM to form a curved boundary when a straight boundary is not sufficient. It does this by calculating similarities between examples rather than explicitly creating a curved plot in the original feature space.

MATHEMATICAL INTUITION: MARGIN AND SUPPORT VECTORS
An SVM tries to separate classes while keeping the decision boundary as far as possible from the nearest examples.
The margin is the distance between the decision boundary and its closest training examples.
A wider margin often helps the model generalize to new examples. The closest examples, called support vectors, have the strongest influence on where the boundary is placed.
The parameter C controls the balance between a wide margin and errors on the training data:
    Smaller C:
        Allows more training examples to be on the wrong side of the margin.
        Often favors a wider, simpler boundary.
    Larger C:
        Penalizes training errors more strongly.
        May create a tighter boundary around the training examples.
This notebook uses the default C value of 1.

NOTEBOOK WORKFLOW:
1. Import the libraries.
   pandas loads and displays the dataset.
   scikit-learn provides the data split, preprocessing tools, SVM classifier, and evaluation metrics.

2. Load the dataset.
   pd.read_csv() loads Heart_dataset into a pandas DataFrame.
   The notebook previews the first rows and checks the table's shape.

3. Separate features and target.
   X contains all columns except HeartDisease.
   y contains HeartDisease, the class the model should predict.
   The target must not be included in X because it is the answer being predicted.

4. Identify feature types.
   Numeric columns are passed to StandardScaler.
   Text columns are passed to OneHotEncoder.

5. Split the examples.
   train_test_split separates the records into training and test sets.
   test_size=0.2 assigns 20 percent of the records to the test set.
   random_state=42 makes the split repeatable.
   stratify=y helps preserve the class proportions in both sets.

6. Create the preprocessing steps.
   ColumnTransformer applies scaling to numeric columns and one-hot encoding to categorical columns.
   The transformations are fitted using training data when the pipeline is trained.

7. Create and train the SVM.
   SVC(kernel="rbf") creates an SVM classifier using the RBF kernel.
   Pipeline combines preprocessing and classification into one sequence.
   model.fit(X_train, y_train) learns the preprocessing values and the SVM boundary from the training data.

8. Make predictions.
   model.predict(X_test) preprocesses the test features and predicts a class for each test record.

9. Evaluate the predictions.
   The predictions are compared with y_test, the known target values for the held-back test records.

IMPORTANT FUNCTIONS AND OBJECTS:
pd.read_csv(file_path)
    Loads the dataset from a CSV file into a pandas DataFrame.

train_test_split(X, y, ...)
    Splits the input features and matching target values into training and test sets.

StandardScaler()
    Scales numeric features using statistics learned from the training data.

OneHotEncoder(handle_unknown="ignore")
    Converts text categories into numeric indicator features.

ColumnTransformer(...)
    Applies different preprocessing steps to different feature columns.

SVC(kernel="rbf")
    Creates an SVM classifier using the RBF kernel.

Pipeline(...)
    Connects preprocessing and classification so they are applied in the correct order.

model.fit(X_train, y_train)
    Learns preprocessing values and the SVM classification boundary from the training data.

model.predict(X_test)
    Predicts a class label for each test example.

accuracy_score(y_test, predictions)
    Calculates the fraction of test predictions that match the actual labels.

confusion_matrix(y_test, predictions)
    Counts correct and incorrect predictions for each actual and predicted class.

classification_report(y_test, predictions)
    Reports precision, recall, F1-score, and support for each class.


EVALUATION METRICS:
Accuracy
    The fraction of test examples classified correctly.
    Accuracy alone may hide poor performance on one class.

Confusion Matrix
    Shows how many records were correctly and incorrectly classified.
    Rows represent actual classes; columns represent predicted classes.

Precision
    Of the records predicted to belong to a class, the fraction that actually belongs to that class.

Recall
    Of the records that actually belong to a class, the fraction the model correctly identifies.

F1-score
    A combined measure of precision and recall. It is useful when both types of errors matter.

Support
    The number of actual test examples belonging to each class.

The test results depend on the train-test split and model settings. A good accuracy score does not guarantee that each class has equally good precision or recall.