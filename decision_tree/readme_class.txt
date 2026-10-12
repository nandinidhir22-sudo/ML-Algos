DECISION TREE CLASSIFICATION - HEART DATASET

PROJECT OVERVIEW:
This project demonstrates decision tree classification using the Heart dataset.
The model uses patient measurements and recorded test results to predict whether a row is labeled as having heart disease.
The notebook loads the data, prepares numeric and categorical features, splits the data into training and test sets, trains a decision tree, evaluates predictions, and displays the trained tree.
This is a classification task because the target has two class labels.

WHAT THE MODEL PREDICTS:
The target is HeartDisease.
    0 = No heart disease
    1 = Heart disease
The model predicts one of these labels for each record. These predictions are for machine-learning practice and are not medical diagnoses.

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
HeartDisease is the target and is not included among the model inputs.

WHAT IS A DECISION TREE?
A decision tree is a supervised learning model that makes predictions by asking a sequence of questions about the input features.
At each node, the tree chooses a feature and split that separates the classes. Each branch represents the result of a split, and each leaf gives the predicted class.
For example, a tree might first split on a measurement such as maximum heart rate, then use another feature to further separate records. The actual questions depend on the training data.


GINI IMPURITY, ENTROPY, AND INFORMATION GAIN:
The model's criterion determines how it measures the quality of a possible split.

Gini impurity:
    Measures how mixed the classes are in a node.
    The tree chooses splits that reduce Gini impurity.
    Gini = 1 - sum(p_class^2)

Entropy:
    Measures uncertainty or class mixing in a node.
    Entropy = -sum(p_class * log2(p_class))

Information gain:
    Measures how much a split reduces entropy.
    Information gain = parent entropy - weighted child entropy
The code uses:
    criterion="gini"
Change it to:
    criterion="entropy"
to use entropy-based splitting, which selects splits based on information gain.

PRE-PRUNING SETTINGS:
The model uses two settings to restrict tree growth:
    max_depth=4
        Limits the tree to four levels.
    min_samples_leaf=5
        Requires at least five training records in each leaf.
These limits help prevent an overly complex tree from memorizing the training examples. This is a form of pre-pruning because it limits growth while the tree is being built.

NOTEBOOK WORKFLOW:
1. Import libraries.
   pandas loads and displays the data.
   scikit-learn provides the data split, classifier, metrics, and tree visualization.
   matplotlib displays the tree.
2. Load the dataset.
   pd.read_csv() reads Heart_dataset into a pandas DataFrame.
3. Separate features and target.
   X contains all columns except HeartDisease.
   y contains HeartDisease, the value the model should predict.
4. Split the data.
   train_test_split separates the data into training and test sets.
   test_size=0.2 reserves 20 percent of the rows for testing.
   random_state=42 makes the split repeatable.
   stratify=y helps preserve the class proportions in both sets.
5. Handle missing values.
   Numeric columns are filled using the median calculated from the training data.
   Categorical columns are filled using the most frequent training value.
   The same fill values are applied to the test data.
6. Encode categorical features.
   pd.get_dummies() converts text categories into numeric indicator columns.
   The test columns are aligned with the training columns so the model receives the same features.
7. Train the decision tree.
   DecisionTreeClassifier() creates the classification model.
   model.fit(X_train, y_train) learns the tree's split rules from the training data.
8. Predict and evaluate.
   model.predict(X_test) predicts a class for each test record.
   Accuracy, confusion matrix, and classification report summarize the results.
9. Visualize the tree.
   plot_tree() displays the model's learned splits, branches, and leaf predictions.


EVALUATION METRICS:
Accuracy
    The fraction of test examples classified correctly.

Confusion Matrix
    Shows correct and incorrect predictions for each class.
    The lower-left cell counts actual heart-disease cases predicted as no heart disease.

Precision
    Of the records predicted to belong to a class, the fraction that actually belongs to that class.

Recall
    Of the records that actually belong to a class, the fraction correctly identified by the model.

F1-score
    A combined measure of precision and recall.

Support
    The number of actual test examples in each class.

Accuracy alone may hide poor performance on one class, so review the confusion matrix and per-class report as well.

