DECISION TREE REGRESSION - TEEN PHONE ADDICTION DATASET

PROJECT OVERVIEW:
This project demonstrates decision tree regression using the Teen Phone Addiction dataset.
The model uses phone-use, wellbeing, and demographic features to estimate the numerical Addiction_Level score.
The workflow loads and inspects the data, separates features from the target, splits the examples into training and test sets, handles missing values, encodes categorical columns, trains a regression tree, and evaluates its predictions.

WHAT THE MODEL PREDICTS:
The target is Addiction_Level, a numerical score.
This is a regression task because the model predicts a number rather than a category. Its predictions are estimates and may differ from the actual scores.

INPUT FEATURES:
The model uses the dataset's input columns except for the target and identifying columns.
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

The model excludes Addiction_Level from the inputs because it is the target.
It also excludes ID, Name, and Location because they identify records or places.

WHAT IS A DECISION TREE REGRESSION MODEL?
A decision tree regressor predicts a number by dividing the data into smaller groups using feature-based questions.
For example, it may split records according to phone usage, then split a group again according to sleep hours. Each final group is called a leaf. The model predicts a numerical value for each leaf, generally based on the average target value of its training examples.
The regression tree selects splits that reduce the variation in target values within the resulting groups.

MATHEMATICAL INTUITION:
The model makes a prediction for each training example and measures its error:
    error = actual Addiction_Level - predicted Addiction_Level
With criterion="squared_error", a split is evaluated by how much it reduces squared errors within the resulting groups. Large errors count more because the errors are squared.
The model's prediction for a leaf is generally the average Addiction_Level of the training examples that reach that leaf.

TREE SETTINGS:
The model uses:
    criterion="squared_error"
        Chooses splits that reduce squared error in the target values.
    max_depth=4
        Limits the tree to four levels.
    min_samples_leaf=5
        Requires at least five training examples in each leaf.
    random_state=42
        Makes results repeatable.
Limiting tree depth and requiring several examples per leaf can help reduce overfitting. These values are starting settings; different settings may give different results.

NOTEBOOK WORKFLOW:
1. Import libraries.
   pandas loads and displays the data.
   scikit-learn provides the data split, regression tree, and evaluation metrics.
2. Load the dataset.
   pd.read_csv() loads teen_phone_addiction_dataset.csv into a pandas DataFrame.
3. Separate features and target.
   X contains the input features.
   y contains Addiction_Level, the score to predict.
   The target must not be included in X because it is the answer being predicted.
4. Exclude identifying columns.
   ID, Name, and Location are removed from the inputs because they identify records or places rather than describe the behavior being modeled.
5. Split the examples.
   train_test_split divides the data into training and test sets.
   test_size=0.2 assigns 20 percent of the rows to the test set.
   random_state=42 makes the split repeatable.
6. Handle missing values.
   Numeric columns are filled using the median from the training data.
   Categorical columns are filled using the most frequent training value.
   The same fill values are applied to the test data.
7. Encode categorical features.
   pd.get_dummies() converts text categories into numeric indicator columns.
   The test data is aligned to the training columns so both sets contain the same input features.
8. Train the regression tree.
   DecisionTreeRegressor() creates the model.
   model.fit(X_train, y_train) learns the tree's splits and leaf predictions from the training examples.
9. Predict and evaluate.
   model.predict(X_test) predicts Addiction_Level for held-back examples.
   The notebook calculates MAE, RMSE, and R-squared.
10. Compare actual and predicted scores.
    A table displays actual and predicted values for the first ten test examples.

EVALUATION METRICS:
MAE — Mean Absolute Error
    The average absolute difference between actual and predicted Addiction_Level scores.
    It is expressed in score units.

RMSE — Root Mean Squared Error
    The square root of the average squared prediction error.
    It is expressed in score units and gives larger errors more weight.

R-squared — R2
    Compares the model with a baseline that always predicts the average target.
    A score of 1 means a perfect fit.
    A score of 0 means about as good as the average-value baseline.
    A negative score means worse than that baseline on the test set.
    R2 is not the percentage of predictions that are correct.

The metric values depend on the train-test split and model settings. Run the notebook to obtain the results for the current data and code.

