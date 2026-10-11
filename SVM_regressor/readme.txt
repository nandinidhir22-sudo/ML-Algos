SUPPORT VECTOR REGRESSION (SVR) - TEEN PHONE ADDICTION DATASET

PROJECT OVERVIEW:
This project demonstrates Support Vector Regression (SVR) using the Teen Phone Addiction dataset.
The model uses phone usage, wellbeing, and demographic features to predict the numerical Addiction_Level score.
The dataset contains 3,000 rows and 25 columns. ID, Name, and Location are excluded from the model inputs because they identify records or places rather than provide useful general measurements for this example.

WHAT THE MODEL PREDICTS:
The target is Addiction_Level, a numerical score.
This is a regression task because the model predicts a number rather than choosing a category. The predicted score is an estimate and may not exactly match the actual score.

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
ID, Name, and Location are removed from the input features. Addiction_Level is the target and is also removed from the inputs.

WHAT IS SUPPORT VECTOR REGRESSION?
Support Vector Regression (SVR) is the regression version of Support Vector Machines. It predicts a numerical value rather than assigning a class label.
The model tries to fit a function that stays close to the training target values. It allows a small tolerance around the function, called the epsilon tube. Errors inside this tube are treated as acceptable; errors outside it are penalized.
A simplified prediction function is:
    predicted Addiction_Level = f(input features)
For a linear SVR, f is a weighted sum of the features. This project uses the RBF kernel, which lets the model learn curved relationships between the input features and Addiction_Level.

MATHEMATICAL INTUITION:
For each example, the prediction error is:
    error = actual Addiction_Level - predicted Addiction_Level
SVR aims to keep prediction errors within an epsilon-sized tolerance where possible. Errors outside that tolerance affect the model's objective.

The main settings include:
    epsilon:
        Controls the width of the tolerance tube around the fitted function.
    C:
        Controls how strongly errors outside the tolerance tube are penalized.
        Larger C tries harder to fit the training data.
        Smaller C allows more errors in exchange for a smoother fit.
    kernel:
        Determines how the model represents the relationship between inputs and target.
        The RBF kernel can model curved patterns.
The code uses scikit-learn's default C and epsilon values unless they are explicitly changed.

NOTEBOOK WORKFLOW:
1. Import the libraries.
   pandas loads and displays the data.
   scikit-learn provides the train-test split, preprocessing tools, SVR model, and evaluation metrics.
   matplotlib is used to plot actual and predicted values.

2. Load the dataset.
   pd.read_csv() loads teen_phone_addiction_dataset.csv into a pandas DataFrame.

3. Separate features and target.
   y contains Addiction_Level, the score to predict.
   X contains the other selected features.
   ID, Name, and Location are excluded because they are identifiers or high-cardinality location labels.

4. Identify numeric and categorical columns.
   Numeric columns are sent to StandardScaler.
   Text or category columns are sent to OneHotEncoder.

5. Split the examples.
   train_test_split divides the data into training and test sets.
   test_size=0.2 assigns 20 percent of rows to the test set.
   random_state=42 makes the split repeatable.

6. Create the preprocessing steps.
   ColumnTransformer applies scaling to numeric columns and one-hot encoding to categorical columns.

7. Create and train the SVR model.
   SVR(kernel="rbf") creates a support vector regression model with the RBF kernel.
   Pipeline combines preprocessing and SVR into one workflow.
   model.fit(X_train, y_train) learns the model from the training examples.

8. Predict test scores.
   model.predict(X_test) predicts Addiction_Level for the held-back test examples.

9. Evaluate the predictions.
   The predicted scores are compared with the actual y_test values using MAE, RMSE, and R-squared.

10. Plot actual versus predicted values.
    A bar graph displays the first ten actual test scores beside their predicted scores.
    Bars with similar heights show closer predictions for those examples.

EVALUATION RESULTS:
The recorded test-set results were:

    MAE:  0.316
    RMSE: 0.399
    R-squared: 0.937

MAE — Mean Absolute Error
    The model's predictions differed from the actual scores by about 0.316 points on average.

RMSE — Root Mean Squared Error
    The root mean squared prediction error was about 0.399 points.
    RMSE gives larger errors more weight than MAE.

R-squared — R2
    The model accounted for about 93.7 percent of the variation in the test-set scores compared with a baseline that predicts the average.
    R2 is not the percentage of predictions that are correct.

These results apply to the particular train-test split and model settings used. Results may differ if the data split or settings change.


ACTUAL V/S PREDICTED BAR GRAPH:
The bar graph compares the first ten actual Addiction_Level scores with the corresponding model predictions.

Each pair of bars represents one test example:
    One bar shows the actual score.
    The other bar shows the predicted score.

Bars with similar heights indicate closer predictions. The graph shows only a small sample; the evaluation metrics summarize performance across the full test set.
