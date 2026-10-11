POLYNOMIAL REGRESSION - SYNTHETIC QUADRATIC DATA

PROJECT OVERVIEW:
This project demonstrates polynomial regression using a small synthetic dataset generated in the notebook.
It first fits ordinary linear regression to the data, then transforms the input into polynomial features and fits a degree-2 model.
The notebook plots the points and fitted models, and compares their performance using the R-squared score.

WHAT THE MODEL PREDICTS:
The model predicts y from a single input feature, X.
The generated target follows an approximate quadratic relationship with added random noise:
    y = 0.8 * X^2 + 0.9 * X + 2 + noise

The notebook creates 200 examples. X values are sampled between -3 and 3, and random noise is added to each target value.
Because no random seed is set for dataset generation, the exact data and evaluation scores can vary between runs.

INPUT FEATURE:
X    A generated numeric input value between -3 and 3

WHAT IS POLYNOMIAL REGRESSION?
Polynomial regression models a curved relationship by expanding an input into powers of that input, then fitting a linear regression model to the expanded features.
The regression remains linear in its learned coefficients, while its predictions can form a curve in the original X-y plot.

For a degree-2 polynomial with a bias term, the model is:
    predicted y = b0 + b1 * X + b2 * X^2

b0 is the intercept, b1 is the coefficient for X, and b2 is the coefficient for X^2. The model learns these values from the training data.

NOTEBOOK WORKFLOW
1. Import the libraries.
   NumPy creates and reshapes arrays.
   Matplotlib plots the data and fitted models.
   scikit-learn provides the train-test split, polynomial feature transformer, linear regression model, and R-squared metric.

2. Generate the dataset.
   The notebook creates 200 random X values in the range -3 to 3.
   It computes y using a quadratic equation and adds random noise.

3. Plot the data.
   A scatter plot shows the generated X and y values before fitting a model.

4. Split the examples.
   train_test_split holds back 20 percent of the examples for evaluation.
   random_state=2 makes the train-test split repeatable, although the generated dataset itself changes between runs.

5. Fit ordinary linear regression.
   LinearRegression() fits a straight line using the training examples.
   The notebook predicts values for the test examples and calculates the R-squared score.
   Since the data follows a curved relationship, a straight line cannot capture its shape well.

6. Expand the features.
   PolynomialFeatures(degree=2, include_bias=True) transforms each X value into polynomial terms, including a constant bias term, X, and X squared.
   The transformer is fitted on the training inputs and then applied to the test inputs.

7. Fit polynomial regression.
   LinearRegression() learns coefficients from the transformed training inputs and their targets.
   The fitted model is used to predict the test targets and to calculate another R-squared score.

8. Plot the fitted curve.
   The notebook creates evenly spaced X values from -3 to 3, transforms them with the same polynomial transformer, and predicts y values.
   The final plot shows the fitted curve along with the training and test points.

EVALUATION: R-SQUARED (R2)
R-squared compares the model's predictions with a baseline that always predicts the average target.
A score of 1 means a perfect fit, 0 means performance similar to that baseline, and a negative score means worse than the baseline on the evaluated data.
A higher score indicates a better fit for this dataset, but it is not the percentage of predictions that are correct.
The notebook reports a polynomial-model score around 0.9199 in its markdown notes; because the generated data includes unseeded randomness, the score may differ when rerun.


