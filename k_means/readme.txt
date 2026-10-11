K-MEANS CLUSTERING - CREDIT CARD CUSTOMERS

PROJECT OVERVIEW:
This project uses K-means clustering to group credit card customers based on their credit limits and banking activity.
The notebook loads the Credit Card Customer Data dataset, checks for missing values, scales the selected features, uses the elbow method and silhouette scores to select a cluster count, fits K-means with three clusters, and visualizes the resulting customer groups.
This is an unsupervised learning task. The dataset has no target label that the model is trying to predict.

DATASET:
The dataset contains 660 customer records and 7 columns.
The notebook uses these five behavior and financial features:
Avg_Credit_Limit       Average credit limit
Total_Credit_Cards     Number of credit cards
Total_visits_bank      Number of visits to a bank
Total_visits_online    Number of online visits
Total_calls_made       Number of calls made

Sl_No and Customer Key are identifiers. They are not used as clustering features because their numeric values do not describe customer behavior.
The CSV in the ZIP archive should be extracted as:
    Credit Card Customer Data.csv
The notebook reads this CSV from the k_means folder.

WHAT IS K-MEANS?
K-means is an algorithm that groups similar records into clusters.
It works by:
1. Choosing a number of clusters, k.
2. Assigning each record to the nearest cluster center.
3. Recalculating each center as the average of the records assigned to it.
4. Repeating the assignments and center updates until the groups stabilize.
The cluster centers are called centroids. K-means uses distances between feature values, so scaling the features is important.
The cluster labels (0, 1, and 2) are identifiers only. They do not represent a ranking or a customer risk category.


NOTEBOOK WORKFLOW:
1. Import the libraries.
   pandas loads and displays the dataset.
   matplotlib creates plots.
   scikit-learn provides the imputer, scaler, K-means algorithm, and silhouette score function.
2. Load and inspect the dataset.
   pd.read_csv() loads Credit Card Customer Data.csv.
   df.shape reports the number of rows and columns.
   df.isna().sum() counts missing values in each column.
   df.head() previews the first rows.
3. Select customer features.
   The notebook selects Avg_Credit_Limit, Total_Credit_Cards, Total_visits_bank, Total_visits_online, and Total_calls_made.
   It excludes Sl_No and Customer Key because they are identifiers rather than meaningful measurements for grouping.
4. Handle missing values.
   SimpleImputer(strategy="median") fills missing feature values with the median of that feature.
   The current CSV has no missing values, so the imputer does not change the data. It is included as a precaution if a version with missing values is used.
5. Scale the features.
   StandardScaler transforms the selected features to comparable scales.
   This prevents Avg_Credit_Limit, which has much larger numbers, from dominating distance calculations.
   The transformation is approximately:

       scaled value = (value - feature mean) / feature standard deviation
6. Use the elbow method.
   The notebook fits K-means for k values from 1 through 9 and records each model's inertia.
   Inertia measures the total squared distance between records and their assigned cluster centers.
   The notebook plots k against inertia and identifies the bend near k=3 as a useful choice.
7. Calculate silhouette scores.
   silhouette_score measures how well-separated clusters are.
   Scores range from -1 to +1:
       Near +1: records fit their assigned cluster well.
       Near 0: clusters overlap or records lie between clusters.
       Below 0: some records may be closer to another cluster.
   The notebook reports k=3 as having the highest silhouette score among the tested values.
8. Fit the final K-means model.
   KMeans(n_clusters=3, random_state=42, n_init=10) creates the final clustering model.
   fit_predict() assigns each row to a cluster.
   random_state=42 makes the result repeatable.
   n_init=10 runs the algorithm multiple times with different starting centers and keeps a good result.
9. Summarize the clusters.
   The notebook groups the data by Cluster and calculates the average value of each selected feature.
   These averages help describe how customer groups differ.
10. Plot the clusters.
    A scatter plot displays Avg_Credit_Limit on the x-axis and Total_visits_online on the y-axis.
    Each point represents one customer and its color represents its cluster.
    K-means used all five selected features; the chart displays only two of them.

ELBOW METHOD:
The elbow method compares inertia for different values of k.
Inertia decreases as more clusters are added. The goal is to find a bend in the curve where adding more clusters gives a smaller improvement. The notebook's notes select k=3 based on the elbow graph.
The elbow is a visual guide. It may not always identify one obvious best value.

SILHOUETTE SCORE:
The silhouette score compares how close a record is to its own cluster with how close it is to the nearest neighboring cluster.
A higher average score generally means better-separated clusters. The notebook's notes report that k=3 has the highest silhouette score among the values tested.
Silhouette score is a guide, not proof that the groups are meaningful. Consider it alongside the elbow graph and the cluster averages.

CLUSTER INTERPRETATION:
The notebook's scatter plot suggests these patterns:
Cluster 0 (purple):
    Mostly lower to mid credit limits and few online visits, generally around 0 to 3.
Cluster 1 (teal):
    Generally higher credit limits and more online visits, often around 6 to 15.
Cluster 2 (yellow):
    Mostly lower credit limits and a moderate number of online visits, around 3 to 5.
    The plot also contains an exception with about 50,000 average credit limit and 10 online visits.
These descriptions are based on a plot of only two features. Since K-means used five features, some points may belong to a cluster because of their card count, bank visits, or calls, even if they look unusual in this two-feature plot.
Cluster numbers are arbitrary labels; they do not mean that one group is better, worse, safer, or riskier.
