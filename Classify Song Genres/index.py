import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')
# Read in track metrics with the features
echonest_metrics = pd.read_json('datasets/echonest-metrics.json', precise_float = True)
# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = echonest_metrics.merge(tracks[['track_id','genre_top']],on='track_id')
# Inspect the resultant dataframe
echo_tracks.info()

# Create a correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

# Import train_test_split function and Decision tree classifier
# ... YOUR CODE ...
from sklearn.model_selection import train_test_split
# Create features
features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values
# Create labels
labels = echo_tracks["genre_top"].values
# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features,labels,random_state=10)

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler
# Scale the features and set the values to a new variable
scaler = StandardScaler()
# Scale train_features and test_features
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# This is just to make plots appear in the notebook
#%matplotlib inline
# Import our plotting module, and PCA class
#... YOUR CODE ...
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_
# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')

# Import numpy
import numpy as np
# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)
# Plot the cumulative explained variance and draw a dashed line at 0.85.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--')

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components=6, random_state=10)
# Fit and transform the scaled training features using pca
train_pca = pca.fit_transform(scaled_train_features)
# Fit and transform the scaled test features using pca
test_pca = pca.transform(scaled_test_features)

# Import Decision tree classifier
# ... YOUR CODE ...
from sklearn.tree import DecisionTreeClassifier
# Train our decision tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca,train_labels)
# Predict the labels for the test data
pred_labels_tree = tree.predict(test_pca)

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
# Train our logistic regression and predict labels for the test set
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca,train_labels)
pred_labels_logit = logreg.predict(test_pca)
# Create the classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)
print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)