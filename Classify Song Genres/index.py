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