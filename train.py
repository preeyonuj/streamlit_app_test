import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

seed = 42

# Read original dataset
iris_df = pd.read_csv("data/iris.csv")
iris_df.sample(frac=1, random_state=seed)
# selecting features and target data
X = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_df[['variety']]
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X, y)
# save the model to disk
pickle.dump(clf, open("rf_model.sav", 'wb'))