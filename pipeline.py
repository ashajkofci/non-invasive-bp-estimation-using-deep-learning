import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak

data = pd.read_csv("data/features.csv")
data = data.sample(frac=1)
split_length = int(data.shape[0] * 0.8)

# train and test
train_data = data.iloc[:split_length]
test_data = data.iloc[split_length:]

label_name = ["label_sys", "label_dia"]
headers = data.columns.values.tolist()[1:-5]
print(headers)

# Initialize the classifier.
clf = ak.StructuredDataRegressor(max_trials=100)

# Evaluate
clf.fit(x=train_data[headers], y=train_data[label_name])

print(
    "Accuracy: {accuracy}".format(
        accuracy=clf.evaluate(x=test_data[headers], y=test_data[label_name])
    )
)