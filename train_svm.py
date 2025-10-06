# Load the JSON file, reading each line and storing it as an element in a list
import gzip
import json
import pandas as pd
import numpy as np

data = []
with gzip.open("dataset0.json.gz", 'rt', encoding='utf-8') as f: 
    for line in f:
        # Skip empty lines
        if line.strip():
            record = json.loads(line)
            data.append(record)

# Load in the m6A labels
data_labels = pd.read_csv("data.info.labelled")

# Aggregate numbers for each unique transcript ID and pos, using mean
def aggregation(record):
    aggregates = []

    for transcript_id, value in record.items():
        for pos, value1 in value.items():
            for mers, value2 in value1.items():

                arr = np.array(value2, dtype=float)
                means = arr.mean(axis=0).tolist()
                aggregates.append({
                    "transcript_id": transcript_id,
                    "position": int(pos),
                    "kmer": mers,
                    "features": means
                })
    return aggregates

# For each record, use the function.
parsed_data = []
for line in data:
    parsed_data.extend(aggregation(line))

# Data manipulation to join and form the X and y subsets
df_features = pd.DataFrame(parsed_data)
features_df = pd.DataFrame(df_features["features"].tolist(),
                           columns = [f"feature_{i+1}" for i in range(9)])
new_df = pd.concat([df_features.drop(columns=["features"]), features_df], axis = 1)
df_labels = pd.DataFrame(data_labels)
ndf = pd.concat([df_labels, new_df.drop(columns=["transcript_id", "position"])], axis = 1)

#  To account for a severe case of imbalanced label ratio;
# the number of label 0 to that of label 1 is very skewed to the former,
# Consider taking a 3:1 ratio of the number of label 0 to that of label 1
df0 = ndf[ndf["label"] == 0]
df1 = ndf[ndf["label"] == 1]
# There are 5475 label 1 and 116363 label 0 (do len(df0), len(df1))
df0_sub = df0.iloc[:15000]
dfs = pd.concat([df0_sub, df1], ignore_index=True)

# Identify the features column(s) and label column
X = dfs.iloc[:, -9:]
y = dfs["label"]

# We can split the data into training and testing sets in a 4:1 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# We will use the SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, f1_score

# Note: SVM is slow on large datasets

# Train SVM on the training data, with the following parameter values
# (the best of which) as a result after hyperparameter tuning
svm_model = SVC(kernel='rbf', C = 10, gamma = 0.01)
svm_model.fit(X_train, y_train)

# After training the model, we can test on our validation set
y_pred = svm_model.predict(X_test)

# Evaluate
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("PR AUC Score:", average_precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# We want to output the data showing prediction and actual labels side by side
result = X_test.copy()
result["y_actual"] = y_test
result["y_predicted"] = y_pred
result["is_matched"] = result["y_actual"] == result["y_predicted"]