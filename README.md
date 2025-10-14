# Task 1 Objectives 
This README is a guide and reflects the thought process for code in training the SVM model. **random_state = 42** is used to ensure reproducibility of results.

## 1. Importing and Transformation of the training data
The data is available as features and labels, in **dataset0.json** and **data.info.labelled** respectively. Each line of **dataset0.json** describes the nanopore direct RNA sequencing reads aligned to the reference transcript sequences.

### 1. Aggregation of feature data for all reads
Each line of the json file describes many sequencing reads. This imples hundreds of duplicates of transcript ID and position if we want to treat each read as a single item. Suppose we treat each read as a single item, the runtime of training the algorithm will take too long. To ensure sanity, while transforming the data, account for the averages of all 9 feature data for each unique transcript ID and position. Mean is preferred to median and geometric mean because it captures the variability of the data and is more suitable for magnitude data respectively. 

### 2. Splitting data
To prevent data leakage (in the event that the test set contains records with `gene_id` already found in the training set), the dataset is split into training and test sets without having `gene_id` repeated across training and test sets. This ensures the test set truly simulates unseen data.

### 3. Accounting for imbalanced data
Observing the number of 0 labels to 1 labels in the training set shows that we are dealing with a severly imbalanced dataset. Class imbalance is a problem for SVM because there will be bias toward the majority class, which is the labelled "0" records. If the entire dataset is used without any method to balance out the classes, the trained model will almost always predict the label 0, which when used on unseen test data with a similar ratio of "0" records to "1" records, will generate high accuracy but a poor roc-auc measurement of 0.5, which is no better than the performance of a random classifier.

There are several ways to solve this issue. The following techniques are taken into consideration:
1. Class weighting while training the SVM (using `class_weight = "balanced"` in SVM to penalize misclassification of minority-class samples)
2. Resampling the data to reduce imbalance - undersampling the majority class (randomly sample a subset of "0" records instead of all "0" records, that roughly satisfies a ratio of 2 to 3:1 to the total number of "1" records)
3. Resampling the data to reduce imbalancement - oversampling the minority class using SMOTE (produce new minority samples to increase the number of "1" records)

After testing all three techniques and weighing their limitations, technique 3 will not be continued. The belief is that creating synthetic data does not reflect variability and distribution of actual data. It is also imperative to note that creating such data in large amounts, thousands upon thousands of records, is not accurate and will contribute to the final model's training ineffectiveness because it overfits the noise (not good representations of either class) from synthetic data. And given that such data will make up a large proportion of the final data, it is unwise to proceed with SMOTE.

Settled on a hybrid between techniques 1 and 2; taking a larger ratio of "0" records to "1" records that is near to 4:1 (so as to capture as much real data as possible) - mild imbalance, and applying class weighting while training the SVM model to offset some of this imbalance.


## 2. The SVM model
### Introduction
Support Vector Machines, traditionally referred to as SVM, are a type of supervised machine learning algorithm generally used for binary classification. It is picked because it is a robust model for high-dimensional data (due to the large number of features used for training).

### Optimization
Implemented GridSearchCV, a technique for hyperparameter tuning to search for an optimal combination of parameters. This technique applies 5-fold Cross Validation while tuning the model, using ranges of parameters defined as C: [1, 10, 100], gamma: [0.0001, 0.001, 0.01], kernel: ['rbf', 'linear'].

The final model has the parameters C = 100, gamma = 0.01 and kernel = 'rbf', which gives the highest accuracy among all combinations. This is performed on a subset of the training data, with the number of "0" records to "1" records in the near ratio of 1:1 for quick tuning. The objective of this step is to simply find the best combination of parameters, judged using the default scoring metric of accuracy.

### Model 1 evaluation
The optimized model has the following evaluation metrics:
- Accuracy: 0.826
- ROC-AUC Score: 0.863
- PR-AUC Score: 0.323
- F1 Score: 0.568
- Precision: 0.156
- Recall: 0.754

### Model 2 training and evaluation
A second model is trained with the same parameters but differently; instead of taking the first 16000 records from a shuffled dataframe in the case for the first model, the last 16000 records were taken. 

This second model produces the following evaluation metrics:
- SVM Accuracy: 0.8284331373254931
- ROC_AUC Score: 0.8598173673300131
- PR_AUC Score: 0.3095739442981673
- F1 Score: 0.2616629368221725
- Precision: 0.15816857440166493
- Recall: 0.7569721115537849

### Model comparison and decision
While both models produce roughly similar evaluation metrics, the second model is preferred. Compare the confusion matrices of both models:
Model 1:
[[19899  4096]
 [  247   757]]
 Model 2:
 [[19950  4045]
 [  244   760]]
 Predicting true positives and true negatives more accurately sets the second model apart from the first model. Note that low precision and PR_AUC scores do not mean the model is inherently bad; this is just a case of working with imbalanced test data.

### Model predictions
Running the model on the test set gives the probability scores. The predictions for test data should be in the format as such:
| transcript_id | transcript_position | score |
| :------------- | :------------------: | -------------------: |
| Transcript ID for every row of data.json | Represents the position within transcript ID | Probability that the position within that transcript has m6A modification |

### 
### Final Model
The final SVM model is stored as **svm_train_2.pkl** .
