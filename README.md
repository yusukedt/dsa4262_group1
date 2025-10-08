# Task 1 Objectives 
This README is a guide and reasoning for code reflected in the two notebooks that carry out training for the SVM model and using the model for predicting unseen data respectively. **random_state = 42** is used to ensure reproducibility of the code.

## 1. Importing and Transformation of the data
Firstly, the data is available as features and labels, in **dataset0.json** and **data.info.labelled** respectively. Each line of **dataset0.json** describes the nanopore direct RNA sequencing reads aligned to the reference transcript sequences.

### 1. Aggregation of feature data for all reads
Each line of the json file describes many sequencing reads. This imples hundreds of duplicates of transcript ID and position if we want to treat each read as a single item. Suppose we treat each read as a single item, the runtime of training the algorithm will be barbaric. Hence, while transforming the data, account for the averages of all 9 feature data for each unique transcript ID and position. Mean is preferred to median and geometric mean because it captures the variability of the data and is more suitable for magnitude data respectively. 

### 2. Accounting for imbalanced data
Observing the number of 0 labels to 1 labels shows that we are dealing with a severly imbalanced dataset. Imbalance is a problem for SVM as there is bias toward the majority class, the 0 label here. As such, the model will almost always predict the label 0, which garners high accuracy but poor roc-auc measurement, 0.5, which reflects that of a random classifier.

There are several ways to solve this. I took into account the following techniques:
1. Class weighting in SVM (using `class_weight = "balanced"` in SVM to penalize misclassification of minority-class samples)
2. Resampling the data to reduce imbalance - undersampling the majority class (randomly take a number of "0" records that satisifes a 3:1 ratio to the number of "1" records)
3. Resampling the data to reduce imbalance - oversampling the minority class using SMOTE (produce new minority samples to increase the number of "1" records)

Each way can change the final model output and its evaluation metrics. I tested all three techniques and weighed their limitations. I believe that creating synthetic data may not reflect variability of actual data. It is also imperative to note that replicating such data in large amounts, thousands on thousands of records, is not accurate and can contribute to the final model's training ineffectiveness.

Therefore, I settled on a hybrid between techniques 1 and 2; taking a larger ratio of "0" records to "1" records that is 6:1 (so as to capture as much real data as possible), and applying class weighting while training the SVM model.


## 2. SVM model training 
Support Vector Machines, traditionally referred to as SVM, are a type of supervised machine learning algorithm generally used for binary classification. I picked it as it is a robust model for high-dimensional data (due to the large number of features used for training). The entire dataset is split randomly into training and test data sets in a 4:1 ratio.

## 3. SVM model optimization
I ran 5-fold Cross Validation with hyperparameter tuning of the SVM model, with the range of parameters defined as C: [1, 10, 100], gamma: [0.0001, 0.001, 0.01], kernel: ['rbf', 'linear'], in hopes to find the combination of parameters that optimizes the model most efficiently. 

Implemented gridSearch CV, a technique for hyperparameter tuning to search for this optimal combination. The final model has the parameters C = 10, gamma = 0.01 and kernel = 'rbf' that gives the **highest accuracy** among all combinations of parameters. This is performed on a subset of the data, with the number of "0" records to "1" records in the ratio of 3:1 for quick tuning.

## 4. Final model evaluation
The optimized model has the following evaluation metrics:
- Accuracy: 0.820
- ROC-AUC Score: 0.872
- PR-AUC Score: 0.612
- F1 Score: 0.568

## 5. Final model output
Running the model on the provided test sets will give the probability scores. The **dataset1_predictions.csv** file is produced after testing the model on the dataset **dataset1.json** and is in the format as such:
| transcript_id | transcript_position | score |
| :------------- | :------------------: | -------------------: |
| Transcript ID for every row of data.json | Represents the position within transcript ID | Probability that the position within that transcript has m6A modification |


## Final Model
The final SVM model is stored as **trained_svm.pkl** .
