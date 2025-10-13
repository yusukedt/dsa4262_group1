# Task 1 Objectives 
This README is a guide and reasoning for code reflected in the **svm_training.ipynb** that carries out training for the SVM model. **random_state = 42** is used to ensure reproducibility of the code.

## 1. Importing and Transformation of the data
The data is available as features and labels, in **dataset0.json** and **data.info.labelled** respectively. Each line of **dataset0.json** describes the nanopore direct RNA sequencing reads aligned to the reference transcript sequences.

### 1. Aggregation of feature data for all reads
Each line of the json file describes many sequencing reads. This imples hundreds of duplicates of transcript ID and position if we want to treat each read as a single item. Suppose we treat each read as a single item, the runtime of training the algorithm will be barbaric. Hence, while transforming the data, account for the averages of all 9 feature data for each unique transcript ID and position. Mean is preferred to median and geometric mean because it captures the variability of the data and is more suitable for magnitude data respectively. 

### 2. Accounting for imbalanced data
Observing the number of 0 labels to 1 labels shows that we are dealing with a severly imbalanced dataset. Imbalancement is a problem for SVM as there is bias toward the majority class, the labelled "0" records. As such, if the entire dataset is used without any method to balance out the classes, the trained model will almost always predict the label 0, which when used on unseen test data with a similar ratio of "0" records to "1" records, will generate high accuracy but a poor roc-auc measurement of 0.5, which is no better than the performance of a random classifier.

There are several ways to solve this issue. The following techniques are taken into consideration:
1. Class weighting while training the SVM (using `class_weight = "balanced"` in SVM to penalize misclassification of minority-class samples)
2. Resampling the data to reduce imbalance - undersampling the majority class (randomly sample a subset of "0" records instead of all "0" records, that roughly satisfies a ratio of 2 to 3:1 to the total number of "1" records)
3. Resampling the data to reduce imbalancement - oversampling the minority class using SMOTE (produce new minority samples to increase the number of "1" records)

After testing all three techniques and weighing their limitations, technique 3 will not be continued. The belief is that creating synthetic data does not reflect variability and distribution of actual data. It is also imperative to note that creating such data in large amounts, thousands upon thousands of records, is not accurate and will contribute to the final model's training ineffectiveness as it will overfit the noise (not good representations of either class) from synthetic data. Given that such data will make up a large proportion of the final data, it is unwise to proceed with SMOTE.

Settled on a hybrid between techniques 1 and 2; taking a larger ratio of "0" records to "1" records that is 6:1 (so as to capture as much real data as possible), and applying class weighting while training the SVM model to offset some of the imbalancement.


## 2. SVM model training 
Support Vector Machines, traditionally referred to as SVM, are a type of supervised machine learning algorithm generally used for binary classification. It is picked because it is a robust model for high-dimensional data (due to the large number of features used for training). The remaining dataset (after taking a random subset of the "0" records + all the "1" records) is split randomly into training and test data sets in a 4:1 ratio. Predictions of labels for the test data set are available in **dataset0_predictions.csv**.

## 3. SVM model optimization
Ran 5-fold Cross Validation with hyperparameter tuning of the SVM model, with the ranges of parameters defined as C: [1, 10, 100], gamma: [0.0001, 0.001, 0.01], kernel: ['rbf', 'linear'], in hopes to find the combination of parameters that optimizes the model most efficiently. 

Implemented GridSearch CV, a technique for hyperparameter tuning to search for this optimal combination. The final model has the parameters C = 100, gamma = 0.01 and kernel = 'rbf', which gives the highest accuracy among all combinations. This is performed on a subset of the data, with the number of "0" records to "1" records in a suitable ratio of 3:1 for quick tuning.

## 4. Final model evaluation
The optimized model has the following evaluation metrics:
- Accuracy, the proportion of correct predicted labels: 0.820.
- ROC-AUC Score, a useful metric for binary classification tasks: 0.872
- PR-AUC Score, a metric that measures the model's performance from classifying "1" records: 0.612
- F1 Score, a fair summary of the model's performance for imbalanced data: 0.568

### Final model predictions
Running the model on the provided test sets will give the probability scores. The **dataset1_predictions.csv** file is produced after testing the model on the dataset **dataset1.json** and is in the format as such:
| transcript_id | transcript_position | score |
| :------------- | :------------------: | -------------------: |
| Transcript ID for every row of data.json | Represents the position within transcript ID | Probability that the position within that transcript has m6A modification |


### Final Model
The final SVM model is stored as **trained_svm.pkl** .
