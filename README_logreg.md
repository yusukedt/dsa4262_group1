# m6A Modification Prediction Pipeline

This repository contains scripts for generating datasets, training a logistic regression model, and predicting m6A modification sites in RNA sequences.

---

## train.py

The `train.py` script is used to train the m6A prediction model. Key outputs and their meanings:

- **Dataset loading and summary**

  Dataset shape: (128461, 14)  
  Columns: ['gene_id', 'transcript_id', 'transcript_position', 'sequence', 'label', 'feature_1', ..., 'feature_9']  
  m6A modification rate: 0.0291  
  Number of unique genes: 50  
  Number of unique transcripts: 62  

  The dataset contains 128,461 samples with 14 columns, including sequence features and labels.  
  m6A sites are rare, with ~2.9% of positions modified.

- **Data splitting**

  Train genes: 35, samples: 68749, m6A ratio: 0.0395  
  Val genes: 5, samples: 30726, m6A ratio: 0.0150  
  Test genes: 10, samples: 28986, m6A ratio: 0.0192  

  The dataset is split by gene_id into train/validation/test sets to avoid leakage.  
  m6A modification rates vary across splits.

- **Feature engineering**

  Original features: 9  
  Engineered features: 23  

  Additional features were generated to improve model performance.

- **Class balancing**

  Original class distribution: [66034  2715]  
  Resampled class distribution: [66034 66034]  

  SMOTE is applied to balance the dataset for rare m6A sites.

- **Cross-validation performance**

  CV AUC-ROC: 0.6975 ± 0.0274  
  CV AUC-PR: 0.6320 ± 0.0175  

  The model shows reasonable predictive ability in cross-validation.

- **Final model metrics**

  Training AUC-ROC: 0.6924  
  Validation AUC-ROC: 0.7735  
  Test AUC-ROC: 0.7050  

  Performance is consistent across train, validation, and test sets.  
  Average precision values are low due to class imbalance in the original data.

- **Outputs**

  Model saved to `model/m6a_model.pkl`  
  Training statistics saved to `model/m6a_model_stats.json`

---

## generate.py

The `generate.py` script generates new test data for prediction. Key outputs:

- **Dataset summary**

  Total samples: 1000  
  Unique genes: 20  
  Unique transcripts: 20  
  m6A modification rate: 0.1340  
  m6A positive samples: 134  

  Generates 1,000 samples with 13 features for prediction.  
  Approximately 13.4% of positions are m6A-modified in the generated dataset.

- **Feature statistics**

  feature_1: mean=0.0057, std=0.0029, min=0.0008, max=0.0215  
  feature_2: mean=4.0159, std=2.9567, min=0.0647, max=23.2700  
  ...  
  feature_9: mean=94.1583, std=15.8096, min=46.1828, max=153.8936  

  Shows the distribution of numeric features in the generated dataset.

- **Sequence analysis**

  Most frequent sequences: AAACC (15.0%), AGACC (13.1%), GGACT (11.8%), TAACC (10.3%), GAACC (10.0%), AGACT (9.9%)  

  Summarizes the frequency of common 5-mer sequences in the dataset.

- **Output**

  Prediction data saved to `test_data/test_genes_predict.csv`

---

## predict.py

The `predict.py` script predicts m6A modification probabilities using the trained model.

- **Input data**

  Loaded 1000 samples from `test_data/test_genes_predict.csv`  
  Columns: ['gene_id', 'transcript_id', 'transcript_position', 'sequence', 'feature_1', ..., 'feature_9']

  Uses the generated test dataset.

- **Model information**

  Model trained with 23 features  
  Original features: 9  
  Balance strategy: SMOTE  
  Scaler type: standard  

  Shows feature count, class balancing method, and scaling used during training.

- **Prediction statistics**

  Total predictions: 1000  
  Score distribution - Min: 0.0000, Max: 1.0000, Mean: 0.4997, Median: 0.4612  
  Predictions > 0.5: 482 (48.2%)  

  Predictions are probabilities between 0 and 1.  
  Roughly half of the samples have predicted scores above 0.5.

- **Sample predictions**
```
      transcript_id  transcript_position     score
  0  ENST00000200042                  101  0.594985
  1  ENST00000200042                  233  0.182885
  2  ENST00000200042                  341  1.000000
  3  ENST00000200042                  455  0.005807
  4  ENST00000200042                  504  0.739772
```
  Example rows of predicted scores for specific transcript positions.

- **Output**

  Predictions saved to `output/predictions.csv`

---

## Summary

- `train.py` → trains the m6A prediction model with feature engineering and SMOTE balancing.  
- `generate.py` → generates test data for model prediction.  
- `predict.py` → predicts m6A probabilities on new sequences using the trained model.  

All scripts save outputs in the `model/` or `test_data/` folders for reproducibility and downstream analysis.
