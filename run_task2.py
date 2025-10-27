import pickle
import sys
import pandas as pd  # Assuming input is a CSV file; adjust if different
import json
import numpy as np

# Function to load the .pkl file (e.g., a model)
def load_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)  # Assuming it's a model; change var name if not
        return model
    except Exception as e:
        print(f"Error loading .pkl file: {e}")
        sys.exit(1)

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

# Main execution
if __name__ == "__main__":
    # Hardcoded specific file names as requested
    pkl_file = 'svm_train_2.pkl'  # Renamed from generic to svm_train_2.pkl
    input_file = 'SGNex_A549_directRNA_replicate6_run1/data.json'   
    
    loaded_model = load_pkl(pkl_file)
    

    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                record = json.loads(line)
                data.append(record)
    
    # For each record, use the function.
    parsed_data = []
    for line in data:
        parsed_data.extend(aggregation(line))

    # Data manipulation of the feature dataset to make it more readable
    df_features = pd.DataFrame(parsed_data)
    features_df = pd.DataFrame(df_features["features"].tolist(),
                           columns = [f"feature_{i+1}" for i in range(9)])
    new_df = pd.concat([df_features.drop(columns=["features"]), features_df], axis = 1)

    
    # Assuming the .pkl is a model that takes input_data for prediction
    # Adjust this based on what the model does (e.g., .predict, .transform, etc.)
    try:
        y_prob = loaded_model.predict_proba(new_df.iloc[:, 3:12])  # Example for scikit-learn model
        # Combine the test data set with the predictions and output the result
        result = pd.DataFrame()
        result["transcript_id"] = new_df["transcript_id"]
        result["transcript_position"] = new_df["position"]
        result["score"] = y_prob[:, 1]

        #Save the output as a csv file
        print("Output/Predictions:")
        result.to_csv("test_result.csv", index = False)
        
        
    except AttributeError:
        print("Loaded object doesn't have a 'predict' method. Inspecting type and contents:")
        print("Type:", type(loaded_model))
        print("Contents:", loaded_model)
    except Exception as e:
        print(f"Error processing input: {e}")
    
    # If you need to save the output, e.g., to a file

    # pd.DataFrame(output).to_csv('output.csv', index=False)
