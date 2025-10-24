import pickle
import sys
import pandas as pd  # Assuming input is a CSV file; adjust if different

# Function to load the .pkl file (e.g., a model)
def load_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)  # Assuming it's a model; change var name if not
        return model
    except Exception as e:
        print(f"Error loading .pkl file: {e}")
        sys.exit(1)

# Function to load input data (e.g., from a CSV file)
def load_input(input_file):
    try:
        data = pd.read_csv(input_file)  # Adjust if input is JSON, text, etc.
        return data
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

# Main execution
if __name__ == "__main__":
    # Hardcoded specific file names as requested
    pkl_file = 'svm_train_2.pkl'  # Renamed from generic to svm_train_2.pkl
    input_file = 'test_set.csv'    # Renamed from generic to test_set.csv
    
    loaded_model = load_pkl(pkl_file)
    
    # Load the input data
    input_data = load_input(input_file)

    
# Group by transcript_id and transcript_position, average for all feature columns
agg_data = (
    input_data.groupby(["transcript_id", "transcript_position"], as_index=False)
      .agg({
        # Unlike the features, do not aggregate gene_id and sequence columns (store the first occurrence of each column)
          "gene_id": "first",
          "sequence": "first",
          **{col: "mean" for col in features}  # average of features
      })
)

    
    # Assuming the .pkl is a model that takes input_data for prediction
    # Adjust this based on what the model does (e.g., .predict, .transform, etc.)
    try:
        output = loaded_model.predict_proba(agg_data.iloc[:, 4:13])  # Example for scikit-learn model
        # Combine the test data set with the predictions and output the result
        result = pd.DataFrame()
        result["transcript_id"] = agg_data["transcript_id"]
        result["transcript_position"] = agg_data["transcript_position"]
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
