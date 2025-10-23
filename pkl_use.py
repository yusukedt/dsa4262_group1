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
    if len(sys.argv) < 3:
        print("Usage: python3 run_pkl.py <pkl_file> <input_file>")
        print("Example: python3 run_pkl.py model.pkl input.csv")
        sys.exit(1)
    
    pkl_file = sys.argv[1]  # e.g., 'model.pkl'
    input_file = sys.argv[2]  # e.g., 'input.csv' or whatever file it takes
    
    loaded_model = load_pkl(pkl_file)
    
    # Load the input data
    input_data = load_input(input_file)
    
    # Assuming the .pkl is a model that takes input_data for prediction
    # Adjust this based on what the model does (e.g., .predict, .transform, etc.)
    try:
        output = loaded_model.predict(input_data)  # Example for scikit-learn model
        print("Output/Predictions:")
        print(output)
    except AttributeError:
        print("Loaded object doesn't have a 'predict' method. Inspecting type and contents:")
        print("Type:", type(loaded_model))
        print("Contents:", loaded_model)
    except Exception as e:
        print(f"Error processing input: {e}")
    
    # If you need to save the output, e.g., to a file
    # pd.DataFrame(output).to_csv('output.csv', index=False)
