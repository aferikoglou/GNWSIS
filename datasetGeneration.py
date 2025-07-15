import os
import argparse
import pandas as pd
from modules.arrayAnalyzer import ArrayAnalyzer
from modules.loopAnalyzer import LoopAnalyzer

def generate_source_code_feature_vector(MODE):
    """
    Generates the source code feature vector dataset for the specified mode.

    Args:
        MODE (str): The mode in which the script runs, either 'CollectiveHLS' or 'HLSDataset'.
    
    Returns:
        pd.DataFrame: A DataFrame containing the feature vectors for each application.
    """
    # Initialize analyzers based on the mode
    array_analyzer = ArrayAnalyzer(MODE)
    loop_analyzer = LoopAnalyzer(MODE)
    
    # Get column names for arrays, loops, and operations
    array_column_names = array_analyzer.get_array_column_names()
    loop_column_names = loop_analyzer.get_loop_column_names()
    operation_column_names = loop_analyzer.get_operation_column_names()

    # Define application name column based on mode
    app_name_column = "app_name" if MODE == "CollectiveHLS" else "Application_Name"
    column_names = [app_name_column] + array_column_names + loop_column_names + operation_column_names

    # Initialize an empty DataFrame with the column names
    source_code_dataset_df = pd.DataFrame(columns=column_names)

    # Iterate through all application datasets
    for app_name in sorted(os.listdir('./data/ApplicationDataset')):
        try:
            # Generate maps for arrays, loops, and operations
            array_map = array_analyzer.get_app_array_map(app_name)
            loop_map = loop_analyzer.get_app_loop_map(app_name)
            operations_map = loop_analyzer.get_app_operations_map(app_name)
            
            # Combine all feature vectors into a single application map
            application_map = {app_name_column: app_name}
            application_map.update(array_map)
            application_map.update(loop_map)
            application_map.update(operations_map)

            # Append the combined map as a new row to the DataFrame
            source_code_dataset_df = pd.concat(
                [source_code_dataset_df, pd.DataFrame([application_map])], ignore_index=True
            )
        except KeyError as e:
            print(f"Error processing {app_name}: {e}")
            continue

    return source_code_dataset_df

def save_feature_vector_to_csv(df, output_path):
    """
    Saves the generated feature vector DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing the feature vectors.
        output_path (str): The file path to save the CSV.
    """
    df.to_csv(output_path, index=False)

def main():
    """
    Main function to run the source code feature vector generation process.
    """
    parser = argparse.ArgumentParser(description='A script to generate the source code dataset.')
    parser.add_argument('--MODE', type=str, required=True, choices=['CollectiveHLS', 'HLSDataset'], help='Mode to run the script')
    args = parser.parse_args()

    MODE = args.MODE
    print("Started source code feature vector generation...\n")

    # Generate the feature vector dataset
    feature_vector_df = generate_source_code_feature_vector(MODE)

    # Save the dataset to a CSV file
    save_feature_vector_to_csv(feature_vector_df, "./data/SourceCodeFeatureVectors.csv")

    print("\nFinished source code feature vector generation...")

if __name__ == "__main__":
    main()
