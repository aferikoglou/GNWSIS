import argparse
import pandas as pd
from modules.applicationDataAnalyzer import ApplicationDataAnalyzer

def main():
    """
    Entry point for the script. Parses command-line arguments and performs analysis
    on a specified application from the GNΩSIS dataset.

    Steps:
    - Parses the application name from command-line arguments.
    - Loads the GNΩSIS dataset from CSV.
    - Filters the dataset for the specified application.
    - Initializes the ApplicationDataAnalyzer and performs analysis.
    """

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Extracts and analyzes key insights for a specific application from the GNΩSIS dataset.')
    parser.add_argument('--APPLICATION_NAME', type=str, required=True, help='Specifies the target application for analysis.')
    args = parser.parse_args()

    application_name = args.APPLICATION_NAME

    # Load the GNΩSIS dataset
    df = pd.read_csv("GNΩSIS.csv")
    print("Loaded GNΩSIS dataset:\n")
    print(df.shape)
    print(df.head())

    # Filter the dataset for the specified application
    df_app = df[df["Application_Name"] == application_name]
    print(f"\nFiltered dataset for application: {application_name}\n")
    print(df_app.shape)
    print(df_app.head())

    # Perform application-specific analysis
    application_data_analyzer = ApplicationDataAnalyzer(application_name, df_app)
    application_data_analyzer.analyze()

if __name__ == "__main__":
    main()
