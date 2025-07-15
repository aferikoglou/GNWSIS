import os
import argparse

from modules.dbReader import DBReader
from modules.applicationDataAnalyzer import ApplicationDataAnalyzer
from modules.paretoDataAnalyzer import ParetoDataAnalyzer

def str2bool(v):
    """
    Converts a string representation of a boolean value to a boolean type.
    
    Handles common string values for true/false such as 'yes', 'no', 'true', 'false', etc.

    Args:
        v (str or bool): The string or boolean value to convert.

    Returns:
        bool: The corresponding boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string does not represent a valid boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Main function to initialize and read databases using DBReader."""
    
    if READ_DATABASES:
        print("Started database read...\n")
        
        db_reader = DBReader()
        db_reader.read_databases()
        
        print("Finished database read...\n")
    
    if PER_APPLICATION_ANALYSES:
        print("Started application data analyses...\n")
        
        for fname in os.listdir("./data/CSVS"):
            # Pathfinder Coalescing from RodiniaHLS does not contain DB for MPSoC UltraScale+ ZCU104 at 300MHz
            application_data_analyzer = ApplicationDataAnalyzer(fname)
            application_data_analyzer.analyze()

        print("\nFinished application data analyses...")
    
    print("Started Pareto optimal data analyses...\n")
    
    pareto_data_analyzer = ParetoDataAnalyzer()
    pareto_data_analyzer.analyze()
        
    print("\nFinished Pareto optimal data analyses...")
     
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='A script to retrieve data from our databases and conduct analyses.')
    parser.add_argument('--READ_DB', type=str2bool, default=False, help='Indicator for database reading.')
    parser.add_argument('--PER_APPLICATION_ANALYSES', type=str2bool, default=False, help='Indicator for per application analyses.')
    args = parser.parse_args()

    READ_DATABASES = args.READ_DB
    PER_APPLICATION_ANALYSES = args.PER_APPLICATION_ANALYSES

    main()