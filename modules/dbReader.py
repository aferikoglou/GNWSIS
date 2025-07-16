import os
import warnings

import pandas as pd

from sqlitedict import SqliteDict
from modules.directivesManipularor import DirectivesManipulator

warnings.simplefilter(action='ignore', category=FutureWarning)

class DBReader:
    def __init__(self):
        """
        Initializes the DBReader class with dataset, database, and CSV paths,
        along with application names, available devices, and clock periods.
        """
        self.DATASET_DIR = "./data/ApplicationDataset"
        self.DATABASE_DIR = "./data/ApplicationDatabases"
        self.CSVS_DIR = "./CSVS"

        os.makedirs(self.CSVS_DIR, exist_ok = True)
        
        # List of application names to process
        self.APPLICATION_NAMES_LIST = [
            "machsuite-gemm-blocked",
            "machsuite-gemm-ncubed",
            "machsuite-md-knn",
            "machsuite-sort-radix",
            "machsuite-spmv-ellpack",
            "machsuite-stencil2d",
            "machsuite-stencil3d",
            "machsuite-viterbi",
            "rodinia-backprop-0-baseline-back",
            "rodinia-backprop-0-baseline-forward",
            "rodinia-backprop-1-tiling-back",
            "rodinia_cfd_flux_0_baseline_0",
            "rodinia_cfd_step_factor_0_baseline_0",
            "rodinia_cfd_step_factor_1_tiling_0",
            "rodinia_cfd_step_factor_2_pipeline_0",
            "rodinia_cfd_step_factor_3_unroll_0",
            "rodinia_cfd_step_factor_4_doublebuffer_0",
            "rodinia_cfd_step_factor_5_coalescing_0",
            "rodinia_dilate_0_baseline_0",
            "rodinia_dilate_1_tiling_0",
            "rodinia_dilate_2_pipeline_0",
            "rodinia_dilate_3_pipeline_0",
            "rodinia_lavaMD_0_baseline",
            "rodinia_lavaMD_1_tiling_0",
            "rodinia_lavaMD_1_tiling_1",
            "rodinia_lavaMD_2_pipeline_0",
            "rodinia_lc_gicov_0_baseline_0",
            "rodinia_lc_mgvf_0_baseline_0",
            "rodinia_lud_0_baseline_0",
            "rodinia_lud_1_tiling_0",
            "rodinia_pathfinder_0_baseline_0",
            "rodinia_pathfinder_4_doublebuffer_0",
            "rodinia_pathfinder_5_coalescing_0",
            "rodinia_streamcluster_0_baseline_0",
            "rodinia_streamcluster_1_tiling_0",
            "rodinia_streamcluster_2_pipeline_1",
            "rodinia_streamcluster_3_doublebuffer_0",
            "rodinia_streamcluster_4_coalescing_0",
            "rodinia-hotspot-0-baseline",
            "rodinia-hotspot-1-tiling",
            "rodinia-hotspot-2-pipeline",
            "rodinia-hotspot-3-unroll",
            "rodinia-kmeans-0-baseline",
            "rodinia-kmeans-1-tiling",
            "rodinia-kmeans-2-pipeline",
            "rodinia-kmeans-3-unroll",
            "rodinia-knn-0-baseline",
            "rodinia-knn-1-tiling",
            "rodinia-knn-2-pipeline",
            "rodinia-knn-3-unroll",
            "rodinia-knn-4-doublebuffer",
            "rodinia-knn-5-coalescing",
            "serrano-kalman-filter",
            "spcl_example_00",
            "spcl_example_01",
            "spcl_example_03",
            "spcl_example_05"
        ]

        # Column names for the output dataframe of each application
        self.APPLICATION_OUTPUT_DATAFRAME_COLUMN_NAMES = [
                'Version',
                'Device',
                'Clock_Period_nsec',
                'Latency_msec', 
                'Synthesis_Time_sec',
                'BRAM_Utilization_percentage',
                'DSP_Utilization_percentage', 
                'FF_Utilization_percentage',
                'LUT_Utilization_percentage',
                "Array_1",
                "Array_2",
                "Array_3",
                "Array_4",
                "Array_5",
                "Array_6",
                "Array_7",
                "Array_8",
                "Array_9",
                "Array_10",
                "Array_11",
                "Array_12",
                "Array_13",
                "Array_14",
                "Array_15",
                "Array_16",
                "Array_17",
                "Array_18",
                "Array_19",
                "Array_20",
                "Array_21",
                "Array_22",
                "OuterLoop_1",
                "OuterLoop_2",
                "OuterLoop_3",
                "OuterLoop_4",
                "OuterLoop_5",
                "OuterLoop_6",
                "OuterLoop_7",
                "OuterLoop_8",
                "OuterLoop_9",
                "OuterLoop_10",
                "OuterLoop_11",
                "OuterLoop_12",
                "OuterLoop_13",
                "OuterLoop_14",
                "OuterLoop_15",
                "OuterLoop_16",
                "OuterLoop_17",
                "OuterLoop_18",
                "OuterLoop_19",
                "OuterLoop_20",
                "OuterLoop_21",
                "OuterLoop_22",
                "OuterLoop_23",
                "OuterLoop_24",
                "OuterLoop_25",
                "OuterLoop_26",
                "InnerLoop_1_1",
                "InnerLoop_1_2",
                "InnerLoop_1_3",
                "InnerLoop_1_4",
                "InnerLoop_1_5",
                "InnerLoop_1_6",
                "InnerLoop_1_7",
                "InnerLoop_1_8",
                "InnerLoop_1_9",
                "InnerLoop_1_10",
                "InnerLoop_1_11",
                "InnerLoop_1_12",
                "InnerLoop_1_13",
                "InnerLoop_1_14",
                "InnerLoop_1_15",
                "InnerLoop_1_16",
                "InnerLoop_2_1",
                "InnerLoop_2_2",
                "InnerLoop_2_3",
                "InnerLoop_2_4",
                "InnerLoop_2_5",
                "InnerLoop_2_6",
                "InnerLoop_2_7",
                "InnerLoop_3_1",
                "InnerLoop_3_2",
                "InnerLoop_3_3",
                "InnerLoop_3_4",
                "InnerLoop_3_5",
                "InnerLoop_4_1",
                "InnerLoop_4_2"]

        # List of available devices
        self.AVAILABLE_DEVICES = ["xczu7ev-ffvc1156-2-e", "xcu200-fsgd2104-2-e"]
        
        # List of available target clock periods
        self.AVAILABLE_TARGET_CLOCK_PERIODS = ["10", "5", "3.33"]

    def _get_top_level_function(self, app_name):
        """
        Retrieves the top-level function name from the kernel_info.txt file.

        Args:
            app_name (str): The name of the application.

        Returns:
            str: The top-level function name.
        """
        kernel_info_path = os.path.join(self.DATASET_DIR, app_name, "kernel_info.txt")
        
        with open(kernel_info_path, 'r') as file:
            lines = file.readlines()
        
        # The top-level function is expected to be the first line in the file
        top_level_function = lines[0].strip()
        
        return top_level_function

    def _get_extension(self, app_name):
        """
        Determines the source file extension (.c or .cpp) based on the application's directory.

        Args:
            app_name (str): The name of the application.

        Returns:
            str: The file extension (.c or .cpp).
        """
        app_input_dir = os.path.join(self.DATASET_DIR, app_name)
        
        # List of discarded files that should not be considered for the extension
        discarded_files_list = ["noDirectives.cpp", "harness.c", "support.c", "local_support.c"]
        
        for file_name in os.listdir(app_input_dir):
            if (file_name.endswith(".c") or file_name.endswith(".cpp")) and file_name not in discarded_files_list:
                extension = '.cpp' if '.cpp' in file_name else '.c'
                return extension

    def read_databases(self):
        """
        Reads the SQLite databases for each application, extracts QoR (Quality of Result) metrics,
        and saves them into CSV files.
        """
        # Iterate over each application name
        for app_name in self.APPLICATION_NAMES_LIST:
            counter = 0  # Initialize counter for versioning
            
            # Create an empty DataFrame to store the results          
            df = pd.DataFrame(columns = self.APPLICATION_OUTPUT_DATAFRAME_COLUMN_NAMES)
            
            # Get the top-level function name and source file extension
            top_level_function = self._get_top_level_function(app_name)
            extension = self._get_extension(app_name)

            directives_manipulator = DirectivesManipulator(app_name)
            
            if app_name in ["rodinia-kmeans-0-baseline", "rodinia-kmeans-2-pipeline", "rodinia-kmeans-3-unroll"]:
                pass

            # Iterate through available devices and clock periods
            for device in self.AVAILABLE_DEVICES:
                for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                    db_name = device_period = f"{device}_{clock_period}"
                    file_name = f"{app_name}_{device}_{clock_period}.sqlite"
                    db_path = os.path.join(self.DATABASE_DIR, db_name, file_name)
                    
                    # Load the SQLite database
                    current_db = SqliteDict(db_path)

                    # Iterate through database entries (design configurations and QoR metrics)
                    for directives_list_str, qor_map in current_db.items():
                        # Parse the directives list
                        directives_list = [int(i) for i in directives_list_str.strip('][').split(' ') if i != '']

                        directives_action_point_representation_map = directives_manipulator.get_directives_action_point_representation(directives_list, device_period)
                        
                        # Create version name for the design (top-level function + counter + extension)
                        version_name = f"{top_level_function}_{counter}{extension}"
                        
                        # Extract QoR metrics from the database entry
                        latency_msec = qor_map["latency"]
                        bram_utilization_percentage = qor_map["util_bram"]
                        dsp_utilization_percentage = qor_map["util_dsp"]
                        ff_utilization_percentage = qor_map["util_ff"]
                        lut_utilization_percentage = qor_map["util_lut"]
                        synthesis_time_sec = qor_map["synth_time"]
                        
                        # Create a temporary DataFrame for the current design configuration
                        temp_map = {
                            'Version': version_name,
                            'Device': device,
                            'Clock_Period_nsec': clock_period,
                            'Latency_msec': latency_msec,
                            'Synthesis_Time_sec': synthesis_time_sec,
                            'BRAM_Utilization_percentage': bram_utilization_percentage,
                            'DSP_Utilization_percentage': dsp_utilization_percentage,
                            'LUT_Utilization_percentage': lut_utilization_percentage,
                            'FF_Utilization_percentage': ff_utilization_percentage,
                        }
                        
                        temp_map = temp_map | directives_action_point_representation_map
                        temp_df = pd.DataFrame(temp_map, index=[0])
                        
                        # Append the temporary DataFrame to the main DataFrame
                        df = df.append(temp_df)
                        
                        counter += 1  # Increment the counter for versioning
            
            # Define output CSV path and save the DataFrame to a CSV file
            output_file_path = os.path.join(self.CSVS_DIR, f"{app_name}.csv")
            df.to_csv(output_file_path, index=False)
