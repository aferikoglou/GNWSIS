import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal

class DirectivesImpactAnalyzer:
    def __init__(self, PARETO_OUTPUT_DIR, cluster_num):
        """
        Initialize the DirectivesImpactAnalyzer class.

        Parameters:
        PARETO_OUTPUT_DIR (str): Path to the directory containing the dataset.
        cluster_num (int): Number of clusters for analysis.
        """
        self.PARETO_OUTPUT_DIR = PARETO_OUTPUT_DIR
        self.OUTPUT_DIR = os.path.join(PARETO_OUTPUT_DIR, "Directives_Impact")
        self.CLUSTER_NUM = cluster_num
        
        # Ensure the output directory exists
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Define metrics for analysis
        self.QUALITY_OF_RESULT_METRICS = [
            "Latency_msec", "Synthesis_Time_sec", "BRAM_Utilization_percentage", 
            "DSP_Utilization_percentage", "FF_Utilization_percentage", "LUT_Utilization_percentage"
        ]
        
        self.QUALITY_OF_RESULT_METRICS_V2 = [ "Speedup", "Synthesis_Time_sec", "BRAMs", "DSPs", "FFs", "LUTs" ]

        # Define available design types
        self.DESIGN_TYPES = ["Latency_efficient", "Resource_efficient"]
        
        self.DESIGN_TYPES_INPUT_FILE_PATH_MAP = {
            "ALL": "01_pareto_frontiers.csv",
            "RE": "02_resource_efficient_designs.csv",
            "LE": "03_latency_efficient_designs.csv"
        }

        # Define available devices and target clock periods for analysis
        self.AVAILABLE_DEVICES = ["xczu7ev-ffvc1156-2-e", "xcu200-fsgd2104-2-e"]
        self.AVAILABLE_TARGET_CLOCK_PERIODS = [10.0, 5.0, 3.33]

        # Define source code features (action points) to be analyzed
        self.SOURCE_CODE_FEATURES = [
            "Array_1", "Array_2", "Array_3", "Array_4", "Array_5", "Array_6", 
            "Array_7", "Array_8", "Array_9", "Array_10", "Array_11", "Array_12", 
            "Array_13", "Array_14", "Array_15", "Array_16", "Array_17", "Array_18", 
            "Array_19", "Array_20", "Array_21", "Array_22", "OuterLoop_1", "OuterLoop_2", 
            "OuterLoop_3", "OuterLoop_4", "OuterLoop_5", "OuterLoop_6", "OuterLoop_7", 
            "OuterLoop_8", "OuterLoop_9", "OuterLoop_10", "OuterLoop_11", "OuterLoop_12", 
            "OuterLoop_13", "OuterLoop_14", "OuterLoop_15", "OuterLoop_16", "OuterLoop_17", 
            "OuterLoop_18", "OuterLoop_19", "OuterLoop_20", "OuterLoop_21", "OuterLoop_22", 
            "OuterLoop_23", "OuterLoop_24", "OuterLoop_25", "OuterLoop_26", "InnerLoop_1_1", 
            "InnerLoop_1_2", "InnerLoop_1_3", "InnerLoop_1_4", "InnerLoop_1_5", "InnerLoop_1_6", 
            "InnerLoop_1_7", "InnerLoop_1_8", "InnerLoop_1_9", "InnerLoop_1_10", "InnerLoop_1_11", 
            "InnerLoop_1_12", "InnerLoop_1_13", "InnerLoop_1_14", "InnerLoop_1_15", "InnerLoop_1_16", 
            "InnerLoop_2_1", "InnerLoop_2_2", "InnerLoop_2_3", "InnerLoop_2_4", "InnerLoop_2_5", 
            "InnerLoop_2_6", "InnerLoop_2_7", "InnerLoop_3_1", "InnerLoop_3_2", "InnerLoop_3_3", 
            "InnerLoop_3_4", "InnerLoop_3_5", "InnerLoop_4_1", "InnerLoop_4_2"
        ]

        self.ALL_DIRECTIVES_ORDERED = [
            "block_2_1",
            "block_2_2",
            "block_4_1",
            "block_4_2",
            "block_8_1",
            "block_8_2",
            "block_16_1",
            "block_32_1",
            "block_64_1",
            "block_128_1",
            "block_256_1",
            "block_512_1",
            "block_1024_1",
            "cyclic_2_1",
            "cyclic_2_2",
            "cyclic_4_1",
            "cyclic_4_2",
            "cyclic_8_1",
            "cyclic_8_2",
            "cyclic_16_1",
            "cyclic_16_2",
            "cyclic_32_1",
            "cyclic_64_1",
            "cyclic_128_1",
            "cyclic_256_1",
            "cyclic_512_1",
            "cyclic_1024_1",
            "complete_1",
            "complete_2",
            "pipeline",
            "pipeline_1",
            "unroll_2",
            "unroll_4",
            "unroll_8",
            "unroll_16",
            "unroll_32",
            "unroll_64",
            "unroll"
        ]
        
        self.ALL_DIRECTIVES_TRANSLATED_MAP = {
            "block_2_1": "Block Fac. 2 (1)",
            "block_2_2": "Block Fac. 2 (2)",
            "block_4_1": "Block Fac. 4 (1)",
            "block_4_2": "Block Fac. 4 (2)",
            "block_8_1": "Block Fac. 8 (1)",
            "block_8_2": "Block Fac. 8 (2)",
            "block_16_1": "Block Fac. 16 (1)",
            "block_32_1": "Block Fac. 32 (1)",
            "block_64_1": "Block Fac. 64 (1)",
            "block_128_1": "Block Fac. 128 (1)",
            "block_256_1": "Block Fac. 256 (1)",
            "block_512_1": "Block Fac. 512 (1)",
            "block_1024_1": "Block Fac. 1024 (1)",
            "complete_1": "Complete (1)",
            "complete_2": "Complete (2)",
            "cyclic_2_1": "Cyclic Fac. 2 (1)",
            "cyclic_2_2": "Cyclic Fac. 2 (2)",
            "cyclic_4_1": "Cyclic Fac. 4 (1)",
            "cyclic_4_2": "Cyclic Fac. 4 (2)",
            "cyclic_8_1": "Cyclic Fac. 8 (1)",
            "cyclic_8_2": "Cyclic Fac. 8 (2)",
            "cyclic_16_1": "Cyclic Fac. 16 (1)",
            "cyclic_16_2": "Cyclic Fac. 16 (2)",
            "cyclic_32_1": "Cyclic Fac. 32 (1)",
            "cyclic_64_1": "Cyclic Fac. 64 (1)",
            "cyclic_128_1": "Cyclic Fac. 128 (1)",
            "cyclic_256_1": "Cyclic Fac. 256 (1)",
            "cyclic_512_1": "Cyclic Fac. 512 (1)",
            "cyclic_1024_1": "Cyclic Fac. 1024 (1)",
            "pipeline": "Pipeline",
            "pipeline_1": "Pipeline II=1",
            "unroll_2": "Unroll Fac. 2",
            "unroll_4": "Unroll Fac. 4",
            "unroll_8": "Unroll Fac. 8",
            "unroll_16": "Unroll Fac. 16",
            "unroll_32": "Unroll Fac. 32",
            "unroll_64": "Unroll Fac. 64",
            "unroll": "Unroll"
        }

    def _normalize_design_latency(self, df):
        """
        Normalize the 'Latency_msec' column in the DataFrame using StandardScaler.

        Parameters:
        df (DataFrame): DataFrame containing the latency data.

        Returns:
        DataFrame: DataFrame with normalized latency values.
        """
        scaler = StandardScaler()
        df["Latency_msec"] = scaler.fit_transform(df[["Latency_msec"]])
        return df

    def _perform_kruskal_wallis_test(self, distribution_ON, distribution_OFF):
        """
        Perform the Kruskal-Wallis test to compare two distributions.

        Parameters:
        distribution_ON (list): List of values when the directive is ON.
        distribution_OFF (list): List of values when the directive is OFF.

        Returns:
        float: P-value from the Kruskal-Wallis test. If non-significant, return 1.0.
        """
        _, p_value = kruskal(distribution_ON, distribution_OFF)
        return p_value if p_value < 0.05 else 1.0

    def _get_per_action_point_highest_impact_directive_map(self, df, qor_metric):
        """
        Analyze and find the directive with the highest impact for each action point.

        Parameters:
        df (DataFrame): Filtered DataFrame containing relevant rows for analysis.
        qor_metric (str): Quality of result (QoR) metric to analyze.

        Returns:
        dict: Mapping of action points to the directive with the highest impact on the QoR metric.
        """
        action_point_directive_map = {}

        for source_code_feature in self.SOURCE_CODE_FEATURES:
            unique_directives = df[source_code_feature].unique()
            p_value_min = 1.0
            directive_with_highest_impact = "NDIR"  # No Directive (default)

            for directive in unique_directives:
                # Split DataFrame into rows where the directive is ON or OFF
                df_directive_ON = df[df[source_code_feature] == directive]
                df_directive_OFF = df[df[source_code_feature] != directive]

                distribution_ON = list(df_directive_ON[qor_metric])
                distribution_OFF = list(df_directive_OFF[qor_metric])

                if len(distribution_ON) > 1 and len(distribution_OFF) > 1:
                    # Perform Kruskal-Wallis test
                    try:
                        p_value = self._perform_kruskal_wallis_test(distribution_ON, distribution_OFF)
                    except:
                        continue

                    # Calculate average values for ON and OFF distributions
                    avg_ON = sum(distribution_ON) / len(distribution_ON)
                    avg_OFF = sum(distribution_OFF) / len(distribution_OFF)

                    # Check if the directive has a significant impact
                    if p_value < p_value_min:
                        if (qor_metric == "Latency_msec" and avg_ON < avg_OFF) or (qor_metric != "Latency_msec" and avg_ON >= avg_OFF):
                            p_value_min = p_value
                            directive_with_highest_impact = directive

            if directive_with_highest_impact != "NDIR":
                action_point_directive_map[source_code_feature] = directive_with_highest_impact

        return action_point_directive_map

    def _get_per_action_point_high_impact_directives_map(self, df, qor_metric):
        """
        Identify directives with the highest impact for each action point based on a specified QoR metric.

        This method analyzes each action point to determine which directives have a significant impact 
        on the given QoR metric. Directives are evaluated for their impact using a Kruskal-Wallis test 
        and are included in the results if:
        - The p-value of the test is below 0.05 (significant).
        - The average QoR metric when the directive is ON is greater than or equal to when it is OFF.

        For each significant directive, its p-value and the frequency of occurrence are recorded.

        Parameters:
        df (DataFrame): DataFrame containing the dataset with action points and QoR metric data.
        qor_metric (str): The Quality of Result (QoR) metric to analyze (e.g., Latency, Resource Utilization).

        Returns:
        dict: A mapping of each action point to a dictionary of significant directives with their p-values 
        and frequencies.
        """
        action_point_directive_map = {}

        for source_code_feature in self.SOURCE_CODE_FEATURES:
            # Filter out rows where the directive is "NDIR" (No Directive)
            df_action_point_wo_NDIR = df[df[source_code_feature] != "NDIR"]
            
            # Get the unique directives for the current action point
            unique_directives = df_action_point_wo_NDIR[source_code_feature].unique()

            high_impact_directives_p_value_freq = {}
            for directive in unique_directives:
                # Split DataFrame into rows where the directive is ON or OFF
                df_directive_ON = df_action_point_wo_NDIR[df_action_point_wo_NDIR[source_code_feature] == directive]
                df_directive_OFF = df_action_point_wo_NDIR[df_action_point_wo_NDIR[source_code_feature] != directive]

                # Extract QoR metric distributions for ON and OFF cases
                distribution_ON = list(df_directive_ON[qor_metric])
                distribution_OFF = list(df_directive_OFF[qor_metric])

                # Count occurrences of ON and OFF
                count_ON = len(distribution_ON)
                count_OFF = len(distribution_OFF)

                # Calculate the frequency of the directive being ON
                frequency = float(count_ON) / (count_ON + count_OFF)

                # Perform analysis only if both distributions have more than one data point
                if len(distribution_ON) > 1 and len(distribution_OFF) > 1:
                    try:
                        # Perform Kruskal-Wallis test
                        p_value = self._perform_kruskal_wallis_test(distribution_ON, distribution_OFF)
                    except Exception as e:
                        print(f"Error performing Kruskal-Wallis test: {e}")
                        continue

                    # Calculate average QoR metric values for ON and OFF cases
                    avg_ON = sum(distribution_ON) / len(distribution_ON)
                    avg_OFF = sum(distribution_OFF) / len(distribution_OFF)

                    # Check if the directive has a significant impact based on p-value and QoR metric average
                    if p_value < 0.05 and avg_ON >= avg_OFF:
                        high_impact_directives_p_value_freq[directive] = {
                            "p_value": p_value,
                            "frequency": frequency
                        }

            # Store the significant directives for the current action point
            if high_impact_directives_p_value_freq != {}:
                action_point_directive_map[source_code_feature] = high_impact_directives_p_value_freq

        return action_point_directive_map

    def _per_cluster_directives_impact_analysis(self):
        """
        Perform the impact analysis of directives for each cluster and save results as JSON.

        This function:
        - Iterates through each design type and cluster.
        - Normalizes latency values.
        - Analyzes each device, clock period, and QoR metric to identify directives with the highest impact.
        - Saves the results to a JSON file for each cluster.

        Output:
        JSON files with results are saved in the output directory.
        """
        for design_type in self.DESIGN_TYPES:
            for cluster in range(self.CLUSTER_NUM):
                input_fpath = os.path.join(self.PARETO_OUTPUT_DIR, f"{design_type}_Cluster_{cluster}.csv")
                
                # Load the dataset
                df = pd.read_csv(input_fpath)
                
                if not df.empty:

                    # Filter out rows with unrealistic latency values
                    df = df[df["Latency_msec"] != 1000000.0]
                    
                    # Normalize the design latency
                    df = self._normalize_design_latency(df)

                    device_clock_period_qor_metric_action_point_directive_map = {}

                    for device in self.AVAILABLE_DEVICES:
                        clock_period_qor_metric_action_point_directive_map = {}
                        
                        for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                            # Filter the DataFrame for the current device and clock period
                            current_df = df[(df["Device"] == device) & (df["Clock_Period_nsec"] == clock_period)]
                            
                            qor_metric_action_point_directive_map = {}

                            for qor_metric in self.QUALITY_OF_RESULT_METRICS:   
                                # Identify directives with the highest impact
                                action_point_directive_map = self._get_per_action_point_highest_impact_directive_map(current_df, qor_metric)
                                qor_metric_action_point_directive_map[qor_metric] = action_point_directive_map
                            
                            clock_period_qor_metric_action_point_directive_map[clock_period] = qor_metric_action_point_directive_map
                        
                        device_clock_period_qor_metric_action_point_directive_map[device] = clock_period_qor_metric_action_point_directive_map

                    # Save results to a JSON file
                    output_file_path = os.path.join(self.OUTPUT_DIR, f"{design_type}_Cluster_{cluster}_Directives_w_Highest_Impact.json")
                    with open(output_file_path, "w") as file:
                        json.dump(device_clock_period_qor_metric_action_point_directive_map, file, indent=4)

    def _directives_impact_analysis(self,):
        
        for design_type in ["ALL", "LE", "RE"]:
            fname = self.DESIGN_TYPES_INPUT_FILE_PATH_MAP[design_type]
            input_fpath = os.path.join(self.PARETO_OUTPUT_DIR, fname)
                    
            # Load the dataset
            df = pd.read_csv(input_fpath)
                    
            if not df.empty:
                # Filter out rows with unrealistic latency values
                df = df[df["Latency_msec"] != 1000000.0]
                        
                device_clock_period_qor_metric_action_point_directive_map = {}

                for device in self.AVAILABLE_DEVICES:
                    clock_period_qor_metric_action_point_directive_map = {}
                            
                    for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                        # Filter the DataFrame for the current device and clock period
                        current_df = df[(df["Device"] == device) & (df["Clock_Period_nsec"] == clock_period)]
                                
                        qor_metric_action_point_directive_map = {}

                        for qor_metric in self.QUALITY_OF_RESULT_METRICS_V2:   
                            # Identify directives with the highest impact
                            action_point_directive_map = self._get_per_action_point_high_impact_directives_map(current_df, qor_metric)
                            qor_metric_action_point_directive_map[qor_metric] = action_point_directive_map
                                
                        clock_period_qor_metric_action_point_directive_map[clock_period] = qor_metric_action_point_directive_map
                            
                    device_clock_period_qor_metric_action_point_directive_map[device] = clock_period_qor_metric_action_point_directive_map

                # Save results to a JSON file
                output_file_path = os.path.join(self.OUTPUT_DIR, f"{design_type}_Directives_w_Highest_Impact.json")
                with open(output_file_path, "w") as file:
                    json.dump(device_clock_period_qor_metric_action_point_directive_map, file, indent=4)

    def _create_p_value_freq_dataframes(self, design_type, device_id, clock_period_nsec, qor_metric):
        """
        Create DataFrames for p-values and frequencies of directives' impact on an action point.

        Parameters:
        - design_type (str): The design type, e.g., "Latency_efficient" or "Resource_efficient".
        - device_id (str): The identifier for the device being analyzed.
        - clock_period_nsec (str): The target clock period in nanoseconds.
        - qor_metric (str): The quality of result (QoR) metric to analyze.

        Returns:
        - tuple: Two pandas DataFrames:
            - df_pvalue: Contains p-values for each directive's impact on the action points.
            - df_frequency: Contains frequencies for each directive's impact on the action points.
        """
        input_fpath = os.path.join(self.OUTPUT_DIR, f"{design_type}_Directives_w_Highest_Impact.json")

        # Load the JSON data
        with open(input_fpath, "r") as f:
            data = json.load(f)

        data_device_clock_period_qor_metric = data[device_id][clock_period_nsec][qor_metric]
        columns = ["action_point"] + self.ALL_DIRECTIVES_ORDERED
        df_pvalue = pd.DataFrame(columns=columns)
        df_frequency = pd.DataFrame(columns=columns)

        for action_point in self.SOURCE_CODE_FEATURES:
            action_point_directives_pvalue_map = {}
            action_point_directives_frequency_map = {}

            for directive in self.ALL_DIRECTIVES_ORDERED:
                # Retrieve p-value and frequency or set defaults
                directive_data = data_device_clock_period_qor_metric.get(action_point, {}).get(directive, {})
                p_value = directive_data.get("p_value", 1)
                frequency = directive_data.get("frequency", 0)

                action_point_directives_pvalue_map[directive] = p_value
                action_point_directives_frequency_map[directive] = frequency

            # Add action point to the maps
            action_point_directives_pvalue_map["action_point"] = action_point
            action_point_directives_frequency_map["action_point"] = action_point

            # Append the maps to their respective DataFrames
            df_pvalue = pd.concat([df_pvalue, pd.DataFrame([action_point_directives_pvalue_map])], ignore_index=True)
            df_frequency = pd.concat([df_frequency, pd.DataFrame([action_point_directives_frequency_map])], ignore_index=True)

        return df_pvalue, df_frequency

    def _plot_heatmaps(self, df_pvalue, df_frequency, output_fpath):
        """
        Plot two heatmaps side by side: one for p-values and another for frequencies.

        This method generates and saves two heatmaps:
        - The first heatmap visualizes the p-values for each directive and action point.
        - The second heatmap visualizes the frequency of each directive's impact.

        Parameters:
        - df_pvalue (DataFrame): DataFrame containing p-value data for the heatmap.
        - df_frequency (DataFrame): DataFrame containing frequency data for the heatmap.
        - output_fpath (str): File path to save the generated heatmap image.

        Steps:
        1. Filter rows and columns to remove irrelevant data (e.g., rows/columns of all 1.0 or 0.0).
        2. Highlight specific cells in the heatmaps (e.g., p-value of 1.0 and frequency of 0.0).
        3. Plot two heatmaps side by side using seaborn with customized color palettes and axes labels.
        4. Save the resulting heatmap as a PDF file.
        """

        def prepare_numeric_df(df, default_value):
            """
            Convert DataFrame to numeric, replacing non-numeric values with a default.

            Parameters:
            - df (DataFrame): Input DataFrame.
            - default_value (float): Value to replace NaNs or invalid entries.

            Returns:
            - DataFrame with numeric values and specified default replacement for NaNs.
            """
            return df.drop(columns=["action_point"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(default_value)

        def add_highlight_patches(ax, df_numeric, highlight_value, color="white"):
            """
            Highlight specific cells in the heatmap with a rectangle.

            Parameters:
            - ax (Axes): Matplotlib Axes object for the heatmap.
            - df_numeric (DataFrame): Numeric DataFrame corresponding to the heatmap.
            - highlight_value (float): Value to highlight in the heatmap.
            - color (str): Color of the highlight rectangle.
            """
            coordinates = np.column_stack(np.where(df_numeric.values == highlight_value))
            for y, x in coordinates:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", lw=0.5))
                
        def add_patches(ax, action_points, directives, df, color="lightgrey"):
            """
            Add grey rectangles to heatmap squares corresponding to combinations of action_points and directives.

            Parameters:
            - ax (Axes): Matplotlib Axes object for the heatmap.
            - action_points (list): List of action points to highlight.
            - directives (list): List of directives to highlight.
            - df (DataFrame): DataFrame used to generate the heatmap.
            - color (str): Color of the rectangle (default is grey).
            """
            # Iterate over all combinations of action_points and directives
            for action_point in action_points:
                for directive in directives:
                    if action_point in df.index and directive in df.columns:
                        y = df.index.get_loc(action_point)  # Get the row index
                        x = df.columns.get_loc(directive)  # Get the column index
                        # Add a rectangle patch to the heatmap
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", lw=1))


        # Set the action_point column as the index for better visualization
        df_pvalue.set_index('action_point', inplace=True)
        df_frequency.set_index('action_point', inplace=True)

        # Filter out rows and columns with uniform values (1.0 for p-value, 0.0 for frequency)
        df_pvalue = df_pvalue[(df_pvalue != 1.0).any(axis=1)]
        df_frequency = df_frequency[(df_frequency != 0.0).any(axis=1)]

        df_pvalue = df_pvalue.loc[:, (df_pvalue != 1.0).any(axis=0)]
        df_frequency = df_frequency.loc[:, (df_frequency != 0.0).any(axis=0)]

        # Get remaining action points and directives after filtering
        action_point_final = df_pvalue.index.tolist()
        directives_final = df_pvalue.columns.tolist()
        
        # Separate action points into array-based and loop-based categories
        array_action_points = [ap for ap in action_point_final if "Array" in ap]
        loop_action_points = [ap for ap in action_point_final if "Array" not in ap]

        # Separate directives into array-based and loop-based categories
        array_directives = [d for d in directives_final if any(keyword in d for keyword in ["block", "cyclic", "complete"])]
        loop_directives = [d for d in directives_final if all(keyword not in d for keyword in ["block", "cyclic", "complete"])]

        # Translate directive labels for the x-axis
        new_labels_x = [self.ALL_DIRECTIVES_TRANSLATED_MAP[x] for x in directives_final]

        # Prepare numeric DataFrames for the heatmaps
        df_pvalue_numeric = prepare_numeric_df(df_pvalue, default_value=1)
        df_frequency_numeric = prepare_numeric_df(df_frequency, default_value=0)

        # Create subplots for the two heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # 1 row, 2 columns

        sns.set_theme()
        sns.set(style="ticks", color_codes=True)

        # Plot heatmap for p-values
        ax1 = sns.heatmap(
            df_pvalue_numeric,
            xticklabels=new_labels_x,
            yticklabels=action_point_final,
            linecolor='black',
            linewidths=0.5,
            cmap=sns.color_palette("viridis_r", as_cmap=True),
            vmin=0.0,
            vmax=0.05,
            ax=axes[0]
        )
        ax1.set_title("P-Value", fontsize=12, weight='bold')
        ax1.set_xlabel("", fontsize=12, weight='bold')
        ax1.set_ylabel("", fontsize=12, weight='bold')
        ax1.set_xticklabels(new_labels_x, fontsize=9, rotation=90, weight='bold')
        ax1.set_yticklabels(action_point_final, fontsize=9, rotation=0, weight='bold')
        add_highlight_patches(ax1, df_pvalue_numeric, highlight_value=1.0)

        # Plot heatmap for frequencies
        ax2 = sns.heatmap(
            df_frequency_numeric,
            xticklabels=new_labels_x,
            yticklabels=False,  # Omit y-axis labels for clarity
            linecolor='black',
            linewidths=0.5,
            cmap=sns.color_palette("magma", as_cmap=True),
            vmin=0.0,
            ax=axes[1]
        )
        ax2.set_title("Frequency", fontsize=12, weight='bold')
        ax2.set_xlabel("", fontsize=12, weight='bold')
        ax2.set_ylabel("")  # No y-label for the second heatmap
        ax2.set_xticklabels(new_labels_x, fontsize=9, rotation=90, weight='bold')
        add_highlight_patches(ax2, df_frequency_numeric, highlight_value=0.0)

        add_patches(ax1, array_action_points, loop_directives, df_pvalue_numeric)
        add_patches(ax1, loop_action_points, array_directives, df_pvalue_numeric)
        add_patches(ax2, array_action_points, loop_directives, df_frequency_numeric)
        add_patches(ax2, loop_action_points, array_directives, df_frequency_numeric)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(output_fpath)
        plt.close()

    def analyze(self):
        """
        Perform the full analysis of directives' impact on design quality metrics.

        This method serves as the entry point for the analysis process. It coordinates the evaluation 
        of the impact of directives (source code features) on Quality of Result (QoR) metrics for various 
        design configurations. The results are saved as JSON files and visualized as heatmaps.

        The analysis includes:
        1. Extracting and normalizing relevant data for each design type and cluster.
        2. Filtering data based on specific devices and clock periods.
        3. Evaluating the impact of directives on QoR metrics (e.g., Speedup, Area, etc.).
        4. Identifying and ranking directives with the highest impact on QoR metrics.
        5. Creating and saving visualizations (heatmaps) of the results.

        The `_directives_impact_analysis` method handles the core computation, while heatmaps 
        are generated using `_create_p_value_freq_dataframes` and `_plot_heatmaps`.

        Outputs:
        - JSON files summarizing the results in the `Directives_Impact` folder.
        - PDF heatmaps for visualizing p-value and frequency distributions.

        Parameters:
        None

        Returns:
        None
        """
        # Perform core directives impact analysis
        self._directives_impact_analysis()

        # Specify analysis parameters
        for design_type in ["ALL"]:
            for device_id in  self.AVAILABLE_DEVICES:
                for clock_period_nsec in  self.AVAILABLE_TARGET_CLOCK_PERIODS:
                    for qor_metric in self.QUALITY_OF_RESULT_METRICS_V2:
                        clock_period_nsec_str = str(clock_period_nsec)
                    
                        # Create DataFrames for p-value and frequency from analysis results
                        df_pvalue, df_frequency = self._create_p_value_freq_dataframes(
                            design_type, device_id, clock_period_nsec_str, qor_metric
                        )

                        # Define output file name for the heatmaps
                        output_fpath = os.path.join(self.OUTPUT_DIR, f"Directives_Impact_{design_type}_{device_id}_{clock_period_nsec}_{qor_metric}.pdf")

                        # Generate and save heatmaps for p-value and frequency
                        self._plot_heatmaps(df_pvalue, df_frequency, output_fpath)
