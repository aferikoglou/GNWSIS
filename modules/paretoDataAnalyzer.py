import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from modules.directivesImpactAnalyzer import DirectivesImpactAnalyzer

class ParetoDataAnalyzer:
       
    def __init__(self):
        """
        Initializes the ParetoDataAnalyzer object.
        """
        self.PER_APPLICATION_DIR = os.path.join("output", "Per_Application")
        self.PARETO_OUTPUT_DIR = os.path.join("output", "Pareto")
        
        self.BASELINE_IMPLEMENTATIONS_PATH = os.path.join("data", "ApplicationBaselineInformation.csv")
        self.SOURCE_CODE_FEATURE_VECTOR_PATH = os.path.join("data", "SourceCodeFeatureVectorsDB.csv")
        
        # Ensure the output directory exists
        os.makedirs(self.PARETO_OUTPUT_DIR, exist_ok=True)
        
        self.AVAILABLE_DEVICES = ["xczu7ev-ffvc1156-2-e", "xcu200-fsgd2104-2-e"]
        self.AVAILABLE_TARGET_CLOCK_PERIODS = [10.0, 5.0, 3.33]
        self.SOURCE_CODE_FEATURES = [
            "Array_1_1",
            "Array_1_2",
            "Array_2_1",
            "Array_2_2",
            "Array_3_1",
            "Array_3_2",
            "Array_4_1",
            "Array_4_2",
            "Array_5_1",
            "Array_5_2",
            "Array_6_1",
            "Array_6_2",
            "Array_7_1",
            "Array_7_2",
            "Array_8_1",
            "Array_8_2",
            "Array_9_1",
            "Array_9_2",
            "Array_10_1",
            "Array_10_2",
            "Array_11_1",
            "Array_11_2",
            "Array_12_1",
            "Array_12_2",
            "Array_13_1",
            "Array_13_2",
            "Array_14_1",
            "Array_14_2",
            "Array_15_1",
            "Array_15_2",
            "Array_16_1",
            "Array_16_2",
            "Array_17_1",
            "Array_17_2",
            "Array_18_1",
            "Array_18_2",
            "Array_19_1",
            "Array_19_2",
            "Array_20_1",
            "Array_20_2",
            "Array_21_1",
            "Array_21_2",
            "Array_22_1",
            "Array_22_2",
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
            "InnerLoop_4_2",
            "ADD",
            "FADD",
            "SUB",
            "FSUB",
            "MUL",
            "FMUL",
            "DIV",
            "FDIV",
            "LOAD",
            "STORE"]
        
        self.TOTAL_AVAILABLE_RESOURCES = {
            "xczu7ev-ffvc1156-2-e": {
                "BRAMs": 624,
                "DSPs": 1728,
                "FFs": 460800,
                "LUTs": 230400,
                "URAMs": 96
            },
            "xcu200-fsgd2104-2-e": {
                "BRAMs": 4320,
                "DSPs": 6840,
                "FFs": 2364480,
                "LUTs": 1182240,
                "URAMs": 960
            },
        }

        self.SEABORN_PALETTE = "bright"
        
        self.PLOT_OUTLIERS = False
        
        self.standard_scaler = None
        self.kmeans = None
        self.cluster_num = 5

        # Perform aggregation of Pareto frontiers
        self._aggregate_pareto_frontiers()
        
        self._get_latency_resource_efficient_designs()
    
    def _add_speedup_and_resource_metrics(self, df_output):
        """
        Adds speedup and resource utilization metrics to the provided DataFrame by comparing with baseline implementations.

        Parameters:
        df_output (pd.DataFrame): DataFrame containing output data to which speedup and resource metrics will be added.

        Returns:
        pd.DataFrame: DataFrame with added speedup and resource metrics columns.
        """
        # Load baseline implementations for comparison
        df_baseline_implementations = pd.read_csv(self.BASELINE_IMPLEMENTATIONS_PATH)

        # Initialize new resource columns
        for resource_metric_name in ["BRAM", "DSP", "FF", "LUT"]:
            new_column_name = f"{resource_metric_name}s"
            df_output[new_column_name] = 0

        # Perform calculations for each row
        for index, row in df_output.iterrows():
            app_name = row["Application_Name"]
            device_id = row["Device"]
            clock_period_nsec = row["Clock_Period_nsec"]
            design_latency_msec = row["Latency_msec"]

            # Filter baseline data for the current application, device, and clock period
            condition = (
                (df_baseline_implementations["app_name"] == app_name) &
                (df_baseline_implementations["device_id"] == device_id) &
                (df_baseline_implementations["clock_period_nsec"] == clock_period_nsec)
            )
            df_app = df_baseline_implementations[condition]

            # Calculate speedup
            baseline_design_latency_msec = float(df_app["design_latency_msec"])
            speedup_value = baseline_design_latency_msec / design_latency_msec
            df_output.at[index, "Speedup"] = speedup_value

            # Calculate resource utilization for each metric
            for resource_metric_name in ["BRAM", "DSP", "FF", "LUT"]:
                new_column_name = f"{resource_metric_name}s"
                percentage_column_name = f"{resource_metric_name}_Utilization_percentage"
                
                # Calculate and assign the integer resource usage
                available_resources = self.TOTAL_AVAILABLE_RESOURCES[device_id][new_column_name]
                df_output.at[index, new_column_name] = int((row[percentage_column_name] / 100) * available_resources)

        return df_output
    
    def _aggregate_pareto_frontiers(self,):
        """
        Aggregates Pareto frontiers from all applications.

        This method iterates over the per-application directories, reads the Pareto frontier CSVs,
        adds the application name to the dataframe, and appends it to an aggregated dataframe.
        Finally, it saves the aggregated dataframe as a CSV file in the output directory.
        """
        df_output = pd.DataFrame()
        
        # Loop through each application directory
        for app_name in os.listdir(self.PER_APPLICATION_DIR):
            if app_name not in ["rodinia_lud_0_baseline_0", "rodinia_lud_1_tiling_0", "spcl_example_01"]:

                # Construct the path to the application's Pareto frontiers CSV file
                APPLICATION_PARETO_FRONTIERS_CSV = os.path.join(self.PER_APPLICATION_DIR, app_name, "04_pareto_frontiers_k.csv")
                
                # Read the CSV file
                df = pd.read_csv(APPLICATION_PARETO_FRONTIERS_CSV)
                # Add a column with the application name
                df["Application_Name"] = app_name

                # Concatenate the current dataframe with the output dataframe
                df_output = pd.concat([df_output, df], ignore_index=True)

        df_output = self._add_speedup_and_resource_metrics(df_output)

        # Define the output filename and path
        fname = "01_pareto_frontiers.csv"
        fpath = os.path.join(self.PARETO_OUTPUT_DIR, fname)

        # Save the aggregated dataframe to a CSV file
        df_output.to_csv(fpath, index=False)
    
    def _get_latency_resource_efficient_designs(self):
        """
        Extract and save the most resource-efficient and latency-efficient designs for each application.

        This function processes Pareto frontier data for each application, extracts the design with 
        the lowest resource utilization and the design with the lowest latency for each combination of 
        device and clock period. The results are saved into two CSV files: one for resource-efficient 
        designs and another for latency-efficient designs.

        Args:
            None

        Returns:
            None
        """
        
        # Initialize empty DataFrames to store the results for resource and latency-efficient designs
        df_resource_efficient_output = pd.DataFrame()
        df_latency_efficient_output = pd.DataFrame()
        
        # Loop through each application directory
        for app_name in os.listdir(self.PER_APPLICATION_DIR):
            if app_name not in ["rodinia_lud_0_baseline_0", "rodinia_lud_1_tiling_0", "spcl_example_01"]:
            
                # Construct the path to the application's Pareto frontiers CSV file
                APPLICATION_PARETO_FRONTIERS_CSV = os.path.join(self.PER_APPLICATION_DIR, app_name, "04_pareto_frontiers_k.csv")
                
                # Read the Pareto frontiers CSV file
                df = pd.read_csv(APPLICATION_PARETO_FRONTIERS_CSV)
                
                # Add a column with the application name to the DataFrame
                df["Application_Name"] = app_name
            
                # Loop through available devices and clock periods
                for device in self.AVAILABLE_DEVICES:
                    for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                        # Filter the DataFrame for the specific device and clock period
                        current_df = df[(df["Device"] == device) & (df["Clock_Period_nsec"] == clock_period)]
                        
                        # Extract the row with the minimum 'Total_Resources_Utilization_percentage'
                        min_row_resource = current_df.loc[current_df['Total_Resources_Utilization_percentage'].idxmin()]
                        df_resource_efficient_design = min_row_resource.to_frame().T
                        
                        # Extract the row with the minimum 'Latency_msec'
                        min_row_latency = current_df.loc[current_df['Latency_msec'].idxmin()]
                        df_latency_efficient_design = min_row_latency.to_frame().T

                        # Concatenate the resource-efficient and latency-efficient rows into their respective DataFrames
                        df_resource_efficient_output = pd.concat([df_resource_efficient_output, df_resource_efficient_design], ignore_index=True)
                        df_latency_efficient_output = pd.concat([df_latency_efficient_output, df_latency_efficient_design], ignore_index=True)
        
        df_resource_efficient_output = self._add_speedup_and_resource_metrics(df_resource_efficient_output)
        
        # Save the resource-efficient designs to a CSV file
        resource_efficient_fname = "02_resource_efficient_designs.csv"
        resource_efficient_fpath = os.path.join(self.PARETO_OUTPUT_DIR, resource_efficient_fname)
        df_resource_efficient_output.to_csv(resource_efficient_fpath, index=False)
        
        df_latency_efficient_output = self._add_speedup_and_resource_metrics(df_latency_efficient_output)
        
        # Save the latency-efficient designs to a CSV file
        latency_efficient_fname = "03_latency_efficient_designs.csv"
        latency_efficient_fpath = os.path.join(self.PARETO_OUTPUT_DIR, latency_efficient_fname)
        df_latency_efficient_output.to_csv(latency_efficient_fpath, index=False)
    
    def _plot_quality_of_result_metrics_distributions(self, input_fname, output_fname):
        """
        Generate and save distribution plots for various Quality of Result (QoR) metrics.

        This function reads a CSV file containing Pareto frontier data, then creates a grid of box plots 
        to visualize the distributions of key QoR metrics across different clock periods and devices.
        The y-axis is logarithmic for better visualization, and each subplot is tailored with custom labels 
        and legends. The final plot is saved as a PDF file.

        Args:
            input_fname (str): The name of the input CSV file containing the data.
            output_fname (str): The desired output filename for the generated plot (PDF format).

        Returns:
            None
        """
        
        # Read the data from the specified input file
        df = pd.read_csv(os.path.join(self.PARETO_OUTPUT_DIR, input_fname))
        
        if not df.empty:
            # Set the Seaborn theme and color palette
            sns.set_theme()
            sns.set(style="ticks", color_codes=True)
            palette = sns.color_palette(self.SEABORN_PALETTE)

            # Mapping QoR metric names to their y-axis labels for better readability
            qor_metric_name_ylabel_map = {
                "Speedup": "Speedup over Vitis",
                "Synthesis_Time_sec": "Synthesis Time (sec)",
                "BRAMs": "Number of BRAMs",
                "DSPs": "Number of DSPs",
                "FFs": "Number of FFs",
                "LUTs": "Number of LUTs"
            }
            
            # Order of clock periods to ensure consistent x-axis in all subplots
            order = self.AVAILABLE_TARGET_CLOCK_PERIODS

            # Create a grid of subplots with a 2x3 layout for the six QoR metrics
            _, axes = plt.subplots(2, 3, figsize=(14, 10))
            
            # Iterate through the specified QoR metrics to create individual box plots
            for count, qor_metric_name in enumerate(["Speedup", "Synthesis_Time_sec", 
                                                    "BRAMs", "DSPs", 
                                                    "FFs", "LUTs"]):
                
                # Calculate the subplot's row and column index
                i, j = divmod(count, 3)

                # Create a box plot for the current QoR metric
                sns_plt = sns.boxplot(data=df, x="Clock_Period_nsec", y=qor_metric_name, 
                                    hue="Device", palette=palette, width=0.65, 
                                    showfliers=self.PLOT_OUTLIERS, fliersize=1, order=order, 
                                    ax=axes[i, j])

                # Add a horizontal red dashed line at 100% for resource utilization metrics
                if qor_metric_name == "Speedup":
                    axes[i, j].axhline(y=1, linestyle='--', color='r', linewidth=1)

                # Set the y-axis to logarithmic scale for better visualization
                axes[i, j].set_yscale('log')

                # Set the labels for x and y axes
                axes[i, j].set_xlabel("", fontsize=12, weight='bold')
                axes[i, j].set_ylabel(qor_metric_name_ylabel_map[qor_metric_name], fontsize=12, weight='bold')

                # Customize the x-ticks and labels for clock periods (in MHz)
                axes[i, j].set_xticks([0, 1, 2])
                axes[i, j].set_xticklabels(["100MHz", "200MHz", "300MHz"], fontsize=12, weight='bold')

                # Get the legend handles (device names)
                handles, _ = sns_plt.get_legend_handles_labels()

                # Set the legend on the top-right of each subplot
                axes[i, j].legend(handles=handles, labels=["ZCU104", "U200"], title="", loc='upper right')

            # Adjust the layout to prevent overlap between subplots
            plt.tight_layout()
            
            # Save the final figure as a PDF file
            fpath = os.path.join(self.PARETO_OUTPUT_DIR, output_fname)
            plt.savefig(fpath)

            # Close the plot to free up resources
            plt.close()

    def _plot_quality_of_result_metrics_distributions_all_versions(self, input_fname):
        """
        Generate and save box plot distributions for Quality of Result (QoR) metrics across different device versions.

        Reads the data for specified devices and creates a grid of box plots, where each subplot represents 
        a QoR metric such as Speedup, Synthesis Time, or hardware resource utilization (BRAM, DSP, FF, LUT). 
        This function also includes customization options for x and y labels, legends, and a horizontal line 
        for reference on Speedup.

        Args:
            input_fname (str): The filename of the input CSV file containing QoR metrics.
        
        Returns:
            None
        """
        # Load data from the input file
        df = pd.read_csv(os.path.join(self.PARETO_OUTPUT_DIR, input_fname))
        
        # Define the QoR metrics with labels for y-axis customization
        qor_metrics_labels = {
            "Speedup": "Speedup over Vitis",
            "Synthesis_Time_sec": "Synthesis Time (sec)",
            "BRAMs": "Number of BRAMs",
            "DSPs": "Number of DSPs",
            "FFs": "Number of FFs",
            "LUTs": "Number of LUTs"
        }

        device_labels = {
            "xczu7ev-ffvc1156-2-e": "ZCU104",
            "xcu200-fsgd2104-2-e": "U200"
        }

        # Set consistent Seaborn styling
        sns.set_theme(style="ticks", palette=self.SEABORN_PALETTE)
        
        # Plot for each available device
        for device in self.AVAILABLE_DEVICES:
            # Filter data by device
            df_device = df[df["Device"] == device]

            if not df_device.empty:
                # Initialize subplots
                fig, axes = plt.subplots(2, 3, figsize=(14, 10))
                
                # Loop through QoR metrics and create box plots
                for idx, (qor_metric, y_label) in enumerate(qor_metrics_labels.items()):
                    row, col = divmod(idx, 3)
                    ax = axes[row, col]
                    
                    sns.boxplot(
                        data=df_device, x="Clock_Period_nsec", y=qor_metric, hue="Version", 
                        palette=self.SEABORN_PALETTE, width=0.65, showfliers=self.PLOT_OUTLIERS, fliersize=1, 
                        order=self.AVAILABLE_TARGET_CLOCK_PERIODS, ax=ax
                    )

                    # Customizations
                    ax.set_yscale('log')
                    ax.set_ylabel(y_label, fontsize=12, weight='bold')
                    ax.set_xlabel("")
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(["100MHz", "200MHz", "300MHz"], fontsize=12, weight='bold')

                    # Add horizontal line for 'Speedup' metric only
                    if qor_metric == "Speedup":
                        ax.axhline(y=1, linestyle='--', color='r', linewidth=1)

                    # Legend settings
                    handles, _ = ax.get_legend_handles_labels()
                    ax.legend(handles=handles, labels=["ALL", "LE", "RE"], loc='upper right')

                # Title and layout adjustments
                fig.suptitle(f"{device_labels[device]}", weight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
                
                # Save the plot to PDF
                output_path = os.path.join(self.PARETO_OUTPUT_DIR, f"qor_metrics_all_versions_{device_labels[device]}_boxplots.pdf")
                plt.savefig(output_path)
                plt.close(fig)

    def _normalize_source_code_features(self, df):
        """
        Standardize the source code feature columns of the given DataFrame.

        This function applies `StandardScaler` from scikit-learn to standardize 
        the values of the source code features, transforming them to have 
        a mean of 0 and a standard deviation of 1.

        Args:
            df (pd.DataFrame): The input DataFrame containing the source code features.

        Returns:
            pd.DataFrame: The DataFrame with the source code feature columns normalized 
                        using the StandardScaler.
        """
        # Initialize the StandardScaler
        self.standard_scaler = StandardScaler()

        # Apply the scaler to the source code feature columns and update the DataFrame
        df[self.SOURCE_CODE_FEATURES] = self.standard_scaler.fit_transform(df[self.SOURCE_CODE_FEATURES])
        # df[self.SOURCE_CODE_FEATURES] = standard_scaler.inverse_transform(df[self.SOURCE_CODE_FEATURES])

        # Return the normalized DataFrame
        return df

    def _denormalize_source_code_features(self, df):
        """
        Reverse the normalization process applied to the source code features in the DataFrame.

        This method uses the fitted `StandardScaler` to reverse the standardization process, 
        transforming the source code feature columns back to their original scale.

        Args:
            df (pd.DataFrame): The input DataFrame containing normalized source code features.

        Returns:
            pd.DataFrame: The DataFrame with the source code features transformed back to their original scale.
        """
        # Reverse the standardization on the source code feature columns
        df[self.SOURCE_CODE_FEATURES] = self.standard_scaler.inverse_transform(df[self.SOURCE_CODE_FEATURES])

        # Return the denormalized DataFrame
        return df

    def _perform_kmeans_source_code_clustering(self, df):
        """
        Perform K-Means clustering on the source code features of a DataFrame.

        This method clusters the source code features into a specified number of clusters
        using the K-Means algorithm. The cluster assignments are stored in a new 'Cluster' column.

        Args:
            df (pd.DataFrame): The input DataFrame containing source code features.
            cluster_num (int): The number of clusters to form.

        Returns:
            pd.DataFrame: The DataFrame with an additional 'Cluster' column containing
                        the cluster assignment for each row.
        """
        # Extract the relevant columns for clustering
        clustering_data = df[self.SOURCE_CODE_FEATURES]

        # Initialize and configure the KMeans model
        self.kmeans = KMeans(
            n_clusters = self.cluster_num,   # Number of clusters
            init = 'k-means++',              # Method for initialization (k-means++)
            n_init = 10,                     # Number of times the algorithm will run with different centroid seeds
            max_iter = 300,                  # Maximum number of iterations
            tol = 0.0001,                    # Tolerance for convergence
            random_state = 42,               # Seed for random number generation (ensures reproducibility)
            algorithm = 'lloyd'              # Algorithm to use (Lloyd's algorithm)
        )

        # Fit the model and predict the cluster for each data point
        df['Cluster'] = self.kmeans.fit_predict(clustering_data)

        # Return the DataFrame with cluster labels
        return df

    def _get_per_cluster_designs(self, df_cluster_o, fname, output_fname):
        """
        Extracts latency-efficient designs for each cluster and saves them into separate CSV files.

        This function processes the clustering results and, for each cluster, retrieves the corresponding
        latency-efficient designs from a previously generated CSV file. The designs are then saved to separate
        cluster-specific CSV files.

        Args:
            df_cluster_o (pd.DataFrame): A DataFrame containing application names and their corresponding cluster IDs.

        Returns:
            None
        """
        
        # Path to the latency-efficient designs CSV file
        fpath = os.path.join("output", "Pareto", fname)
        df = pd.read_csv(fpath)  # Load the latency-efficient designs

        for cluster in range(self.cluster_num):
            df_output = pd.DataFrame()  # DataFrame to store designs for the current cluster
            
            # Filter the applications belonging to the current cluster
            condition = df_cluster_o["Cluster"] == cluster
            df_temp = df_cluster_o[condition]
            application_names = list(df_temp["Application_Name"])  # Get the application names for this cluster
            
            # Iterate over the application names and gather their corresponding designs
            for app_name in application_names:
                condition = df["Application_Name"] == app_name
                df_app = df[condition]  # Filter designs for the current application
                df_output = pd.concat([df_output, df_app], ignore_index=True)  # Append the results to the cluster output
            
            # Save the latency-efficient designs for the current cluster to a CSV file
            fname = os.path.join(self.PARETO_OUTPUT_DIR, f"{output_fname}_Cluster_{cluster}.csv")
            df_output.to_csv(fname, index=False)  # Save the output to a cluster-specific file

    def _plot_per_cluster_quality_of_result_metrics_distributions(self):
        """
        Generates and saves boxplot PDFs for quality of result (QoR) metrics distributions 
        across different clusters and design types (Resource-efficient, Latency-efficient).

        The method iterates over all clusters for each design type and calls an internal 
        plotting function to create the boxplots based on the input CSV files. 
        The output PDFs are named according to the design type and cluster number.

        Args:
            None
        
        Returns:
            None
        """

        for design_type in ["Resource_efficient", "Latency_efficient"]:
            for cluster in range(self.cluster_num):
                finput = f"{design_type}_Cluster_{cluster}.csv"
                foutput = f"qor_metrics_{design_type}_Cluster_{cluster}_boxplots.pdf"
                self._plot_quality_of_result_metrics_distributions(finput, foutput)

    def _generate_source_code_based_on_source_code_feature_vector(self, source_code_feature_vector):
        """
        Generate source code based on the provided feature vector and write it to a file.

        Args:
            source_code_feature_vector (pd.Series): A feature vector containing array sizes, loop trip counts, and cluster info.
        
        Returns:
            None
        """
        MAX_ARRAY_NUM    = 22
        MAX_OUTER_LOOP   = 26
        MAX_INNER_LOOP_1 = 16
        MAX_INNER_LOOP_2 = 7
        MAX_INNER_LOOP_3 = 5
        MAX_INNER_LOOP_4 = 2
        
        cluster = int(source_code_feature_vector["Cluster"])
        
        # Initialize the lines list to hold all code lines
        lines = []

        # Function signature
        lines.append(f"void cluster_{cluster}_centroid_function() {{\n")
        lines.append("\n")
        
        # Write operation comments based on the operation names and their counts
        operations = ["ADD", "FADD", "SUB", "FSUB", "MUL", "FMUL", "DIV", "FDIV", "LOAD", "STORE"]
        
        # Iterate through each operation and append its count as a comment
        for operation_name in operations:
            operation_count = int(source_code_feature_vector.get(operation_name, 0))
            # lines.append(f"\t// {operation_name}: {operation_count}\n")
        # lines.append(f"\t\n")
        
        # Generate arrays
        for array_num in range(1, MAX_ARRAY_NUM + 1):
            column_name_dim_1 = f"Array_{array_num}_1"
            column_name_dim_2 = f"Array_{array_num}_2"
            
            dim1_size = int(source_code_feature_vector[column_name_dim_1])
            dim2_size = int(source_code_feature_vector[column_name_dim_2])
            
            if dim1_size > 0 and dim2_size > 0:
                lines.append(f"\tfloat A_{array_num}[{dim1_size}][{dim2_size}];\n")
            elif dim1_size > 0:
                lines.append(f"\tfloat A_{array_num}[{dim1_size}];\n")
        
        # Add a new line after array declarations
        lines.append("\n")
        
        # Generate nested loops
        for outer_loop in range(1, MAX_OUTER_LOOP + 1):
            column_name_outer_loop = f"OuterLoop_{outer_loop}"
            outer_loop_tripcount = int(source_code_feature_vector[column_name_outer_loop])
            
            if outer_loop_tripcount > 0:
                lines.append(f"\tfor (int i_{outer_loop} = 0; i_{outer_loop} < {outer_loop_tripcount}; i_{outer_loop}++) {{\n")
                
                for inner_loop_1 in range(1, MAX_INNER_LOOP_1 + 1):
                    column_name_inner_loop_1 = f"InnerLoop_1_{inner_loop_1}"
                    inner_loop_1_tripcount = int(source_code_feature_vector[column_name_inner_loop_1])
                    
                    if inner_loop_1_tripcount > 0:
                        lines.append(f"\t\tfor (int j_{inner_loop_1} = 0; j_{inner_loop_1} < {inner_loop_1_tripcount}; j_{inner_loop_1}++) {{\n")
                        
                        for inner_loop_2 in range(1, MAX_INNER_LOOP_2 + 1):
                            column_name_inner_loop_2 = f"InnerLoop_2_{inner_loop_2}"
                            inner_loop_2_tripcount = int(source_code_feature_vector[column_name_inner_loop_2])
                            
                            if inner_loop_2_tripcount > 0:
                                lines.append(f"\t\t\tfor (int k_{inner_loop_2} = 0; k_{inner_loop_2} < {inner_loop_2_tripcount}; k_{inner_loop_2}++) {{\n")
                                
                                for inner_loop_3 in range(1, MAX_INNER_LOOP_3 + 1):
                                    column_name_inner_loop_3 = f"InnerLoop_3_{inner_loop_3}"
                                    inner_loop_3_tripcount = int(source_code_feature_vector[column_name_inner_loop_3])
                                    
                                    if inner_loop_3_tripcount > 0:
                                        lines.append(f"\t\t\t\tfor (int l_{inner_loop_3} = 0; l_{inner_loop_3} < {inner_loop_3_tripcount}; l_{inner_loop_3}++) {{\n")
                                        
                                        for inner_loop_4 in range(1, MAX_INNER_LOOP_4 + 1):
                                            column_name_inner_loop_4 = f"InnerLoop_4_{inner_loop_4}"
                                            inner_loop_4_tripcount = int(source_code_feature_vector[column_name_inner_loop_4])
                                            
                                            if inner_loop_4_tripcount > 0:
                                                lines.append(f"\t\t\t\t\tfor (int m_{inner_loop_4} = 0; m_{inner_loop_4} < {inner_loop_4_tripcount}; m_{inner_loop_4}++) {{\n")
                                                lines.append("\t\t\t\t\t}\n")
                                        
                                        lines.append("\t\t\t\t}\n")
                                
                                lines.append("\t\t\t}\n")
                        
                        lines.append("\t\t}\n")
                
                lines.append("\t}\n")
        
        # Add the closing brace for the function
        lines.append("}\n")
        
        # Write the generated code to a file
        output_fname = os.path.join(self.PARETO_OUTPUT_DIR, f"cluster_{cluster}_centroid_source_code.cpp")
        with open(output_fname, 'w') as f:
            f.writelines(lines)

    def _get_per_cluster_centroid_source_code(self):
        
        # Retrieve the centroids of the k-means clusters
        centroids = self.kmeans.cluster_centers_
        
        # Create a DataFrame from the centroids and label them with the source code feature names
        centroids_df = pd.DataFrame(centroids, columns=self.SOURCE_CODE_FEATURES)
        
        # Add a column to indicate the cluster number for each centroid
        centroids_df['Cluster'] = range(len(centroids_df))
        
        # Denormalize the centroids to bring the features back to their original scale
        centroids_df = self._denormalize_source_code_features(centroids_df)
        
        fpath = os.path.join(self.PARETO_OUTPUT_DIR, "06_cluster_centroid_source_code_feature_vectors.csv")
        centroids_df.to_csv(fpath, index = False)
              
        for _, row in centroids_df.iterrows():
            self._generate_source_code_based_on_source_code_feature_vector(row)
                
    def analyze(self):
        
        # Generate QoR metric box plots for Pareto front designs
        resource_efficient_input = "01_pareto_frontiers.csv"
        resource_efficient_output = "qor_metrics_pf_boxplots.pdf"
        self._plot_quality_of_result_metrics_distributions(resource_efficient_input, resource_efficient_output)
        
        # Generate QoR metric box plots for resource-efficient designs
        resource_efficient_input = "02_resource_efficient_designs.csv"
        resource_efficient_output = "qor_metrics_resource_efficient_boxplots.pdf"
        self._plot_quality_of_result_metrics_distributions(resource_efficient_input, resource_efficient_output)
        
        # Generate QoR metric box plots for latency-efficient designs
        latency_efficient_input = "03_latency_efficient_designs.csv"
        latency_efficient_output = "qor_metrics_latency_efficient_boxplots.pdf"
        self._plot_quality_of_result_metrics_distributions(latency_efficient_input, latency_efficient_output)

        # Plot Quality of Result metrics for ALL, RE, and LE for UltraScale+ ZCU104 and Alveo U200
        version_fname_map = {
                "ALL": "01_pareto_frontiers.csv",
                "LE": "03_latency_efficient_designs.csv",
                "RE": "02_resource_efficient_designs.csv"
            }
        
        # Load and label dataframes
        df_list = []
        for version, fname in version_fname_map.items():
            df = pd.read_csv(os.path.join(self.PARETO_OUTPUT_DIR, fname))
            df["Version"] = version
            df_list.append(df)

        # Concatenate dataframes and save the combined version
        df_combined = pd.concat(df_list, ignore_index=True)
        combined_output_path = os.path.join(self.PARETO_OUTPUT_DIR, "04_ALL_LE_RE_designs.csv")
        df_combined.to_csv(combined_output_path, index=False)

        self._plot_quality_of_result_metrics_distributions_all_versions("04_ALL_LE_RE_designs.csv")
        
        # Load the source code feature vectors
        df = pd.read_csv(self.SOURCE_CODE_FEATURE_VECTOR_PATH)

        # Normalize the source code features
        df_norm = self._normalize_source_code_features(df)

        # Perform K-Means clustering on the normalized source code features
        df_norm_cluster = self._perform_kmeans_source_code_clustering(df_norm)

        # Optional: Denormalize the clustered data
        # df_cluster = self._denormalize_source_code_features(df)

        # Save only the application names and their corresponding clusters to a CSV file
        df_cluster_o = df_norm_cluster[["Application_Name", "Cluster"]]
        df_cluster_o = df_cluster_o.sort_values(by = 'Cluster')
        df_cluster_o.to_csv(os.path.join(self.PARETO_OUTPUT_DIR, "05_clustered_applications.csv"), index=False)
        
        # Obtain per-cluster designs based on the cluster dataframe (df_cluster_o)
        # and save the resulting data for resource-efficient designs
        self._get_per_cluster_designs(df_cluster_o, "02_resource_efficient_designs.csv", "Resource_efficient")

        # Obtain per-cluster designs based on the cluster dataframe (df_cluster_o)
        # and save the resulting data for latency-efficient designs
        self._get_per_cluster_designs(df_cluster_o, "03_latency_efficient_designs.csv", "Latency_efficient")

        # Plot and save the boxplots for Quality of Result (QoR) metrics distributions 
        # for each cluster and design type (resource-efficient, latency-efficient)
        self._plot_per_cluster_quality_of_result_metrics_distributions()
        
        # Get the centroid source code per cluster
        self._get_per_cluster_centroid_source_code()
        
        # Analyze the impact of directives for each cluster       
        directive_impact_analyzer = DirectivesImpactAnalyzer(self.PARETO_OUTPUT_DIR, self.cluster_num)
        directive_impact_analyzer.analyze()