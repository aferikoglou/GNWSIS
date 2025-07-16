import os

import pandas as pd
pd.options.mode.chained_assignment = None

import seaborn as sns
import matplotlib.pyplot as plt

from paretoset import paretoset

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

class ApplicationDataAnalyzer():
    def __init__(self, application_name, df_app):
        self.APPLICATION_NAME = application_name
        self.df = df_app

        self.APPLICATION_OUTPUT_DIR = os.path.join("output", self.APPLICATION_NAME)
        
        os.makedirs(self.APPLICATION_OUTPUT_DIR, exist_ok = True)
        
        self.AVAILABLE_DEVICES = ["xczu7ev-ffvc1156-2-e", "xcu200-fsgd2104-2-e"]
        self.AVAILABLE_TARGET_CLOCK_PERIODS = [10.0, 5.0, 3.33]
        
        self.SEABORN_PALETTE = "viridis"
        
        self.AVAILABLE_DEVICES_MAP = {"xczu7ev-ffvc1156-2-e": "ZCU104", "xcu200-fsgd2104-2-e": "U200"}
        self.AVAILABLE_TARGET_CLOCK_PERIODS_MAP = {10.0: "100MHz", 5.0: "200MHz", 3.33: "300MHz"}

    def _get_synthesible_feasible_designs_and_statistics(self, df):
        """
        Computes and exports synthesizability and feasibility statistics for different devices and clock periods.

        This function evaluates designs for synthesizability and feasibility based on various metrics (latency, BRAM,
        DSP, FF, and LUT utilization). It produces a summary of total designs, synthesizable designs, and feasible designs
        for each device and clock period combination, and also exports the synthesizable and feasible designs.

        Args:
            df (pandas.DataFrame): Input dataframe containing design metrics for various devices and clock periods.

        Outputs:
            - Exports synthesizability and feasibility statistics to "01_synthesizability_feasibility_statistics.csv".
            - Exports the synthesizable and feasible designs to "02_synthesizable_feasible_designs.csv".
        """

        # Initialize DataFrames for storing statistics and synthesizable-feasible designs
        df_stats = pd.DataFrame(columns=["Device", "Clock_Period_nsec", "Total", "Synthesizable", "Not_Synthesizable", "Feasible", "Not_Feasible"])
        df_synth_feasible_designs = pd.DataFrame()

        # Loop through available devices and clock periods
        for device in self.AVAILABLE_DEVICES:
            for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                # Filter the input dataframe for the specific device and clock period
                current_df = df[(df["Device"] == device) & (df["Clock_Period_nsec"] == clock_period)]

                # Compute total number of designs
                total = len(current_df)

                # Condition for synthesizability: Latency >= 0, Utilization < 101%
                synth_condition = (current_df["Latency_msec"] >= 0) & \
                                (current_df["BRAM_Utilization_percentage"] != 101) & \
                                (current_df["DSP_Utilization_percentage"] != 101) & \
                                (current_df["FF_Utilization_percentage"] != 101) & \
                                (current_df["LUT_Utilization_percentage"] != 101)

                synthesizable_df = current_df[synth_condition]
                synth_count = len(synthesizable_df)
                not_synth_count = total - synth_count

                # Condition for feasibility: Utilization <= 100%
                feasible_condition = (synthesizable_df["Latency_msec"] >= 0) & \
                                    (synthesizable_df["BRAM_Utilization_percentage"] <= 100) & \
                                    (synthesizable_df["DSP_Utilization_percentage"] <= 100) & \
                                    (synthesizable_df["FF_Utilization_percentage"] <= 100) & \
                                    (synthesizable_df["LUT_Utilization_percentage"] <= 100)

                feasible_df = synthesizable_df[feasible_condition]
                feasible_count = len(feasible_df)
                not_feasible_count = synth_count - feasible_count

                # Collect statistics in a dictionary
                stats = {
                    "Device": device,
                    "Clock_Period_nsec": clock_period,
                    "Total": total,
                    "Synthesizable": synth_count,
                    "Not_Synthesizable": not_synth_count,
                    "Feasible": feasible_count,
                    "Not_Feasible": not_feasible_count
                }

                # Append statistics and synthesizable-feasible designs
                df_stats = pd.concat([df_stats, pd.DataFrame([stats])], ignore_index=True)
                df_synth_feasible_designs = pd.concat([df_synth_feasible_designs, feasible_df], ignore_index=True)

        # Save the synthesizability and feasibility statistics to CSV
        df_stats.to_csv(os.path.join(self.APPLICATION_OUTPUT_DIR, "01_synthesizability_feasibility_statistics.csv"), index=False)
        
        # Define the list of columns to apply the replacement
        resources_columns = [
            'BRAM_Utilization_percentage', 
            'DSP_Utilization_percentage', 
            'FF_Utilization_percentage', 
            'LUT_Utilization_percentage'
        ]

        # Replace 0 with 1 in each of the specified columns
        for column in resources_columns:
            df_synth_feasible_designs[column] = df_synth_feasible_designs[column].replace(0, 1)

        # Sum the values of the specified resource utilization columns (BRAM, DSP, FF, LUT) for each row 
        # and store the result in a new column 'Total_Resources_Utilization_percentage'
        df_synth_feasible_designs['Total_Resources_Utilization_percentage'] = df_synth_feasible_designs[resources_columns].sum(axis=1)

        # Save the synthesizable and feasible designs to CSV
        df_synth_feasible_designs.to_csv(os.path.join(self.APPLICATION_OUTPUT_DIR, "02_synthesizable_feasible_designs.csv"), index=False)

    def _plot_synthesizability_feasibility_statistics(self):
        """
        Generates and saves pie charts visualizing synthesizability and feasibility statistics for different devices
        and target clock periods. The data is read from a CSV file generated in a previous step.

        Each subplot in the figure represents a device and target clock period combination, showing the distribution of 
        "Not Synthesizable", "Feasible", and "Not Feasible" entries as a pie chart.

        The figure is saved as a PDF file in the output directory.

        Args:
            None

        Returns:
            None
        """
        # Path to the CSV file containing synthesizability and feasibility statistics
        fname = os.path.join(self.APPLICATION_OUTPUT_DIR, "01_synthesizability_feasibility_statistics.csv")
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(fname)
        
        # Create a 2x3 grid of subplots
        _, axes = plt.subplots(2, 3, figsize=(14, 10))
        
        # Set the Seaborn theme and color palette
        sns.set_theme()
        sns.set(style="ticks", color_codes=True)
        colors = sns.color_palette(self.SEABORN_PALETTE, 3)
        
        # Initialize counters for subplots
        count_i = 0
        for device in self.AVAILABLE_DEVICES:
            count_j = 0
            for target_clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                # Filter the DataFrame for the current device and clock period
                device_clock_period_condition = (df["Device"] == device) & (df["Clock_Period_nsec"] == target_clock_period)
                current_df = df[device_clock_period_condition]
        
                # Extract the counts for "Not Synthesizable", "Feasible", and "Not Feasible"
                not_synthesizable = int(current_df["Not_Synthesizable"])
                feasible = int(current_df["Feasible"])
                not_feasible = int(current_df["Not_Feasible"])
        
                # Define labels, counts, and colors for the pie chart
                labels = ["Not Synthesizable", "Feasible", "Not Feasible"]
                counts = [not_synthesizable, feasible, not_feasible]
                
                # Set the title for the pie chart based on the device and target clock period
                pie_chart_title = self.AVAILABLE_DEVICES_MAP[device] + "@" + self.AVAILABLE_TARGET_CLOCK_PERIODS_MAP[target_clock_period]
                
                # Plot the pie chart on the appropriate subplot
                axes[count_i, count_j].pie(
                    counts, 
                    colors=colors, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    wedgeprops={'edgecolor': 'black'}
                )
                axes[count_i, count_j].set_title(pie_chart_title)

                count_j += 1
            count_i += 1

        # Add a global legend for all pie charts
        plt.legend(labels, loc="best")
        
        # Adjust layout to prevent overlapping titles and labels
        plt.tight_layout()
        
        # Save the figure as a PDF file
        fname = "synthesizability_feasibility_piecharts.pdf"
        fpath = os.path.join(self.APPLICATION_OUTPUT_DIR, fname)
        plt.savefig(fpath)
        
        plt.close()

    def _plot_quality_of_result_metrics_distributions(self):
        """
        Plots the distributions of quality of result (QoR) metrics for synthesizable and feasible designs.
        
        This function generates box plots for various QoR metrics across different clock periods and devices, 
        visualizing the synthesizability and feasibility of designs based on the specified metrics.
        
        Returns:
            None: The plots are saved to the output directory.
        """
        # Load the dataset from the specified output directory
        df = pd.read_csv(os.path.join(self.APPLICATION_OUTPUT_DIR, "02_synthesizable_feasible_designs.csv"))

        # Set the Seaborn theme and color palette
        sns.set_theme()
        sns.set(style="ticks", color_codes=True)
        palette = sns.color_palette(self.SEABORN_PALETTE, 2)

        # Mapping QoR metric names to their y-axis labels
        qor_metric_name_ylabel_map = {
            "Latency_msec": "Design Latency (msec)",
            "Synthesis_Time_sec": "Synthesis Time (sec)",
            "BRAM_Utilization_percentage": "BRAM Utilization (%)",
            "DSP_Utilization_percentage": "DSP Utilization (%)",
            "FF_Utilization_percentage": "FF Utilization (%)",
            "LUT_Utilization_percentage": "LUT Utilization (%)"
        }
        
        order = self.AVAILABLE_TARGET_CLOCK_PERIODS

        # Create a grid of subplots
        _, axes = plt.subplots(2, 3, figsize=(14, 10))
        
        # Iterate through the specified QoR metrics
        for count, qor_metric_name in enumerate(["Latency_msec", "Synthesis_Time_sec", 
                                                "BRAM_Utilization_percentage", "DSP_Utilization_percentage", 
                                                "FF_Utilization_percentage", "LUT_Utilization_percentage"]):
            
            # Calculate the subplot indices
            i, j = divmod(count, 3)

            # Create a box plot for the current QoR metric
            sns_plt = sns.boxplot(data=df, x="Clock_Period_nsec", y=qor_metric_name, 
                                hue="Device", palette=palette, width=0.65, 
                                showfliers=True, fliersize=1, order=order, 
                                ax=axes[i, j])

            if qor_metric_name != "Latency_msec" and qor_metric_name != "Synthesis_Time_sec":
                axes[i, j].axhline(y=100, linestyle='--', color='r', linewidth=1)

            # Set the y-axis to logarithmic scale
            axes[i, j].set_yscale('log')

            # Set the x and y labels
            axes[i, j].set_xlabel("", fontsize=12, weight='bold')
            axes[i, j].set_ylabel(qor_metric_name_ylabel_map[qor_metric_name], fontsize=12, weight='bold')

            # Customize the x-ticks and labels
            axes[i, j].set_xticks([0, 1, 2])
            axes[i, j].set_xticklabels(["100MHz", "200MHz", "300MHz"], fontsize=12, weight='bold')

            # Get the legend handles and labels
            handles, _ = sns_plt.get_legend_handles_labels()

            # Set the legend for the current subplot
            axes[i, j].legend(handles=handles, labels=["ZCU104", "U200"], title="", loc='upper right')

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        
        # Save the figure as a PDF file
        fname = "quality_of_result_metrics_boxplots.pdf"
        fpath = os.path.join(self.APPLICATION_OUTPUT_DIR, fname)
        plt.savefig(fpath)

        # Close the plotting context
        plt.close()

    def _get_and_plot_pareto_frontiers(self):
        """
        Calculate, visualize, and export Pareto frontiers for synthesizable and feasible designs based on latency and total resource utilization.

        The function performs the following steps:
        1. Reads the synthesizable and feasible designs from a CSV file.
        2. For each combination of device and clock period, it computes the Pareto frontiers by minimizing latency and total resource utilization.
        3. Visualizes the results as scatter plots, highlighting the Pareto-optimal designs.
        4. Exports the calculated Pareto frontiers into a CSV file.

        The scatter plots depict total resource utilization (BRAM%, DSP%, FF%, and LUT% combined) on the x-axis and design latency on the y-axis. 
        Pareto-optimal designs are highlighted in each plot.

        Args:
            None

        Returns:
            None
        """
        # Load the CSV file containing synthesizable and feasible designs
        df = pd.read_csv(os.path.join(self.APPLICATION_OUTPUT_DIR, "02_synthesizable_feasible_designs.csv"))

        # Initialize an empty DataFrame to store Pareto frontier results for all devices and clock periods
        df_application_pareto_frontiers = pd.DataFrame()

        # Initialize a 2x3 grid for the scatter plots
        _, axes = plt.subplots(2, 3, figsize=(14, 10))
        
        # Set the Seaborn theme and color palette
        sns.set_theme()
        sns.set(style="ticks", color_codes=True)
        palette = sns.color_palette(self.SEABORN_PALETTE, 2)

        count = 0  # Counter to track subplot indices

        # Iterate over each device in the available devices list
        for device in self.AVAILABLE_DEVICES:
            # Iterate over each target clock period in the available target clock periods
            for clock_period in self.AVAILABLE_TARGET_CLOCK_PERIODS:
                # Filter the DataFrame for the current device and clock period
                current_df = df[(df["Device"] == device) & (df["Clock_Period_nsec"] == clock_period)]

                # Calculate the Pareto set based on minimizing latency and total resource utilization
                pareto_mask = paretoset(current_df[["Latency_msec", "Total_Resources_Utilization_percentage"]], sense=["min", "min"])

                # Extract the Pareto frontier designs based on the calculated mask
                df_pareto_frontier = current_df[pareto_mask]
                
                # Add 'Pareto_Optimal' column and assign scatter point size based on whether the design is Pareto-optimal
                current_df.loc[:, "Pareto_Optimal"] = pareto_mask
                current_df["Scatter_Point_Size"] = current_df["Pareto_Optimal"].apply(lambda x: 75 if x else 20)

                # Calculate the subplot indices using divmod (counter // 3 gives row, counter % 3 gives column)
                i, j = divmod(count, 3)
                
                # Create scatter plot on the respective subplot
                sns.scatterplot(
                    x = "Total_Resources_Utilization_percentage", 
                    y = "Latency_msec", 
                    hue = "Pareto_Optimal", 
                    size = "Scatter_Point_Size", 
                    sizes = (20, 75), 
                    data = current_df, 
                    legend = False, 
                    palette = palette, 
                    ax = axes[i, j]
                )

                # Set scatter plot title based on device and clock period
                scatterplot_title = self.AVAILABLE_DEVICES_MAP[device] + "@" + self.AVAILABLE_TARGET_CLOCK_PERIODS_MAP[clock_period]                
                    
                # Set labels and title for the current subplot
                axes[i, j].set_xlabel("BRAM% + DSP% + FF% + LUT%", fontsize=12, weight='bold')
                axes[i, j].set_ylabel("Design Latency (msec)", fontsize=12, weight='bold')
                axes[i, j].set_title(scatterplot_title, fontsize=12, weight='bold')
                
                count += 1  # Increment the counter for the next subplot

                # Append the current Pareto frontier to the overall results DataFrame
                df_application_pareto_frontiers = pd.concat([df_application_pareto_frontiers, df_pareto_frontier], ignore_index=True)

        # Adjust layout for the figure and display the plots
        plt.tight_layout()
        
        # Save the figure as a PDF file
        fname = "pareto_frontier_scatterplots.pdf"
        fpath = os.path.join(self.APPLICATION_OUTPUT_DIR, fname)
        plt.savefig(fpath)
        
        plt.close()

        # Export the collected Pareto frontiers to a CSV file
        pareto_output_path = os.path.join(self.APPLICATION_OUTPUT_DIR, "03_pareto_frontiers.csv")
        df_application_pareto_frontiers.to_csv(pareto_output_path, index=False)

    def analyze(self):
        """
        Perform analysis to compute and visualize synthesizability and feasibility statistics for the application.

        This function reads the application's data from a CSV file, generates statistics about synthesizability and feasibility
        for different devices and clock periods, and creates visualizations including pie charts and quality-of-result metrics.
        The results are saved in the specified output directory.

        Args:
            None

        Returns:
            None
        """
        # Log the current application name
        print(f"\nAnalyzing Application: {self.APPLICATION_NAME}\n")

        # Generate statistics on synthesizability and feasibility and save to CSV
        self._get_synthesible_feasible_designs_and_statistics(self.df)
        
        # Create and save pie charts based on the synthesizability and feasibility statistics
        self._plot_synthesizability_feasibility_statistics()
        
        # Plot quality of result metrics distributions
        self._plot_quality_of_result_metrics_distributions()
        
        # Get and visualize Pareto frontiers
        self._get_and_plot_pareto_frontiers()

        print(f"\nFinished Application Analysis")
