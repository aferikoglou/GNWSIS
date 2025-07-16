import os
import pandas as pd
from modules.dbReader import DBReader

# Total available resources for each FPGA device
TOTAL_AVAILABLE_RESOURCES = {
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


def add_speedup_and_resource_metrics(df_output):
    """
    Adds speedup and estimated resource utilization columns to a DataFrame.

    Args:
        df_output (pd.DataFrame): DataFrame with design results for each application configuration.

    Returns:
        pd.DataFrame: Modified DataFrame including speedup and estimated BRAM, DSP, FF, and LUT usage.
    """
    # Load baseline data for comparison
    df_baseline_implementations = pd.read_csv('./data/ApplicationBaselineInformation.csv')

    # Initialize resource columns
    for resource_metric_name in ["BRAM", "DSP", "FF", "LUT"]:
        new_column_name = f"{resource_metric_name}s"
        df_output[new_column_name] = 0

    # Iterate over each row to compute metrics
    for index, row in df_output.iterrows():
        app_name = row["Application_Name"]
        device_id = row["Device"]
        clock_period_nsec = row["Clock_Period_nsec"]
        design_latency_msec = row["Latency_msec"]

        # Filter baseline data for the matching app/device/clock
        condition = (
            (df_baseline_implementations["app_name"] == app_name) &
            (df_baseline_implementations["device_id"] == device_id) &
            (df_baseline_implementations["clock_period_nsec"] == clock_period_nsec)
        )
        df_app = df_baseline_implementations[condition]

        # Compute speedup
        try:
            baseline_design_latency_msec = float(df_app["design_latency_msec"])
            speedup_value = baseline_design_latency_msec / design_latency_msec
            df_output.at[index, "Speedup"] = speedup_value
        except:
            df_output.at[index, "Speedup"] = None

        # Compute estimated resource utilization from percentage
        for resource_metric_name in ["BRAM", "DSP", "FF", "LUT"]:
            new_column_name = f"{resource_metric_name}s"
            percentage_column_name = f"{resource_metric_name}_Utilization_percentage"

            percentage = row[percentage_column_name]
            if percentage > 100:
                df_output.at[index, new_column_name] = None
            else:
                available_resources = TOTAL_AVAILABLE_RESOURCES[device_id][new_column_name]
                df_output.at[index, new_column_name] = int((percentage / 100) * available_resources)

    return df_output


def aggregate_data():
    """
    Aggregates all Pareto frontier CSV files across applications and computes final metrics.

    Returns:
        pd.DataFrame: Aggregated DataFrame with added metrics and fixed zero-percent resources.
    """
    df_output = pd.DataFrame()

    # Load each application's CSV and concatenate
    for fname in os.listdir("./CSVS"):
        INPUT_PATH = os.path.join("./CSVS", fname)
        app_name = fname.split(".")[0]

        df = pd.read_csv(INPUT_PATH)
        df["Application_Name"] = app_name
        df_output = pd.concat([df_output, df], ignore_index=True)

    # Replace zero utilization percentages with 1%
    columns_to_replace = [
        "BRAM_Utilization_percentage", 
        "DSP_Utilization_percentage", 
        "FF_Utilization_percentage", 
        "LUT_Utilization_percentage"
    ]
    df_output[columns_to_replace] = df_output[columns_to_replace].replace(0.0, 1.0)

    # Add metrics
    df_output = add_speedup_and_resource_metrics(df_output)

    return df_output


if __name__ == "__main__":
    print("Started database read...\n")

    db_reader = DBReader()
    db_reader.read_databases()
    
    print("Finished database read...\n")

    print("Aggregating CSV files and computing metrics...\n")

    # Aggregate and process the final CSV file
    df_output = aggregate_data()

    new_order = [
        'Application_Name',
        'Version',
        'Device',
        'Clock_Period_nsec',
        "Array_1", "Array_2", "Array_3", "Array_4", "Array_5", "Array_6", "Array_7", "Array_8", "Array_9", "Array_10",
        "Array_11", "Array_12", "Array_13", "Array_14", "Array_15", "Array_16", "Array_17", "Array_18", "Array_19", "Array_20", "Array_21", "Array_22",
        "OuterLoop_1", "OuterLoop_2", "OuterLoop_3", "OuterLoop_4", "OuterLoop_5", "OuterLoop_6", "OuterLoop_7", "OuterLoop_8", "OuterLoop_9", "OuterLoop_10",
        "OuterLoop_11", "OuterLoop_12", "OuterLoop_13", "OuterLoop_14", "OuterLoop_15", "OuterLoop_16", "OuterLoop_17", "OuterLoop_18", "OuterLoop_19",
        "OuterLoop_20", "OuterLoop_21", "OuterLoop_22", "OuterLoop_23", "OuterLoop_24", "OuterLoop_25", "OuterLoop_26",
        "InnerLoop_1_1", "InnerLoop_1_2", "InnerLoop_1_3", "InnerLoop_1_4", "InnerLoop_1_5", "InnerLoop_1_6", "InnerLoop_1_7", "InnerLoop_1_8",
        "InnerLoop_1_9", "InnerLoop_1_10", "InnerLoop_1_11", "InnerLoop_1_12", "InnerLoop_1_13", "InnerLoop_1_14", "InnerLoop_1_15", "InnerLoop_1_16",
        "InnerLoop_2_1", "InnerLoop_2_2", "InnerLoop_2_3", "InnerLoop_2_4", "InnerLoop_2_5", "InnerLoop_2_6", "InnerLoop_2_7",
        "InnerLoop_3_1", "InnerLoop_3_2", "InnerLoop_3_3", "InnerLoop_3_4", "InnerLoop_3_5",
        "InnerLoop_4_1", "InnerLoop_4_2",
        'Latency_msec',
        'Synthesis_Time_sec',
        'BRAM_Utilization_percentage', 'DSP_Utilization_percentage',
        'FF_Utilization_percentage', 'LUT_Utilization_percentage',
        "Speedup", "BRAMs", "DSPs", "FFs", "LUTs"
    ]

    # Reorder columns
    df_output = df_output[new_order]

    # Save to CSV
    fpath = os.path.join("GNΩSIS.csv")
    df_output.to_csv(fpath, index=False)

    print(f"GNΩSIS dataset generated successfully at {fpath}")
