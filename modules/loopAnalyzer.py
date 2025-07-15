import os
import json

class LoopAnalyzer:
    """
    A class to analyze loop structures and operations within a dataset of applications.
    """

    def __init__(self, MODE):
        # Set the dataset directory
        self.DATASET_DIR = os.path.join("data", "ApplicationDataset")
        self.ACTION_POINT_LABEL_MAPPING_DIR = os.path.join("data", "ApplicationAPLMapping")
        
        self.LOOP_MAGIC_NUMBER = -2000000
        
        self.MODE = MODE
        
        # TODO: Hand-written part for CollectiveHLS
        self.ACTION_POINT_INDEX_START = 133
        
        # Load loop information for each application in the dataset
        self.dataset_loops_map = self._get_dataset_loops_map()

        # Analyze the loop structures and extract relevant metrics
        self.maximum_dataset_loop_structures, self.maximum_dataset_nesting_level, self.maximum_dataset_loops_in_each_level, self.transformed_dataset_operations_map = self._analyze_dataset_loops(self.dataset_loops_map)
        
        # Display key statistics from the dataset analysis
        print(f"The maximum number of loop structures in an application in our dataset is {self.maximum_dataset_loop_structures}")
        print(f"The maximum nesting level in a loop structure in an application in our dataset is {self.maximum_dataset_nesting_level}")
        print(f"The maximum number of loops in each level of our dataset applications are {self.maximum_dataset_loops_in_each_level}")
        
        # TODO: Hand-written part for CollectiveHLS
        self.loop_action_point_names = [str(index) for index in range(45, 100 + 1)] if self.MODE == "CollectiveHLS" else self._get_loop_action_point_names(self.maximum_dataset_loops_in_each_level)
        self.operation_names = [str(index) for index in range(101, 110 + 1)] if self.MODE == "CollectiveHLS" else ["ADD", "FADD", "SUB", "FSUB", "MUL", "FMUL", "DIV", "FDIV", "LOAD", "STORE"]
        
        self.transformed_dataset_loop_map = self._get_app_loop_vectors(self.dataset_loops_map, self.maximum_dataset_nesting_level, self.maximum_dataset_loops_in_each_level)
        
    def _get_outermost_loops(self, loops_information):
        """
        Retrieves the outermost loops in a given application's loop information.

        Args:
            loops_information (dict): A dictionary containing information about loops.
        
        Returns:
            list: A list of loop labels corresponding to the outermost loops.
        """
        outermost_loops = []
        for loop_label, loop_info in loops_information.items():
            if loop_info.get("Outermost", False):
                outermost_loops.append(loop_label)
        return outermost_loops

    def _get_application_maximum_nesting_level(self, loops_information):
        """
        Finds the maximum nesting level of loops in the application.

        Args:
            loops_information (dict): A dictionary containing information about loops.
        
        Returns:
            int: The maximum nesting level of the loops.
        """
        return max(loop_info["NestingLevel"] for loop_info in loops_information.values())

    def _get_loop_label_from_uid(self, loops_information, uid):
        """
        Retrieves the loop label corresponding to a given UID.

        Args:
            loops_information (dict): A dictionary containing information about loops.
            uid (str): The unique identifier (UID) of the loop.
        
        Returns:
            str: The label of the loop that matches the UID.
        """
        for loop_label, loop_info in loops_information.items():
            if loop_info.get("UID") == uid:
                return loop_label
        return ""

    def _get_application_loops_adjacency_list(self, loops_information):
        """
        Builds an adjacency list representing the loop hierarchy for the application.

        Args:
            loops_information (dict): A dictionary containing information about loops.
        
        Returns:
            dict: An adjacency list where each loop label maps to its subloop labels.
        """
        loops_adjacency_list = {}
        for loop_label, loop_info in loops_information.items():
            subloop_labels = []
            subloop_uids = loop_info.get("Subloops", [])
            for subloop_uid in subloop_uids:
                subloop_label = self._get_loop_label_from_uid(loops_information, subloop_uid)
                if subloop_label:
                    subloop_labels.append(subloop_label)
            loops_adjacency_list[loop_label] = subloop_labels
        return loops_adjacency_list

    def _get_application_loops_in_each_level(self, loops_information, application_maximum_nesting_level):
        """
        Counts the number of loops in each nesting level of the application.

        Args:
            loops_information (dict): A dictionary containing information about loops.
            application_maximum_nesting_level (int): The maximum nesting level for the application.
        
        Returns:
            list: A list where the index represents the nesting level and the value represents the number of loops at that level.
        """
        loops_in_each_level = [0] * application_maximum_nesting_level
        for loop_info in loops_information.values():
            nesting_level = loop_info["NestingLevel"]
            loops_in_each_level[nesting_level - 1] += 1
        return loops_in_each_level

    def _get_application_loop_data_in_level_k(self, loops_information, k):
        """
        Retrieves loop data for a specific nesting level.

        Args:
            loops_information (dict): A dictionary containing information about loops.
            k (int): The nesting level to retrieve data for.
        
        Returns:
            list: A list of tuples containing the loop label, actual loop limit, and inferred loop limit for each loop at level k.
        """
        return [
            (loop_label, loop_info["LoopLimActual"], loop_info["LoopLimInferred"])
            for loop_label, loop_info in loops_information.items()
            if loop_info["NestingLevel"] == k
        ]

    def _get_application_loop_data_in_all_levels(self, loops_information, application_maximum_nesting_level):
        """
        Collects loop data for all nesting levels in the application.

        Args:
            loops_information (dict): A dictionary containing information about loops.
            application_maximum_nesting_level (int): The maximum nesting level for the application.
        
        Returns:
            dict: A dictionary where each nesting level maps to a list of loop data tuples.
        """
        return {
            k: self._get_application_loop_data_in_level_k(loops_information, k)
            for k in range(1, application_maximum_nesting_level + 1)
        }

    def _get_application_loop_operations(self, loops_information, application_maximum_nesting_level, application_loop_data_in_all_levels):
        """
        Calculates the total number of operations (e.g., addition, subtraction, multiplication) 
        for loops at the deepest nesting level in an application.

        Args:
            loops_information (dict): A dictionary containing information about each loop in the application.
            application_maximum_nesting_level (int): The maximum nesting level for loops in the application.
            application_loop_data_in_all_levels (dict): Loop data for all levels of the application.

        Returns:
            list: A list containing the total counts for different types of operations in the form:
                [add_ops, fadd_ops, sub_ops, fsub_ops, mul_ops, fmul_ops, udiv_ops, fdiv_ops, load_ops, store_ops].
        """
        # Get loops in the deepest nesting level
        loops_in_deepest_level = application_loop_data_in_all_levels.get(application_maximum_nesting_level, [])

        # Initialize operation counters
        add_ops, fadd_ops = 0, 0
        sub_ops, fsub_ops = 0, 0
        mul_ops, fmul_ops = 0, 0
        udiv_ops, sdiv_ops, fdiv_ops = 0, 0, 0
        load_ops, store_ops = 0, 0

        # Iterate through each loop in the deepest level
        for loop_data in loops_in_deepest_level:
            loop_label = loop_data[0]
            loop_localops = loops_information.get(loop_label, {}).get("LocalOps", [])

            # Accumulate operation counts
            add_ops   += loop_localops[13]
            fadd_ops  += loop_localops[14]
            sub_ops   += loop_localops[15]
            fsub_ops  += loop_localops[16]
            mul_ops   += loop_localops[17]
            fmul_ops  += loop_localops[18]
            udiv_ops  += loop_localops[19]
            sdiv_ops  += loop_localops[20]
            fdiv_ops  += loop_localops[21]
            load_ops  += loop_localops[32]
            store_ops += loop_localops[33]

        # Return the aggregated operation counts
        return [add_ops, fadd_ops, sub_ops, fsub_ops, mul_ops, fmul_ops, udiv_ops, fdiv_ops, load_ops, store_ops]


    def get_application_loop_info(self, app_name):
        """
        Retrieves loop-related information for a given application.

        Args:
            app_name (str): The name of the application.
        
        Returns:
            tuple: Various loop-related data such as outermost loops, nesting level, adjacency list, and loop operations.
        """
        source_info_path = os.path.join(self.DATASET_DIR, app_name, "src_info.json")
        with open(source_info_path, 'r') as f:
            data = json.load(f)
        
        loops_information = data["loops"]
        
        outermost_loops = self._get_outermost_loops(loops_information)
        max_nesting_level = self._get_application_maximum_nesting_level(loops_information)
        adjacency_list = self._get_application_loops_adjacency_list(loops_information)
        loops_in_each_level = self._get_application_loops_in_each_level(loops_information, max_nesting_level)
        loop_data_all_levels = self._get_application_loop_data_in_all_levels(loops_information, max_nesting_level)
        loop_operations = self._get_application_loop_operations(loops_information, max_nesting_level, loop_data_all_levels)

        return (outermost_loops, max_nesting_level, adjacency_list, loops_in_each_level, loop_data_all_levels, loop_operations)

    def _get_dataset_loops_map(self):
        """
        Constructs a mapping of loop information for all applications in the dataset.

        Returns:
            dict: A dictionary where each application maps to its loop-related information.
        """
        dataset_loops_map = {}
        for app_name in os.listdir(self.DATASET_DIR):
            dataset_loops_map[app_name] = self.get_application_loop_info(app_name)
        return dataset_loops_map

    def _analyze_dataset_loops(self, dataset_loops_map):
        """
        Analyzes the loop structures and operations across all applications in the dataset.

        Args:
            dataset_loops_map (dict): A dictionary containing loop information for each application.
        
        Returns:
            tuple: 
                - maximum_dataset_loop_structures (int): The maximum number of outermost loops in any application.
                - maximum_dataset_nesting_level (int): The maximum nesting level of loops across the dataset.
                - maximum_dataset_loops_in_each_level (list): A list containing the maximum number of loops at each nesting level.
        """
        transformed_dataset_operations_map = {}

        # Initialize variables to track the maximum structures and nesting levels across the dataset
        maximum_dataset_loop_structures = 0
        maximum_dataset_nesting_level = 0

        # Iterate over each application to analyze its loop structure
        for app_name, app_loop_info in sorted(dataset_loops_map.items()):
            outermost_loops, app_max_nesting_level, _, app_loops_in_each_level, _, app_loop_operations = app_loop_info
            
            # Track the maximum number of outermost loops in the dataset
            maximum_dataset_loop_structures = max(maximum_dataset_loop_structures, len(outermost_loops))
            
            # Track the maximum nesting level across all applications
            maximum_dataset_nesting_level = max(maximum_dataset_nesting_level, app_max_nesting_level)
            
            # Store the operations for the current application
            transformed_dataset_operations_map[app_name] = app_loop_operations

        # Initialize a list to track the maximum number of loops at each nesting level
        maximum_dataset_loops_in_each_level = [0] * maximum_dataset_nesting_level

        # For each application, update the maximum number of loops found at each nesting level
        for app_name, app_loop_info in dataset_loops_map.items():
            _, app_max_nesting_level, _, app_loops_in_each_level, _, _ = app_loop_info
            
            # Update the max loops count at each nesting level based on the current application
            for level in range(len(app_loops_in_each_level)):
                maximum_dataset_loops_in_each_level[level] = max(
                    maximum_dataset_loops_in_each_level[level], 
                    app_loops_in_each_level[level]
                )

        return maximum_dataset_loop_structures, maximum_dataset_nesting_level, maximum_dataset_loops_in_each_level, transformed_dataset_operations_map
    
    def _get_loop_action_point_names(self, maximum_dataset_loops_in_each_level):
        """
        Generates action point names for loops at different nesting levels.

        Args:
            maximum_dataset_loops_in_each_level (list): A list where each element represents
                                                        the maximum number of loops in a particular nesting level.

        Returns:
            list: A list of action point names for both outer and inner loops.
        """
        action_point_names = []

        # Generate names for outer loops
        action_point_names.extend(
            [f"OuterLoop_{outer_loop_num}" for outer_loop_num in range(1, maximum_dataset_loops_in_each_level[0] + 1)]
        )

        # Generate names for inner loops at each nesting level
        for nesting_level, max_loops in enumerate(maximum_dataset_loops_in_each_level[1:], start=1):
            action_point_names.extend(
                [f"InnerLoop_{nesting_level}_{inner_loop_num}" for inner_loop_num in range(1, max_loops + 1)]
            )

        return action_point_names

    def _get_app_loop_vectors(self, dataset_loops_map, maximum_dataset_nesting_level, maximum_dataset_loops_in_each_level):
        """
        Converts loop data for each application into vectors that represent the loop limits across 
        different nesting levels. If an application does not have loops at a particular level, 
        a magic number is added as padding.

        Args:
            dataset_loops_map (dict): A dictionary containing loop data for each application.
            maximum_dataset_nesting_level (int): The maximum nesting level across all applications in the dataset.
            maximum_dataset_loops_in_each_level (list): A list representing the maximum number of loops at each nesting level.

        Returns:
            dict: A dictionary where the keys are application names and the values are loop vectors representing 
                the loop limits, including padding where necessary.
        """
        transformed_dataset_loops_map = {}

        # Process each application in the dataset
        for app_name, app_loop_data in dataset_loops_map.items():
            application_loop_vector = []
            count = 0
            action_point_index = self.ACTION_POINT_INDEX_START if self.MODE == "CollectiveHLS" else 0
            
            # Path to save loop-to-action point mapping
            fname = f"{app_name}.txt"
            fpath = os.path.join(self.ACTION_POINT_LABEL_MAPPING_DIR, fname)
            
            with open(fpath, "a") as f:
                # Extract the application's maximum nesting level and loop data for all levels
                application_maximum_nesting_level = app_loop_data[1]
                application_loop_data_in_all_levels = app_loop_data[4]

                # Iterate over each nesting level
                for i in range(maximum_dataset_nesting_level):
                    max_loops_in_level = maximum_dataset_loops_in_each_level[i]
                    current_nesting_level = i + 1  # Nesting levels are 1-based

                    # Check if the application has loops at this nesting level
                    if current_nesting_level <= application_maximum_nesting_level:
                        # Retrieve loop data for the current nesting level
                        loop_data_in_level = application_loop_data_in_all_levels[current_nesting_level]
                        size = len(loop_data_in_level)

                        # Add the actual loop limits to the vector and write loop-action mappings to file
                        for loop_label, loop_lim_actual, loop_lim_inferred in loop_data_in_level:
                            application_loop_vector.append(loop_lim_actual)
                            count += 1
                            
                            if self.MODE == "CollectiveHLS":
                                f.write(f"{action_point_index},{loop_label}")
                            else:
                                f.write(f"{self.loop_action_point_names[action_point_index]},{loop_label}")
                            
                            action_point_index += 1

                        # Add padding (magic numbers) if there are fewer loops than the maximum allowed
                        padding_zeros = [self.LOOP_MAGIC_NUMBER] * (max_loops_in_level - size)
                    else:
                        # If no loops at this level, add only padding (magic numbers)
                        padding_zeros = [self.LOOP_MAGIC_NUMBER] * max_loops_in_level

                    # Append the padding and level data to the loop vector
                    application_loop_vector.extend(padding_zeros)
                    count = len(padding_zeros)
                    action_point_index += count

            # Store the loop vector for the current application
            transformed_dataset_loops_map[app_name] = application_loop_vector

        return transformed_dataset_loops_map

    
    def get_app_loop_vector(self, app_name):
        """
        Retrieves the loop vector for a specific application.

        Parameters:
            app_name (str): The name of the application for which to retrieve the loop vector.

        Returns:
            list: The loop vector for the specified application.

        Raises:
            KeyError: If the application name is not found in the transformed dataset loop map.
        """
        try:
            return self.transformed_dataset_loop_map[app_name]
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset loop map.")

    def get_app_operations_vector(self, app_name):
        """
        Retrieves the operations vector for a specific application.

        Parameters:
            app_name (str): The name of the application for which to retrieve the loop vector.

        Returns:
            list: The operations vector for the specified application.

        Raises:
            KeyError: If the application name is not found in the transformed dataset operations map.
        """
        try:
            return self.transformed_dataset_operations_map[app_name]
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset operations map.")
        
    def get_app_loop_map(self, app_name):
        """
        Retrieves the transformed loop map for a specified application.

        Args:
            app_name (str): The name of the application whose loop map is to be retrieved.

        Returns:
            dict: A dictionary where keys are loop indices (starting from 45, 1-based) 
                and values are the corresponding loop elements.

        Raises:
            KeyError: If the application name is not found in the transformed dataset loop map.
        """
        try:
            # Fetch the transformed loop vector for the given application
            transformed_loop_vector = self.transformed_dataset_loop_map[app_name]

            # Calculate total number of loops from the maximum loops per nesting level
            total_loops = sum(self.maximum_dataset_loops_in_each_level)

            if self.MODE == "CollectiveHLS":
                # Build a loop map with 1-based indices starting from 45
                app_loop_map = {
                    str(index + 45): transformed_loop_vector[index]
                    for index in range(total_loops)
                }
            else:
                # Build a loop map using custom loop action point names
                app_loop_map = {
                    self.loop_action_point_names[index]: transformed_loop_vector[index]
                    for index in range(total_loops)
                }

            return app_loop_map
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset loop map.")

    def get_app_operations_map(self, app_name):
        """
        Retrieves the transformed operations map for a specified application.

        Args:
            app_name (str): The name of the application whose operations map is to be retrieved.

        Returns:
            dict: A dictionary where keys are operation indices (starting from 101, 1-based) 
                and values are the corresponding operation elements.

        Raises:
            KeyError: If the application name is not found in the transformed dataset operations map.
        """
        try:
            # Fetch the transformed operations vector for the given application
            transformed_operations_vector = self.transformed_dataset_operations_map[app_name]

            if self.MODE == "CollectiveHLS":
                # Build an operations map with 1-based indices starting from 101
                app_operations_map = {
                    str(index + 101): transformed_operations_vector[index]
                    for index in range(len(transformed_operations_vector))
                }
            else:
                # Build an operations map using custom operation names
                app_operations_map = {
                    self.operation_names[index]: transformed_operations_vector[index]
                    for index in range(len(transformed_operations_vector))
                }

            return app_operations_map
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset operations map.")

    def get_loop_column_names(self):
        """
        Retrieves the loop action point names.

        Returns:
            list: A list containing the loop action point names.
        """
        return self.loop_action_point_names

    def get_operation_column_names(self):
        """
        Retrieves the operation names.

        Returns:
            list: A list containing the operation names.
        """
        return self.operation_names
