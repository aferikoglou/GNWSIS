import os

class DirectivesManipulator:
    def __init__(self, app_name):
        """
        Initializes the DirectivesManipulator class by loading relevant file paths and
        extracting the top-level function name, file name, extension, and directives 
        per action point for old and new DSEs.
        """
        APPLICATION_INPUT_DIR = os.path.join("data", "ApplicationDataset", app_name)
        KERNEL_INFO_PATH = os.path.join(APPLICATION_INPUT_DIR, "kernel_info.txt")
        APPLICATION_APL_MAPPING_PATH = os.path.join("data", "ApplicationAPLMapping", app_name + ".txt")
        
        self.ACTION_POINT_NAMES = [
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

        self.label_action_point_map = self._get_label_action_point_map(APPLICATION_APL_MAPPING_PATH)

        self.directives_per_action_point_old_dse = self._get_directives_per_action_point(KERNEL_INFO_PATH, 1024)
        self.directives_per_action_point_new_dse = self._get_directives_per_action_point(KERNEL_INFO_PATH, 128)

    def _get_label_action_point_map(self, application_apl_mapping_path):
        """
        Reads a mapping file that contains action points and corresponding labels, and stores this information
        in a dictionary where the key is the label, and the value is the action point.

        Args:
            application_apl_mapping_path (str): Path to the file that contains the action point-to-label mappings.

        Returns:
            dict: A dictionary with labels as keys and action points as values.
        """
        label_action_point_map = {}

        # Open and read the mapping file line by line
        with open(application_apl_mapping_path, 'r') as file:
            lines = file.readlines()

        # Process each line to extract action points and labels
        for line in lines:
            # Clean and split each line into parts (action_point, label)
            action_point, label = line.strip().split(',')
            
            # Map the label to the action point
            label_action_point_map[label] = action_point

        return label_action_point_map

    def _get_directives_per_action_point(self, kernel_info_path, array_maximum_size):   
        """
        Generates a list of optimization directives (e.g., HLS pragmas) for each action point in a given kernel.

        This method parses the kernel information file to extract action points, identifies whether they are loops 
        or arrays, and then generates corresponding High-Level Synthesis (HLS) directives. For loops, directives such 
        as pipeline and unroll are generated, while for arrays, partitioning directives (complete, block, cyclic) 
        are generated depending on the size of the arrays and their dimensions.

        Args:
            kernel_info_path (str): The path to the file that contains kernel information including loops and arrays.
            array_maximum_size (int): The maximum allowable size for array partitioning.

        Returns:
            list: A list of lists where each inner list contains the generated directives (e.g., HLS pragmas) for a 
            specific action point (loop or array).
        """
        
        # Initialize an empty list to store directives per action point
        directives_per_action_point = []
        
        # Open the kernel info file and read all lines
        with open(kernel_info_path, 'r') as fr:
            lines = fr.readlines()

        lines_num = len(lines)  # Get the number of lines
        action_point_counter = 0  # Initialize a counter for action points
        
        # Iterate through each line (starting from 1 to skip the top-level function)
        for i in range(1, lines_num):
            stripped_line = lines[i].strip('\n').strip('\t')  # Clean the line
            parts = stripped_line.split(',')  # Split the line into parts
            parts_len = len(parts)
            
            # Get the type of action point (e.g., loop, array)
            action_point_type = parts[1]

            output = []  # Initialize a list to store directives for this action point
            cnt = 0  # Directive insertion counter
            
            if action_point_type == "loop":
                # Handle loop directives
                loop_iter = int(parts[2])  # Get the number of iterations for the loop

                # PIPELINE directives
                directive = "#pragma HLS pipeline"
                output.insert(cnt, directive)  # Add pipeline directive
                cnt += 1

                directive = "#pragma HLS pipeline II=1"
                output.insert(cnt, directive)  # Add pipeline II=1 directive
                cnt += 1

                if loop_iter != -1:  # If the loop has iterations defined
                    # UNROLL directive
                    max_factor = 0
                    if loop_iter <= 64:
                        # Determine the maximum unroll factor
                        max_factor = loop_iter / 2 if (loop_iter % 2 == 0) else (loop_iter / 2) - 1

                        directive = "#pragma HLS unroll"
                        output.insert(cnt, directive)  # Add unroll directive
                        cnt += 1
                    else:
                        max_factor = 64  # Cap the unroll factor at 64

                    factor = 2
                    # Generate unroll directives with increasing factors
                    while factor <= max_factor:
                        directive = "#pragma HLS unroll factor=" + str(factor)
                        output.insert(cnt, directive)  # Add unroll factor directive
                        cnt += 1
                        factor *= 2

            elif action_point_type == "array":
                # Handle array directives
                array_name = parts[2]  # Get the array name

                # Loop through dimensions and sizes (starting from index 3)
                for i in range(3, parts_len, 2):
                    array_dim = parts[i]  # Get the array dimension
                    size = int(parts[i + 1])  # Get the size of the dimension

                    if size < array_maximum_size:
                        # If size is less than the maximum allowed, apply complete partitioning
                        directive = "#pragma HLS array_partition variable=" + array_name + " complete dim=" + array_dim
                        output.insert(cnt, directive)  # Add complete partition directive
                        cnt += 1

                    # Generate block and cyclic partitioning directives
                    for t in ['block', 'cyclic']:
                        max_factor = 0
                        if size > array_maximum_size:
                            max_factor = array_maximum_size  # Cap the partition factor to array maximum size
                        else:
                            max_factor = size / 2 if (size % 2 == 0) else (size / 2) - 1

                        factor = 2
                        # Generate partitioning directives with increasing factors
                        while factor <= max_factor:
                            directive = "#pragma HLS array_partition variable=" + array_name + " " + t + " factor=" + str(factor) + " dim=" + array_dim
                            output.insert(cnt, directive)  # Add partition directive
                            cnt += 1
                            factor *= 2

            # Add the list of directives for the current action point to the overall list
            directives_per_action_point.insert(action_point_counter, output)
            action_point_counter += 1  # Increment the action point counter
        
        # Return the list of directives per action point
        return directives_per_action_point

    def _translate_directive_list(self, directive_list, device_period):
        """
        Translates a list of directive indices to actual HLS directives based on the action point.
        It handles different directive sets for different devices and clock periods.

        Args:
            directive_list (list): List of directive indices.
            device_period (str): Device and clock period information.

        Returns:
            list: Translated list of HLS directives.
        """
        directives_per_action_point = self.directives_per_action_point_new_dse if device_period != "xczu7ev-ffvc1156-2-e_3.33" else self.directives_per_action_point_old_dse
        
        translated_directive_list = []

        for i, directive_index in enumerate(directive_list):
            max_index = len(directives_per_action_point[i]) - 1
            if directive_index <= max_index:
                directive = directives_per_action_point[i][directive_index]
            else:
                directive = directives_per_action_point[i][max_index]
                print("Out of Index: Adjusted to maximum available directive.")
                # This handles cases where directives exceed known limits for certain devices.
            translated_directive_list.append(directive)

        return translated_directive_list
    
    def _translate_array_directive(self, array_directive):
        """
        Translate an array directive into a more descriptive label.

        Args:
            array_directive (str): The array directive to translate.

        Returns:
            str: A descriptive label for the array directive.
        """
        if array_directive != "NDIR":
            parts = array_directive.split(" ")
            label = ""
            if "complete" in parts:
                dimension = parts[5].split("=")[1]
                label = f"complete_{dimension}"
            else:
                partition_type = parts[4]
                factor = parts[5].split("=")[1]
                dimension = parts[6].split("=")[1]
                label = f"{partition_type}_{factor}_{dimension}"
            
            return label
        else:
            return "NDIR"

    def _translate_loop_directive(self, loop_directive):
        """
        Translate a loop directive into a more descriptive label.

        Args:
            loop_directive (str): The loop directive to translate.

        Returns:
            str: A descriptive label for the loop directive.
        """
        if loop_directive != "NDIR":
            parts = loop_directive.split(" ")
            label = ""
            if "unroll" in parts:
                if len(parts) > 3:
                    factor = parts[3].split("=")[1]
                    label = f"unroll_{factor}"
                else:
                    label = "unroll"
            else:
                label = "pipeline_1" if len(parts) > 3 else "pipeline"

            return label
        else:
            return "NDIR"
    
    def get_directives_action_point_representation(self, directive_list, device_period):
        """
        Generates a mapping between action points and their corresponding directives 
        based on the provided directive list and device period.

        This method takes a list of directives, translates them according to the device period,
        and maps the directives to the corresponding action points. If a directive is not found
        for a particular action point, it defaults to "NDIR" (No Directive).

        Args:
            directive_list (list): A list of directive indices for each action point.
            device_period (str): The device period, used to select the appropriate directives.

        Returns:
            dict: A dictionary where the keys are action point names and the values are 
                the corresponding directives. Action points without a directive are marked as "NDIR".
        """
        
        # Translate the directive list according to the device period
        translated_directives_list = self._translate_directive_list(directive_list, device_period)
        
        # Initialize maps for label-to-directive and action point-to-directive mappings
        translated_label_directives_map = {}
        translated_action_point_directives_map = {}
        
        # Iterate through the translated directives list and create label-action mappings
        for index, directive in enumerate(translated_directives_list):
            label = "L" + str(index + 1)  # Generate label based on the index (e.g., L1, L2, ...)
            directive_label = self._translate_array_directive(directive) if "array" in directive else self._translate_loop_directive(directive)
            translated_label_directives_map[label] = directive_label  # Map label to directive
            
            # Try mapping the directive to the corresponding action point using label-action point map
            try:
                action_point = self.label_action_point_map[label]
                translated_action_point_directives_map[action_point] = directive_label
            except KeyError:
                # Continue if the label is not found in the label-action point map
                continue
        
        # Debugging: Print various maps for diagnostics
        # print(translated_directives_list)  # Translated directives list
        # print(translated_label_directives_map)   # Label-to-directive map
        # print(translated_action_point_directives_map) # Action point-to-directive map
        # print(self.label_action_point_map) # Label-to-action point map
        
        # Initialize a map for the final representation of directives per action point
        directives_action_point_representation_map = {}
        
        # Default all action points to "NDIR" (No Directive)
        for action_point_name in self.ACTION_POINT_NAMES:
            directives_action_point_representation_map[action_point_name] = "NDIR"
        
        # Update the map with the actual directives found
        for action_point_name, directive in translated_action_point_directives_map.items():
            directives_action_point_representation_map[action_point_name] = directive

        return directives_action_point_representation_map

    def apply_directives(self, input_file_path, output_file_path, actual_proposed_directives):
        """
        Applies the translated HLS directives to a source file by inserting them
        at specific line markers (L1, L2, etc.).

        Args:
            input_file_path (str): Path to the source file where directives will be applied.
            output_file_path (str): Path to save the new file with directives.
            actual_proposed_directives (dict): A dictionary of proposed directives for each action point.

        """
        with open(input_file_path, 'r') as fr, open(output_file_path, 'w') as fw:
            cnt = 1
            for line in fr:
                stripped_line = line.replace(' ', '').replace('\n', '').replace('\t', '')
                
                pattern = f'L{cnt}'  # Look for the line markers
                if pattern in stripped_line:
                    fw.write(line)
                    if pattern in actual_proposed_directives:
                        fw.write(actual_proposed_directives[pattern] + '\n')
                    cnt += 1
                else:
                    fw.write(line)
