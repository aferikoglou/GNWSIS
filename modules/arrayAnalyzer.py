import os
import operator

from functools import reduce

class ArrayAnalyzer:
    """
    A class to analyze array data within a dataset of applications.
    """

    def __init__(self, MODE):
        self.DATASET_DIR = os.path.join("data", "ApplicationDataset")
        self.ACTION_POINT_LABEL_MAPPING_DIR = os.path.join("data", "ApplicationAPLMapping")
        
        self.ARRAY_MAGIC_NUMBER = -1000000
        
        self.MODE = MODE
        
        # TODO: Hand-written part for CollectiveHLS
        self.ACTION_POINT_INDEX_START = 111
        
        # Retrieve the dataset array mappings when the class is initialized
        self.dataset_arrays_map, self.dataset_array_name_label_map = self._get_dataset_arrays_map()
        
        self.dataset_maximum_array_dimensions, self.dataset_maximum_application_arrays = self._analyze_dataset_arrays()
        print(f"The maximum number of array dimensions in the dataset is: {self.dataset_maximum_array_dimensions}")
        print(f"The maximum number of arrays in a single application is: {self.dataset_maximum_application_arrays}")
    
        # TODO: Hand-written part for CollectiveHLS
        self.array_action_point_names = [str(index) for index in range(1, 44 + 1)] if self.MODE == "CollectiveHLS" else self._get_array_action_point_names(self.dataset_maximum_array_dimensions, self.dataset_maximum_application_arrays)
    
        self.transformed_dataset_array_map = self._get_app_array_vectors(self.dataset_maximum_array_dimensions, self.dataset_maximum_application_arrays, self.dataset_arrays_map, self.dataset_array_name_label_map)
    
    def _get_application_array_map(self, app_name):
        """
        Retrieves array information from the 'kernel_info.txt' file for a specific application.
        
        Args:
            app_name (str): The name of the application to extract arrays from.

        Returns:
            app_arrays_map (dict): A dictionary mapping array names to their dimension sizes.
            app_name_label_map (dict): A dictionary mapping array names to their labels.
        """
        app_arrays_map = {}
        app_name_label_map = {}

        # Path to the kernel_info.txt file
        kernel_info_path = os.path.join(self.DATASET_DIR, app_name, "kernel_info.txt")

        try:
            # Open and read the kernel_info.txt file
            with open(kernel_info_path, 'r') as file:
                lines = file.readlines()

            # Parse each line in the file
            for line in lines:
                stripped_line = line.strip()
                parts = stripped_line.split(',')
                length = len(parts)

                if length > 1:
                    label = parts[0]
                    action_point_type = parts[1]
                    
                    # Check if the action point type is 'array'
                    if action_point_type == "array":
                        array_name = parts[2]
                        dimension_sizes_list = []

                        # Extract dimension sizes from the line
                        for i in range(4, length, 2):
                            dimension_size = int(parts[i])
                            dimension_sizes_list.append(dimension_size)

                        # Map array names to their dimensions and labels
                        app_arrays_map[array_name] = dimension_sizes_list
                        app_name_label_map[array_name] = label

        except FileNotFoundError:
            print(f"No kernel_info.txt file found for application: {app_name}")

        return app_arrays_map, app_name_label_map
    
    def _get_dataset_arrays_map(self):
        """
        Retrieves array information for all applications in the dataset directory.

        Returns:
            dataset_arrays_map (dict): A dictionary mapping applications to their arrays and dimensions.
            dataset_array_name_label_map (dict): A dictionary mapping applications to their array labels.
        """
        dataset_arrays_map = {}
        dataset_array_name_label_map = {}

        # Loop through all applications in the dataset directory
        for app_name in os.listdir(self.DATASET_DIR):
            # For each application, retrieve its array map and label map
            try:
                # Retrieve the array and label mappings for the current application
                app_arrays_map, app_name_label_map = self._get_application_array_map(app_name)

                # Store the mappings in the dataset-level dictionaries
                dataset_arrays_map[app_name] = app_arrays_map
                dataset_array_name_label_map[app_name] = app_name_label_map

            except Exception as e:
                # Handle any errors encountered while processing an application
                print(f"Error processing {app_name}: {e}")
        
        # Return the complete dataset mappings
        return dataset_arrays_map, dataset_array_name_label_map
    
    def _analyze_dataset_arrays(self):
        """
        Analyzes the dataset to determine the maximum array dimensions 
        and the maximum number of arrays per application.

        Returns:
            dataset_maximum_array_dimensions (int): The highest number of dimensions found in any array.
            dataset_maximum_application_arrays (int): The largest number of arrays found in any application.
        """
        dataset_maximum_array_dimensions = 0
        dataset_maximum_application_arrays = 0

        # Loop through all applications in the dataset
        for _, app_array_map in sorted(self.dataset_arrays_map.items()):
            app_array_num = len(app_array_map)  # Total number of arrays in this application
            app_max_dimensions = 0  # Maximum dimensions of arrays within the application

            # Loop through each array and analyze its dimensions
            for array_sizes_list in app_array_map.values():
                array_dimensions = len(array_sizes_list)

                # Update the maximum array dimensions for the dataset
                dataset_maximum_array_dimensions = max(dataset_maximum_array_dimensions, array_dimensions)
                
                # Track the maximum dimensions of arrays within the current application
                app_max_dimensions = max(app_max_dimensions, array_dimensions)

            # Update the maximum number of arrays per application in the dataset
            dataset_maximum_application_arrays = max(dataset_maximum_application_arrays, app_array_num)

        return dataset_maximum_array_dimensions, dataset_maximum_application_arrays
    
    def _get_array_action_point_names(self, max_array_dimensions, max_application_arrays):
        """
        Generate names for action points based on the number of arrays and their dimensions.

        Args:
            max_array_dimensions (int): The maximum number of dimensions that each array can have.
            max_application_arrays (int): The maximum number of arrays in the application.

        Returns:
            list: A list of strings, where each string is formatted as "Array_{array_num}_{array_dim}",
                representing the action point names for each array and its corresponding dimensions.
        """        
        action_point_names = [
            f"Array_{array_num}_{array_dim}"
            for array_num in range(1, max_application_arrays + 1)
            for array_dim in range(1, max_array_dimensions + 1)
        ]
        
        return action_point_names
    
    def _get_app_array_vectors(self, max_array_dimensions, max_application_arrays, arrays_map, array_name_label_map):
        """
        Transforms the dataset arrays into a uniform vector format by adding padding where necessary.
        
        Args:
            max_array_dimensions (int): Maximum dimensions across all arrays.
            max_application_arrays (int): Maximum number of arrays across all applications.
            arrays_map (dict): A dictionary mapping applications to their array sizes.
            array_name_label_map (dict): A dictionary mapping array names to labels for each application.
            
        Returns:
            dict: A dictionary mapping application names to their transformed array vectors.
        """
        transformed_arrays = {}

        for app_name, app_array_map in arrays_map.items():
            app_array_vector = []
            action_point_index = self.ACTION_POINT_INDEX_START if self.MODE == "CollectiveHLS" else 0
            labels_file_path = os.path.join(self.ACTION_POINT_LABEL_MAPPING_DIR, f"{app_name}.txt")

            with open(labels_file_path, "w") as f:
                array_name_labels = array_name_label_map[app_name]
                sorted_arrays = self._sort_arrays_by_size(app_array_map)

                for array_name, array_sizes in sorted_arrays:
                    padded_sizes = self._pad_array_sizes(array_sizes, max_array_dimensions)
                    app_array_vector.extend(padded_sizes)
                    self._write_array_metadata(f, array_name, padded_sizes, array_name_labels[array_name], action_point_index)

                    action_point_index += 1 if self.MODE == "CollectiveHLS" else 2

            app_array_vector = self._pad_application_vector(app_array_vector, max_array_dimensions, max_application_arrays)
            transformed_arrays[app_name] = app_array_vector

        return transformed_arrays

    def _sort_arrays_by_size(self, app_array_map):
        """Sort arrays by size (product of dimensions) in descending order."""
        return sorted(app_array_map.items(), key=lambda x: reduce(operator.mul, x[1], 1), reverse=True)

    def _pad_array_sizes(self, array_sizes, max_dimensions):
        """Pad the array sizes list to the maximum dimensions."""
        return array_sizes + [self.ARRAY_MAGIC_NUMBER] * (max_dimensions - len(array_sizes))

    def _write_array_metadata(self, file, array_name, padded_sizes, label, action_point_index):
        """Write array metadata to the labels file."""
        if self.MODE == "CollectiveHLS":
            file.write(f"{action_point_index},{array_name},{label},{padded_sizes}\n")
        else:
            parts = self.array_action_point_names[action_point_index].split("_")
            action_point_name = f"{parts[0]}_{parts[1]}"
            file.write(f"{action_point_name},{label}\n")

    def _pad_application_vector(self, app_array_vector, max_array_dimensions, max_application_arrays):
        """Pad the application array vector to the target length."""
        target_length = max_array_dimensions * max_application_arrays
        return app_array_vector + [self.ARRAY_MAGIC_NUMBER] * (target_length - len(app_array_vector))

    def get_app_array_vector(self, app_name):
        """
        Retrieves the transformed array vector for a given application.
        
        Args:
            app_name (str): The name of the application whose array vector is to be retrieved.
        
        Returns:
            list: The transformed array vector for the application if available.
        
        Raises:
            KeyError: If the application name is not found in the transformed dataset map.
        """
        try:            
            return self.transformed_dataset_array_map[app_name]
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset array map.")

    def get_app_array_map(self, app_name):
        """
        Retrieves the transformed array map for a given application.

        Args:
            app_name (str): The name of the application whose array map is to be retrieved.

        Returns:
            dict: A dictionary where keys are array indices (1-based) or action point names,
                and values are the corresponding array elements.

        Raises:
            KeyError: If the application name is not found in the transformed dataset array map.
        """
        transformed_array_vector = self._get_transformed_array_vector(app_name)
        return self._construct_app_array_map(transformed_array_vector)

    def _get_transformed_array_vector(self, app_name):
        """Retrieve the transformed array vector for the given application."""
        try:
            return self.transformed_dataset_array_map[app_name]
        except KeyError:
            raise KeyError(f"Application '{app_name}' not found in the transformed dataset array map.")

    def _construct_app_array_map(self, transformed_array_vector):
        """Construct the application array map based on the mode."""
        max_indices = self.dataset_maximum_application_arrays * self.dataset_maximum_array_dimensions
        
        if self.MODE == "CollectiveHLS":
            return {
                str(index + 1): transformed_array_vector[index]
                for index in range(max_indices)
            }
        else:
            return {
                self.array_action_point_names[index]: transformed_array_vector[index]
                for index in range(max_indices)
            }

    def get_array_column_names(self):
        """
        Retrieves the array action point names.

        Returns:
            list: A list of array action point names.
        """
        return self.array_action_point_names
