import json

class seedSweep:
    def __init__(self, json_file_path, param_name):
        """
        Initializes the seedSweep instance.

        Args:
            json_file_path (str): Path to the JSON file to be updated.
            param_name (str): The name of the parameter to be modified.
        """
        self.json_file_path = json_file_path
        self.param_name = param_name

    def edit_json(self, new_value):
        """
        Updates the JSON file with a new parameter value.

        Args:
            new_value: The new value for the parameter.
        """
        with open(self.json_file_path, 'r') as f:
            data = json.load(f)
        
        data[self.param_name] = new_value
        
        with open(self.json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"JSON file '{self.json_file_path}' modified successfully with {self.param_name}={new_value}.", flush=True)
        
    
    def change_json_file_path(self,new_json_file_path):
        self.json_file_path = new_json_file_path
