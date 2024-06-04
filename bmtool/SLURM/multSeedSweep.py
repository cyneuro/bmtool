import json
from seedSweep import seedSweep

class multSeedSweep(seedSweep):
    """
    MultSeedSweeps are centered around some base JSON cell file. When that base JSON is updated, the other JSONs
    change according to their ratio with the base JSON.
    """
    def __init__(self, base_json_file_path, param_name,syn_dict_list=[], base_ratio=1):
        """
        Initializes the multipleSeedSweep instance. 

        Args:
            base_json_file_path (str): file path for the base json file path
            param_name (str): The name of the parameter to be modified.
            syn_dict_list (list): A list containing dictionaries with the 'json_file_path' and 'ratio'(in comparison to the base_json) for each JSON file.
            base_ratio (float): The ratio between the other JSONs; usually the current value for the parameter.
        """
        
        self.base_json_file_path = base_json_file_path
        self.syn_dict_list = syn_dict_list
        self.param_name = param_name
        self.base_ratio = base_ratio
        

    def edit_json(self, new_value):
        """
        Updates the base JSON file with a new parameter value.

        Args:
            new_value: The new value for the parameter.
        """
        with open(self.base_json_file_path, 'r') as f:
            data = json.load(f)
        
        data[self.param_name] = new_value
        
        with open(self.base_json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"JSON file '{self.base_json_file_path}' modified successfully with {self.param_name}={new_value}.", flush=True)

    def edit_all_jsons(self, new_value):
        """
        Updates the base JSON file with a new parameter value and then updates the other JSON files based on the ratio.

        Args:
            new_value: The new value for the parameter in the base JSON.
        """

        self.edit_json(new_value)  
        base_ratio = self.base_ratio
        for syn_dict in self.syn_dict_list:
            json_file_path = syn_dict['json_file_path']
            new_ratio = syn_dict['ratio'] / base_ratio
            
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            altered_value = (new_ratio) * new_value
            data[self.param_name] = altered_value
        
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=4)
        
            print(f"JSON file '{json_file_path}' modified successfully with {self.param_name}={altered_value}.", flush=True)
            

    def change_json_file_path(self,new_json_file_path):
        self.base_json_file_path = new_json_file_path
            
