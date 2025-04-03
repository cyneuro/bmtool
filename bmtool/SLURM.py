import time
import os
import subprocess
import json
import requests
import shutil
import time
import copy
import numpy as np


def check_job_status(job_id):
    """
    Checks the status of a SLURM job using scontrol.

    Args:
        job_id (str): The SLURM job ID.

    Returns:
        str: The state of the job.
    """
    try:
        result = subprocess.run(['scontrol', 'show', 'job', job_id], capture_output=True, text=True)
        if result.returncode != 0:
            # this check is not needed if check_interval is less than 5 min (~300 seconds)
            if 'slurm_load_jobs error: Invalid job id specified' in result.stderr:
                return 'COMPLETED'  # Treat invalid job ID as completed because scontrol expires and removed job info when done.
            #raise Exception(f"Error checking job status: {result.stderr}")

        job_state = None
        for line in result.stdout.split('\n'):
            if 'JobState=' in line:
                job_state = line.strip().split('JobState=')[1].split()[0]
                break

        if job_state is None:
            raise Exception(f"Failed to retrieve job status for job ID: {job_id}")

        return job_state
    except Exception as e:
        print(f"Exception while checking job status: {e}", flush=True)
        return 'UNKNOWN'


def submit_job(script_path):
    """
    Submits a SLURM job script.

    Args:
        script_path (str): The path to the SLURM job script.

    Returns:
        str: The job ID of the submitted job.

    Raises:
        Exception: If there is an error in submitting the job.
    """
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error submitting job: {result.stderr}")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def send_teams_message(webhook,message):
    """Sends a message to a teams channel or chat

    Args:
        webhook (str): A microsoft teams webhook
        message (str): A message to send in the chat/channel
    """
    message = {
        "text": f"{message}" 
    }

    # Send POST request to trigger the flow
    response = requests.post(
        webhook,
        json=message,  # Using 'json' instead of 'data' for automatic serialization
        headers={'Content-Type': 'application/json'}
    )


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


# class could just be added to seedSweep but for now will make new class since it was easier
class multiSeedSweep(seedSweep):
    """
    MultSeedSweeps are centered around some base JSON cell file. When that base JSON is updated, the other JSONs
    change according to their ratio with the base JSON.
    """
    def __init__(self, base_json_file_path, param_name, syn_dict, base_ratio=1):
        """
        Initializes the multipleSeedSweep instance.

        Args:
            base_json_file_path (str): File path for the base JSON file.
            param_name (str): The name of the parameter to be modified.
            syn_dict_list (list): A list containing dictionaries with the 'json_file_path' and 'ratio' (in comparison to the base_json) for each JSON file.
            base_ratio (float): The ratio between the other JSONs; usually the current value for the parameter.
        """
        super().__init__(base_json_file_path, param_name)
        self.syn_dict_for_multi = syn_dict
        self.base_ratio = base_ratio

    def edit_all_jsons(self, new_value):
        """
        Updates the base JSON file with a new parameter value and then updates the other JSON files based on the ratio.

        Args:
            new_value: The new value for the parameter in the base JSON.
        """
        self.edit_json(new_value)
        base_ratio = self.base_ratio

        json_file_path = self.syn_dict_for_multi['json_file_path']
        new_ratio = self.syn_dict_for_multi['ratio'] / base_ratio
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        altered_value = new_ratio * new_value
        data[self.param_name] = altered_value
    
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
    
        print(f"JSON file '{json_file_path}' modified successfully with {self.param_name}={altered_value}.", flush=True)


class SimulationBlock:
    def __init__(self, block_name, time, partition, nodes, ntasks, mem, simulation_cases, output_base_dir,account=None,additional_commands=None,status_list = ['COMPLETED', 'FAILED', 'CANCELLED'],component_path=None):
        """
        Initializes the SimulationBlock instance.

        Args:
            block_name (str): Name of the block.
            time (str): Time limit for the job.
            partition (str): Partition to submit the job to.
            nodes (int): Number of nodes to request.
            ntasks (int): Number of tasks.
            mem (int) : Number of gigabytes (per node)
            simulation_cases (dict): Dictionary of simulation cases with their commands.
            output_base_dir (str): Base directory for the output files.
            account (str) : account to charge on HPC 
            additional commands (list): commands to run before bmtk model starts useful for loading modules
            status_list (list): List of things to check before running next block. 
                Adding RUNNING runs blocks faster but uses MUCH more resources and is only recommended on large HPC 
        """
        self.block_name = block_name
        self.time = time
        self.partition = partition
        self.nodes = nodes
        self.ntasks = ntasks
        self.mem = mem
        self.simulation_cases = simulation_cases
        self.output_base_dir = output_base_dir
        self.account = account 
        self.additional_commands = additional_commands if additional_commands is not None else []
        self.status_list = status_list
        self.job_ids = []
        self.component_path = component_path

    def create_batch_script(self, case_name, command):
        """
        Creates a SLURM batch script for the given simulation case.

        Args:
            case_name (str): Name of the simulation case.
            command (str): Command to run the simulation.
        
        Returns:
            str: Path to the batch script file.
        """
        block_output_dir = os.path.join(self.output_base_dir, self.block_name)  # Create block-specific output folder
        case_output_dir = os.path.join(block_output_dir, case_name)  # Create case-specific output folder
        os.makedirs(case_output_dir, exist_ok=True)

        batch_script_path = os.path.join(block_output_dir, f'{case_name}_script.sh')
        additional_commands_str = "\n".join(self.additional_commands)
        # Conditional account linegit
        account_line = f"#SBATCH --account={self.account}\n" if self.account else ""
        env_var_component_path = f"export COMPONENT_PATH={self.component_path}" if self.component_path else ""
        mem_per_cpu = int(np.ceil(int(self.mem)/int(self.ntasks))) # do ceil cause more mem is always better then less

        # Write the batch script to the file
        with open(batch_script_path, 'w') as script_file:
            script_file.write(f"""#!/bin/bash
#SBATCH --job-name={self.block_name}_{case_name}
#SBATCH --output={block_output_dir}/%x_%j.out
#SBATCH --error={block_output_dir}/%x_%j.err
#SBATCH --time={self.time}
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --mem-per-cpu={mem_per_cpu}G
{account_line}

# Additional user-defined commands
{additional_commands_str}

#enviroment vars
{env_var_component_path}

export OUTPUT_DIR={case_output_dir}

{command}
""")

        #print(f"Batch script created: {batch_script_path}", flush=True)

        return batch_script_path

    def submit_block(self):
        """
        Submits all simulation cases in the block as separate SLURM jobs.
        """
        for case_name, command in self.simulation_cases.items():
            script_path = self.create_batch_script(case_name, command)
            result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                self.job_ids.append(job_id)
                print(f"Submitted {case_name} with job ID {job_id}", flush=True)
            else:
                print(f"Failed to submit {case_name}: {result.stderr}", flush=True)

    def check_block_status(self):
        """
        Checks the status of all jobs in the block.

        Returns:
            bool: True if all jobs in the block are completed, False otherwise.
        """
        for job_id in self.job_ids:
            status = check_job_status(job_id)
            if status not in self.status_list:
                return False
        return True
    
    def check_block_completed(self):
        """checks if all the jobs in the block have been completed by slurm

        Returns:
            bool: True if all block jobs have been ran, false if job is still running
        """
        for job_id in self.job_ids:
            status = check_job_status(job_id)
            #print(f"status of job is {status}")
            if status != 'COMPLETED': # can add PENDING here for debugging NOT FOR ACTUALLY USING IT 
                return False
        return True

    def check_block_running(self):
        """checks if a job is running

        Returns:
            bool: True if jobs are RUNNING false if anything else
        """
        for job_id in self.job_ids:
            status = check_job_status(job_id)
            if status != 'RUNNING': # 
                return False
        return True
    
    def check_block_submited(self):
        """checks if a job is running

        Returns:
            bool: True if jobs are RUNNING false if anything else
        """
        for job_id in self.job_ids:
            status = check_job_status(job_id)
            if status != 'PENDING': # 
                return False
        return True


def get_relative_path(endpoint, absolute_path):
    """Convert absolute path to relative path for Globus transfer."""
    try:
        # Get the directories at the mount point
        result = subprocess.run(["globus", "ls", f"{endpoint}:/"], capture_output=True, text=True, check=True)
        dirs = set(result.stdout.splitlines())  # Convert to a set for quicker lookup

        # Split the absolute path into parts
        path_parts = absolute_path.strip("/").split("/")

        # Find the first matching directory in the list
        for i, part in enumerate(path_parts):
            if part+"/" in dirs:
                # The mount point is everything up to and including this directory
                mount_point = "/" + "/".join(path_parts[:i])
                relative_path = absolute_path.replace(mount_point, "", 1).lstrip("/")
                return relative_path
        
        print("Error: Could not determine relative path.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving directories from Globus: {e}")
        return None

def globus_transfer(source_endpoint, dest_endpoint, source_path, dest_path):
    """
    Transfers file using custom globus transfer function. 
    For more info see https://github.com/GregGlickert/transfer-files/blob/main/globus_transfer.sh
    work in progress still... kinda forgot about this 
    """
    relative_source_path = get_relative_path(source_endpoint, source_path)
    if relative_source_path is None:
        print("Transfer aborted: Could not determine relative source path.")
        return
    
    command = f"globus transfer {source_endpoint}:{relative_source_path} {dest_endpoint}:{dest_path} --label 'bmtool slurm transfer'"
    os.system(command)



class BlockRunner:
    """
    Class to handle submitting multiple blocks sequentially.

    Attributes:
        blocks (list): List of SimulationBlock instances to be run.
        json_editor (seedSweep or multiSweep): Instance of seedSweep to edit JSON file.
        param_values (list): List of values for the parameter to be modified.
        webhook (str): a microsoft webhook for teams. When used will send teams messages to the hook!
    """

    def __init__(self, blocks, json_editor=None,json_file_path=None, param_name=None, 
                 param_values=None, check_interval=60,syn_dict = None,
                 webhook=None):
        self.blocks = blocks
        self.json_editor = json_editor
        self.param_values = param_values
        self.check_interval = check_interval
        self.webhook = webhook
        self.param_name = param_name
        self.json_file_path = json_file_path
        self.syn_dict = syn_dict

    def submit_blocks_sequentially(self):
        """
        Submits all blocks sequentially, ensuring each block starts only after the previous block has completed or is running.
        Updates the JSON file with new parameters before each block run.
        """
        for i, block in enumerate(self.blocks):
            print(block.output_base_dir)
            # Update JSON file with new parameter value
            if self.json_file_path == None and self.param_values == None:
                source_dir = block.component_path
                destination_dir = f"{source_dir}{i+1}"
                block.component_path = destination_dir
                shutil.copytree(source_dir, destination_dir,dirs_exist_ok = True) # create new components folder 
                print(f"skipping json editing for block {block.block_name}",flush=True)
            else:
                if len(self.blocks) != len(self.param_values):
                    raise Exception("Number of blocks needs to each number of params given")
                new_value = self.param_values[i]
                # hope this path is correct
                source_dir = block.component_path
                destination_dir = f"{source_dir}{i+1}"
                block.component_path = destination_dir

                shutil.copytree(source_dir, destination_dir,dirs_exist_ok = True) # create new components folder
                json_file_path = os.path.join(destination_dir,self.json_file_path)
                
                if self.syn_dict == None:
                    json_editor = seedSweep(json_file_path , self.param_name)
                    json_editor.edit_json(new_value)
                else:
                    # need to keep the orignal around
                    syn_dict_temp = copy.deepcopy(self.syn_dict)
                    json_to_be_ratioed = syn_dict_temp['json_file_path']
                    corrected_ratio_path = os.path.join(destination_dir,json_to_be_ratioed)
                    syn_dict_temp['json_file_path'] = corrected_ratio_path
                    json_editor = multiSeedSweep(json_file_path ,self.param_name,
                                                syn_dict=syn_dict_temp,base_ratio=1)
                    json_editor.edit_all_jsons(new_value) 

            # Submit the block
            print(f"Submitting block: {block.block_name}", flush=True)
            block.submit_block()
            if self.webhook:
                message = f"SIMULATION UPDATE: Block {i} has been submitted! There are {(len(self.blocks)-1)-i} left to be submitted"
                send_teams_message(self.webhook,message)

            # Wait for the block to complete
            if i == len(self.blocks) - 1:  
                while not block.check_block_status():
                    print(f"Waiting for the last block {i} to complete...")
                    time.sleep(self.check_interval)
            else:  # Not the last block so if job is running lets start a new one (checks status list)
                while not block.check_block_status():
                    print(f"Waiting for block {i} to complete...")
                    time.sleep(self.check_interval)
            
            print(f"Block {block.block_name} completed.", flush=True)
        print("All blocks are done!",flush=True)
        if self.webhook:
            message = "SIMULATION UPDATE: Simulation are Done!"
            send_teams_message(self.webhook,message)

    def submit_blocks_parallel(self):
        """
        submits all the blocks at once onto the queue. To do this the components dir will be cloned and each block will have its own.
        Also the json_file_path should be the path after the components dir
        """
        for i, block in enumerate(self.blocks):
            if self.param_values == None:
                source_dir = block.component_path
                destination_dir = f"{source_dir}{i+1}"
                block.component_path = destination_dir
                shutil.copytree(source_dir, destination_dir,dirs_exist_ok = True) # create new components folder 
                print(f"skipping json editing for block {block.block_name}",flush=True)
            else:
                if block.component_path == None:
                    raise Exception("Unable to use parallel submitter without defining the component path")
                new_value = self.param_values[i]
                
                source_dir = block.component_path
                destination_dir = f"{source_dir}{i+1}"
                block.component_path = destination_dir

                shutil.copytree(source_dir, destination_dir,dirs_exist_ok = True) # create new components folder 
                json_file_path = os.path.join(destination_dir,self.json_file_path)
                
                if self.syn_dict == None:
                    json_editor = seedSweep(json_file_path , self.param_name)
                    json_editor.edit_json(new_value)
                else:
                    # need to keep the orignal around
                    syn_dict_temp = copy.deepcopy(self.syn_dict)
                    json_to_be_ratioed = syn_dict_temp['json_file_path']
                    corrected_ratio_path = os.path.join(destination_dir,json_to_be_ratioed)
                    syn_dict_temp['json_file_path'] = corrected_ratio_path
                    json_editor = multiSeedSweep(json_file_path ,self.param_name,
                                                syn_dict_temp,base_ratio=1)
                    json_editor.edit_all_jsons(new_value) 
                # submit block with new component path 
            print(f"Submitting block: {block.block_name}", flush=True)
            block.submit_block()
            if i == len(self.blocks) - 1:
                print("\nEverything has been submitted. You can close out of this or keep this script running to get a message when everything is finished\n")
                while not block.check_block_status():
                    print(f"Waiting for the last block {i} to complete...")
                    time.sleep(self.check_interval)
                
        if self.webhook:
            message = "SIMULATION UPDATE: Simulations are Done!"
            send_teams_message(self.webhook,message)
            
            
