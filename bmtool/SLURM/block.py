from submit import submit_job
from monitor import check_job_status
import time

import os
import subprocess

class SimulationBlock:
    def __init__(self, block_name, time, partition, nodes, ntasks, mem, simulation_cases, output_base_dir,account,additional_commands=None,
                 status_list = ['COMPLETED', 'FAILED', 'CANCELLED']):
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

        batch_script_path = os.path.join(block_output_dir, 'script.sh')
        additional_commands_str = "\n".join(self.additional_commands)

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
#SBATCH --mem={self.mem}
#SBATCH --account={self.account}

# Additional user-defined commands
{additional_commands_str}

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


class SequentialBlockRunner:
    """
    Class to handle submitting multiple blocks sequentially.

    Attributes:
        blocks (list): List of SimulationBlock instances to be run.
        json_editor (seedSweep): Instance of seedSweep to edit JSON file.
        param_values (list): List of values for the parameter to be modified.
        check_interval (int): Time interval (in seconds) to check job status needs to be less than 300 seconds for scontrol to work.
    """

    def __init__(self, blocks, json_editor=None, param_values=None, check_interval=200):
        self.blocks = blocks
        self.json_editor = json_editor
        self.param_values = param_values
        self.check_interval = check_interval

    def submit_blocks_sequentially(self):
        """
        Submits all blocks sequentially, ensuring each block starts only after the previous block has completed.
        Updates the JSON file with new parameters before each block run.
        """
        for i, block in enumerate(self.blocks):
            # Update JSON file with new parameter value
            if self.json_editor == None and self.param_values == None:
                print(f"skipping json editing for block {block.block_name}",flush=True)
            else:
                if len(self.blocks) != len(self.param_values):
                    raise Exception("Number of blocks needs to each number of params given")
                new_value = self.param_values[i]
                print(f"Updating JSON file with parameter value for block: {block.block_name}", flush=True)
                self.json_editor.edit_json(new_value)

            # Submit the block
            print(f"Submitting block: {block.block_name}", flush=True)
            block.submit_block()
            
            # Wait for the block to complete
            while not block.check_block_status():
                print(f"Waiting for block {block.block_name} to complete...", flush=True)
                time.sleep(self.check_interval)
            
            print(f"Block {block.block_name} completed.", flush=True)
        print("All blocks are done!",flush=True)

