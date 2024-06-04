import subprocess

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
