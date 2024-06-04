import subprocess

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
            #if 'slurm_load_jobs error: Invalid job id specified' in result.stderr:
            #    return 'COMPLETED'  # Treat invalid job ID as completed because scontrol expires and removed job info when done.
            raise Exception(f"Error checking job status: {result.stderr}")

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
