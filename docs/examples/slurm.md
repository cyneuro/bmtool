# SLURM Tutorials

The SLURM module provides tools for managing and running simulations on SLURM-based high-performance computing clusters.

## Features

- Automate simulation job submission to SLURM clusters
- Manage simulation parameters and configurations
- Track simulation status and results
- Parallelize parameter sweeps and batch runs

The [Block Runner Tutorial](notebooks/SLURM/using_BlockRunner.ipynb) demonstrates how to manage simulations on SLURM clusters. In this notebook, you'll learn:

- How to set up simulation configurations for SLURM
- How to submit and monitor jobs
- How to parallelize parameter sweeps
- How to collect and analyze results from distributed simulations

## Basic API Usage

Here are some basic examples of how to use the SLURM module in your code:

### Block Runner

```python
from bmtool.SLURM import BlockRunner

# Initialize a block runner for a BMTK model
runner = BlockRunner(
    model_dir='/path/to/model',
    config='simulation_config.json',
    steps_per_block=10,  # Number of simulation steps per SLURM job
    total_steps=100      # Total simulation steps
)

# Submit the jobs to SLURM
runner.run()

# Check the status of submitted jobs
status = runner.check_status()
print(status)

# Collect results from completed jobs
results = runner.collect_results()
```

### Parameter Sweeps

```python
from bmtool.SLURM import ParameterSweep

# Create a parameter sweep
sweep = ParameterSweep(
    base_config='simulation_config.json',
    model_dir='/path/to/model',
    parameter_specs={
        'syn_weight': [0.001, 0.002, 0.003, 0.004],
        'conn_prob': [0.1, 0.2, 0.3],
        'input_rate': [10, 20, 30, 40, 50]
    }
)

# Generate configurations
configs = sweep.generate_configs()

# Run the parameter sweep
sweep.run(time_limit='2:00:00', memory='16G')
```

### Custom SLURM Runner

```python
from bmtool.SLURM import SLURMRunner

# Create a custom SLURM runner
runner = SLURMRunner(
    job_name='bmtk_simulation',
    partition='normal',
    nodes=1,
    cores_per_node=16,
    memory_gb=32,
    time_limit='08:00:00',
    email='user@example.com',
    email_options=['END', 'FAIL']
)

# Submit a BMTK simulation
runner.submit(
    model_dir='/path/to/model',
    config='simulation_config.json',
    modules_to_load=['neuron', 'python']
)
```

For more advanced examples and detailed usage, please refer to the Jupyter notebook tutorial above. 