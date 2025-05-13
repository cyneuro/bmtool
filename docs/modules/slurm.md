# SLURM Module

The SLURM module provides utilities for running BMTK simulations on SLURM-based high-performance computing clusters. It simplifies the process of submitting jobs, running parameter sweeps, and managing simulation workflows.

## Features

- **Job Submission**: Simplify SLURM job submission for BMTK models
- **Parameter Sweeps**: Vary model parameters systematically across simulations
- **Job Management**: Monitor and manage running jobs
- **Result Collection**: Gather results from multiple simulations

## Block Runner

The BlockRunner is a core component that manages simulation workflows:

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

## Parameter Sweeps

The SLURM module supports parameter sweeps to explore model behavior:

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

## Job Customization

Customize SLURM job parameters for specific requirements:

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

## Advanced Features

### Custom Job Arrays

Create job arrays for parameter variations:

```python
from bmtool.SLURM import JobArray

array = JobArray(
    base_config='simulation_config.json',
    model_dir='/path/to/model',
    parameter_variations=[
        {'input_rate': 10, 'conn_prob': 0.1},
        {'input_rate': 20, 'conn_prob': 0.1},
        {'input_rate': 30, 'conn_prob': 0.2}
    ],
    array_size=3
)

array.submit()
```

### Result Analysis

Analyze results from parameter sweeps:

```python
from bmtool.SLURM import SweepAnalyzer

analyzer = SweepAnalyzer(sweep_dir='/path/to/sweep/results')
summary = analyzer.summarize()
analyzer.plot_parameter_effects('input_rate', 'mean_firing_rate')
```

## Command Line Interface

The SLURM module can also be accessed through the command line:

```bash
# Create a new parameter sweep
bmtool util slurm sweep-create --config simulation_config.json --param syn_weight 0.001 0.002 0.003

# Submit a sweep to SLURM
bmtool util slurm sweep-run --sweep-dir sweeps/sweep_001

# Check status of running jobs
bmtool util slurm status
```
