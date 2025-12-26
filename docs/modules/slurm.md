# SLURM Module

The SLURM module provides utilities for running BMTK simulations on SLURM-based high-performance computing clusters. It manages job submission, parameter sweeps, and workflow automation through a YAML-based configuration system.

## Quick Start

The simplest way to use the SLURM module is through the YAML configuration workflow:

```bash
python run_simulation_w_config.py slurm_config_setup.yaml
```

This command reads a YAML configuration file that specifies:
- Which simulation configs to run
- SLURM resource requirements (time, nodes, memory, partition)
- Environment-specific module loading commands
- Optional parameter sweeps across model parameters

## Features

- **YAML-Based Workflow**: Configure entire simulation campaigns in a single YAML file
- **Parameter Sweeps**: Systematically vary model parameters with value-based or percentage-based sweeps
- **Multi-Environment Support**: Switch between local, Expanse, Hellbender, or other HPC systems
- **Component Directory Cloning**: Automatically creates isolated parameter copies for parallel execution
- **Job Monitoring**: Tracks SLURM job status and manages dependencies
- **Teams Notifications**: Optional webhook notifications for job progress updates
- **Flexible Submission**: Submit blocks sequentially or in parallel based on resource availability

## YAML Configuration

The YAML configuration file controls all aspects of the simulation workflow. Here's a complete example:

```yaml
# Simulation configuration
simulation:
  output_name: "../Run-Storage/final_runs_SST2PV_+25%"  # Output directory
  environment: "local"                     # Execution environment (local, expanse, hellbender)
  seed_sweep: 'none'                       # Parameter sweep type (none, seed, multiseed)
  component_path: "components"             # Base path for simulation components
  parallel_submission: true                # Submit all jobs at once (true) or sequentially (false)
  use_coreneuron: false                    # Whether to use CoreNeuron backend
  webhook_url: "https://..."               # Microsoft Teams webhook URL for notifications

# SLURM configurations for different environments
slurm:
  local:
    time: "08:00:00"       # Maximum runtime (HH:MM:SS)
    partition: "batch"     # SLURM partition name
    nodes: 1               # Number of nodes
    ntasks: 40             # Number of tasks (cores)
    mem: 80               # Total memory in GB

  expanse:
    time: "02:30:00"
    partition: "shared"
    nodes: 1
    ntasks: 120
    mem: 240
    account: "umc113"      # SLURM account (required for some HPC systems)

  hellbender:
    time: "02:30:00"
    partition: "general"
    nodes: 1
    ntasks: 30
    mem: 60

# Module loading commands for different environments
module_commands:
  expanse:
    - "module purge"
    - "module load slurm"
    - "module load cpu/0.17.3b"
    - "module load gcc/10.2.0/npcyll4"
    - "module load openmpi/4.1.1"
    - "export HDF5_USE_FILE_LOCKING=FALSE"
  
  hellbender:
    - "module load intel_mpi"
    - "module load gcc/12.2.1"
    - "export HDF5_USE_FILE_LOCKING=FALSE"
  
  local:
    - "module load mpich-x86_64-nopy"

# Simulation cases configuration
simulation_cases:
  baseline:
    config_file: "simulation_config_baseline.json"  
  short_1s:
    config_file: "simulation_config_short.json"   
  long_1s:
    config_file: "simulation_config_long.json"

# Parameter sweep configuration (only used if seed_sweep is not 'none')
sweep_config:
  param_name: "initW"              # Parameter name in JSON file to modify
  sweep_method: "value"            # "value" for explicit values, "percentage" for incremental changes
  
  # Option 1: Specify exact values to use
  param_values: [2, 3, 4, 5, 6, 7]
  
  # Option 2: Use percentage-based changes (alternative to param_values)
  # base_value: 5.0                # Starting parameter value
  # percent_change: 10             # Percentage increase per iteration
  # iterations: 2                  # Number of iterations
  
  base_json_file: "synaptic_models/synapses_final/SST2PV.json"  # JSON file to modify (path after components/)
  
  # For multiseed sweeps: specify related files that change proportionally
  multiseed_files:
    - path: "synaptic_models/synapses_final/Pulse2IT.json"
      ratio: 1.28  # Ratio relative to base parameter
```

### Configuration Sections Explained

#### `simulation`
- **`output_name`**: Directory where results will be stored (can be relative or absolute)
- **`environment`**: Which SLURM config to use (`local`, `expanse`, `hellbender`, or add your own)
- **`seed_sweep`**: Type of parameter sweep
  - `'none'`: Run simulations without parameter variations
  - `'seed'`: Vary a single JSON parameter across blocks
  - `'multiseed'`: Vary multiple related JSON parameters with fixed ratios
- **`component_path`**: Base directory containing simulation components (mechanisms, synaptic models, etc.)
- **`parallel_submission`**: 
  - `true`: Submit all blocks at once (faster but uses more resources)
  - `false`: Submit blocks sequentially (one completes before next starts)
- **`use_coreneuron`**: Enable CoreNEURON for faster simulations
- **`webhook_url`**: Microsoft Teams webhook for job status notifications

#### `slurm`
Define SLURM parameters for each environment. The active environment is selected via `simulation.environment`.

- **`time`**: Job time limit in `HH:MM:SS` format
- **`partition`**: SLURM partition/queue name
- **`nodes`**: Number of compute nodes
- **`ntasks`**: Total number of MPI tasks (typically number of cores)
- **`mem`**: Total memory in GB (automatically divided across tasks as `--mem-per-cpu`)
- **`account`**: SLURM account to charge (optional, required on some HPC systems)

#### `module_commands`
Shell commands executed before running simulations. Typically used to:
- Load required modules (MPI, compilers, Python)
- Set environment variables
- Purge conflicting modules

#### `simulation_cases`
Dictionary of simulation cases to run. Each case specifies:
- **`config_file`**: Path to BMTK simulation config JSON file

All cases run for each parameter value in a sweep, or once if `seed_sweep: 'none'`.

#### `sweep_config`
Only used when `seed_sweep` is `'seed'` or `'multiseed'`.

- **`param_name`**: JSON key to modify (e.g., `"initW"`, `"tau1"`)
- **`sweep_method`**: How to generate parameter values
  - `"value"`: Use explicit values from `param_values`
  - `"percentage"`: Calculate values from `base_value` and `percent_change`
- **`param_values`**: List of explicit parameter values (for `sweep_method: "value"`)
- **`base_value`**, **`percent_change`**, **`iterations`**: For percentage-based sweeps
- **`base_json_file`**: Path to JSON file to modify (relative to `component_path`)
- **`multiseed_files`**: (For `seed_sweep: 'multiseed'`) List of related JSON files
  - **`path`**: Path to related JSON file
  - **`ratio`**: Scaling ratio relative to base parameter value

## Core Classes

The SLURM module provides several classes for direct Python usage. While most users should use the YAML workflow, these classes can be used for custom workflows.

### SimulationBlock

Represents a single SLURM job block containing one or more simulation cases.

```python
from bmtool.SLURM import SimulationBlock

block = SimulationBlock(
    block_name="block1",
    time="02:00:00",
    partition="shared",
    nodes=1,
    ntasks=40,
    mem=80,  # GB total memory
    simulation_cases={
        "baseline": "mpirun nrniv -mpi -python run_network.py simulation_config.json",
        "long": "mpirun nrniv -mpi -python run_network.py simulation_config_long.json"
    },
    output_base_dir="/path/to/output",
    account="myaccount",  # Optional
    additional_commands=[  # Optional
        "module load neuron",
        "export HDF5_USE_FILE_LOCKING=FALSE"
    ],
    status_list=["COMPLETED", "FAILED", "CANCELLED"],  # Job states to wait for
    component_path="components"  # Optional
)

# Submit the block to SLURM
block.submit_block()

# Check if all jobs in block are complete
is_done = block.check_block_status()
```

**Key Methods:**
- **`submit_block()`**: Creates batch scripts and submits all simulation cases as SLURM jobs
- **`check_block_status()`**: Returns `True` if all jobs have reached a state in `status_list`
- **`check_block_completed()`**: Returns `True` only if all jobs are `COMPLETED`
- **`check_block_running()`**: Returns `True` if all jobs are currently `RUNNING`
- **`create_batch_script(case_name, command)`**: Generates SLURM batch script for a case

### BlockRunner

Manages multiple `SimulationBlock` instances and coordinates parameter sweeps.

```python
from bmtool.SLURM import BlockRunner, seedSweep

# Create multiple blocks
blocks = [
    SimulationBlock("block1", time="02:00:00", ...),
    SimulationBlock("block2", time="02:00:00", ...),
    SimulationBlock("block3", time="02:00:00", ...)
]

# Create runner with parameter sweep
runner = BlockRunner(
    blocks=blocks,
    json_file_path="synaptic_models/synapses_final/SST2PV.json",  # Relative to component_path
    param_name="initW",
    param_values=[2.0, 3.0, 4.0],
    check_interval=60,  # Seconds between status checks
    webhook="https://..."  # Optional Teams webhook
)

# Submit all blocks in parallel
runner.submit_blocks_parallel()

# Or submit sequentially (each waits for previous to complete)
# runner.submit_blocks_sequentially()
```

**Key Methods:**
- **`submit_blocks_sequentially()`**: Submits blocks one at a time, waiting for each to complete
- **`submit_blocks_parallel()`**: Submits all blocks at once for simultaneous execution
- **`restore_component_paths()`**: Restores original component directory paths after cloning

**Parameters:**
- **`blocks`**: List of `SimulationBlock` instances
- **`json_file_path`**: Path to JSON file to modify (relative to `component_path`)
- **`param_name`**: Parameter name to modify in JSON
- **`param_values`**: List of parameter values (one per block)
- **`check_interval`**: Seconds to wait between SLURM status checks
- **`syn_dict`**: Dictionary for multiseed sweeps `{"json_file_path": "...", "ratio": 1.28}`
- **`webhook`**: Microsoft Teams webhook URL for notifications

### seedSweep

Edits a single parameter in a JSON file for parameter sweeps.

```python
from bmtool.SLURM import seedSweep

editor = seedSweep(
    json_file_path="components/synaptic_models/SST2PV.json",
    param_name="initW"
)

# Update the parameter value
editor.edit_json(5.0)

# Change to a different JSON file
editor.change_json_file_path("components/synaptic_models/Pulse2IT.json")
editor.edit_json(3.5)
```

**Methods:**
- **`edit_json(new_value)`**: Updates the JSON file with the new parameter value
- **`change_json_file_path(new_json_file_path)`**: Points the editor to a different JSON file

### multiSeedSweep

Extends `seedSweep` to modify multiple related JSON files with proportional scaling.

```python
from bmtool.SLURM import multiSeedSweep

editor = multiSeedSweep(
    base_json_file_path="components/synaptic_models/SST2PV.json",
    param_name="initW",
    syn_dict={
        "json_file_path": "components/synaptic_models/Pulse2IT.json",
        "ratio": 1.28  # Pulse2IT.initW = SST2PV.initW * 1.28
    },
    base_ratio=1
)

# Update base JSON to 5.0 and related JSON to 5.0 * 1.28 = 6.4
editor.edit_all_jsons(5.0)
```

**Methods:**
- **`edit_all_jsons(new_value)`**: Updates base JSON and scales related JSON proportionally

## Workflow Explanation

Understanding the complete workflow helps debug issues and customize behavior.

### 1. Configuration Loading
`run_simulation_w_config.py` parses the YAML file and extracts:
- SLURM resource requirements
- Environment-specific commands
- Simulation cases to run
- Parameter sweep settings

### 2. Parameter Value Generation
Based on `sweep_config.sweep_method`:
- **`"value"`**: Uses `param_values` directly
- **`"percentage"`**: Calculates values: `[base_value, base_value * (1 + percent/100), ...]`

### 3. SimulationBlock Creation
One `SimulationBlock` is created for each parameter value (or just one if `seed_sweep: 'none'`).

Each block contains all simulation cases from `simulation_cases`.

### 4. Component Directory Cloning
For parameter sweeps, `BlockRunner` clones the component directory for each block:
- `components` → `components1`, `components2`, `components3`, ...
- Each clone gets its own parameter value edited in the JSON files
- Prevents conflicts during parallel execution

### 5. JSON Parameter Editing
For each cloned directory:
- **seed sweep**: Edits `base_json_file` using `seedSweep`
- **multiseed sweep**: Edits base file and related files using `multiSeedSweep`
- All edits happen in the cloned directory (e.g., `components1/...`)

### 6. Batch Script Generation
For each simulation case in each block, a SLURM batch script is created with:
- SLURM resource directives (`--time`, `--partition`, `--nodes`, `--ntasks`, `--mem-per-cpu`)
- Module loading commands
- Environment variables: `COMPONENT_PATH` and `OUTPUT_DIR`
- Simulation command

### 7. SLURM Submission
Scripts are submitted via `sbatch`. Depending on `parallel_submission`:
- **Parallel**: All blocks submitted immediately
- **Sequential**: Each block waits for previous to reach a state in `status_list`

### 8. Job Monitoring
`BlockRunner` periodically checks job status using `scontrol show job`. When all jobs complete:
- Original component paths are restored
- Teams notification sent (if webhook configured)

### 9. Simulation Execution
Each SLURM job runs `run_network.py` which:
- Reads `COMPONENT_PATH` environment variable
- Temporarily updates network config to use the cloned component directory
- Runs BMTK simulation
- Saves synaptic parameters report
- Copies component directory to output for reproducibility
- Restores original network config

## Examples

### Example 1: Single Simulation (No Parameter Sweep)

```yaml
simulation:
  output_name: "../Run-Storage/single_run"
  environment: "local"
  seed_sweep: 'none'
  component_path: "components"
  parallel_submission: false
  use_coreneuron: false

slurm:
  local:
    time: "04:00:00"
    partition: "batch"
    nodes: 1
    ntasks: 40
    mem: 80

module_commands:
  local:
    - "module load mpich-x86_64-nopy"

simulation_cases:
  baseline:
    config_file: "simulation_config_baseline.json"
```

Run with:
```bash
python run_simulation_w_config.py slurm_config_setup.yaml
```

### Example 2: Value-Based Parameter Sweep

```yaml
simulation:
  output_name: "../Run-Storage/sweep_initW"
  environment: "expanse"
  seed_sweep: 'seed'
  component_path: "components"
  parallel_submission: true
  use_coreneuron: false

slurm:
  expanse:
    time: "02:00:00"
    partition: "shared"
    nodes: 1
    ntasks: 120
    mem: 240
    account: "umc113"

module_commands:
  expanse:
    - "module purge"
    - "module load cpu/0.17.3b"
    - "module load gcc/10.2.0/npcyll4"
    - "module load openmpi/4.1.1"

simulation_cases:
  baseline:
    config_file: "simulation_config_baseline.json"

sweep_config:
  param_name: "initW"
  sweep_method: "value"
  param_values: [2.0, 3.0, 4.0, 5.0, 6.0]
  base_json_file: "synaptic_models/synapses_final/SST2PV.json"
```

This creates 5 blocks, each with a different `initW` value in `SST2PV.json`.

### Example 3: Percentage-Based Parameter Sweep

```yaml
sweep_config:
  param_name: "tau1"
  sweep_method: "percentage"
  base_value: 10.0
  percent_change: 25
  iterations: 3
  base_json_file: "synaptic_models/synapses_final/SST2PV.json"
```

This generates parameter values:
- Block 1: `10.0`
- Block 2: `10.0 * 1.25 = 12.5`
- Block 3: `12.5 * 1.25 = 15.625`
- Block 4: `15.625 * 1.25 = 19.53`

### Example 4: Multi-Seed Sweep with Proportional Scaling

```yaml
simulation:
  seed_sweep: 'multiseed'
  # ... other settings ...

sweep_config:
  param_name: "initW"
  sweep_method: "value"
  param_values: [2.0, 4.0, 6.0]
  base_json_file: "synaptic_models/synapses_final/SST2PV.json"
  multiseed_files:
    - path: "synaptic_models/synapses_final/Pulse2IT.json"
      ratio: 1.28
```

This modifies both files in each block:
- Block 1: `SST2PV.initW = 2.0`, `Pulse2IT.initW = 2.0 * 1.28 = 2.56`
- Block 2: `SST2PV.initW = 4.0`, `Pulse2IT.initW = 4.0 * 1.28 = 5.12`
- Block 3: `SST2PV.initW = 6.0`, `Pulse2IT.initW = 6.0 * 1.28 = 7.68`

### Example 5: Multiple Simulation Cases

```yaml
simulation_cases:
  baseline:
    config_file: "simulation_config_baseline.json"
  short:
    config_file: "simulation_config_short.json"
  long:
    config_file: "simulation_config_long.json"

sweep_config:
  param_values: [2.0, 4.0]
  # ... other sweep settings ...
```

Each block runs all three cases:
- Block 1 (param=2.0): runs baseline, short, and long configs
- Block 2 (param=4.0): runs baseline, short, and long configs

Total: 6 SLURM jobs (2 blocks × 3 cases)

## Advanced Features

### Microsoft Teams Notifications

Configure webhook notifications to track simulation progress:

```yaml
simulation:
  webhook_url: "https://prod-XX.westus.logic.azure.com/workflows/..."
```

Notifications are sent when:
- Each block is submitted
- All simulations complete

### Parallel vs Sequential Submission

**Parallel submission** (`parallel_submission: true`):
- All blocks submitted immediately to SLURM queue
- Faster overall completion if resources available
- Requires more disk space (all component directories cloned at once)
- Higher queue usage

**Sequential submission** (`parallel_submission: false`):
- Each block waits for previous to reach `status_list` states before submitting
- Lower resource usage at any given time
- Useful when queue has strict limits
- Default `status_list: ["COMPLETED", "FAILED", "CANCELLED"]` waits for full completion

### Custom Status Lists

Control when the next block submits by customizing `status_list` in `SimulationBlock`:

```python
# Wait for completion before next block
status_list=["COMPLETED", "FAILED", "CANCELLED"]

# Submit next block as soon as current starts running (aggressive)
status_list=["RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
```

### Component Path Cloning

The component cloning mechanism:
1. Original directory: `components/`
2. Cloned directories: `components1/`, `components2/`, `components3/`, ...
3. Each clone has independent parameter values
4. Original directory restored after completion

This allows parallel sweeps without file conflicts.

### Globus File Transfer

For transferring large result datasets between HPC systems:

```python
from bmtool.SLURM import globus_transfer

globus_transfer(
    source_endpoint="endpoint-uuid-1",
    dest_endpoint="endpoint-uuid-2",
    source_path="/path/on/source",
    dest_path="/path/on/dest"
)
```

## Utility Functions

```python
from bmtool.SLURM import check_job_status, submit_job, send_teams_message

# Check SLURM job status
status = check_job_status("12345678")  # Returns "RUNNING", "COMPLETED", etc.

# Submit a batch script
job_id = submit_job("path/to/script.sh")

# Send Teams notification
send_teams_message(
    webhook="https://...",
    message="Simulation started!"
)
```
