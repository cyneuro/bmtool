# Stimulus Module

## Overview

The **Stimulus** module provides a unified interface for generating time-varying Poisson spike trains for BMTK networks. Create node assemblies (subsets of your network) using multiple strategies, then generate realistic stimulus patterns with precise temporal control.

## Features

- **Assembly Creation**: Group nodes using three methods:
  - **Random**: Randomly distribute nodes across assemblies
  - **Grid**: Group nodes by spatial location (x, y coordinates)
  - **Property-based**: Group nodes by any attribute (e.g., cell type, pulse_group_id)

- **Firing Rate Patterns**: Six configurable temporal patterns for stimulus delivery:
  - **`'long'`**: Contiguous bursts, one assembly active per cycle
  - **`'short'`**: Sequential bursts delivered to each assembly within each cycle
  - **`'ramp'`**: Linear firing rate increase/decrease with customizable slopes
  - **`'join'`**: Gradual recruitment of neurons with substep control
  - **`'fade'`**: Smooth fade transitions between paired assemblies
  - **`'loop'`**: Cycling patterns with configurable on-times per cycle

- **Background Activity**: Generate baseline and shell (population-specific) activity with optional lognormal/normal distributions

- **SONATA Format**: All outputs are BMTK-compatible SONATA HDF5 files ready for simulation

## Getting Started

### Initialize the StimulusBuilder

```python
from bmtool.stimulus.core import StimulusBuilder

# Load from BMTK config file
sb = StimulusBuilder(config='path/to/config.json', net_seed=123, psg_seed=1)
```

The `StimulusBuilder` loads your network configuration and is ready to create assemblies and generate stimuli.

### Create Node Assemblies

Define groups of nodes using one of three methods:

**Random Assembly**:
```python
sb.create_assemblies(
    name='random_groups',
    network_name='my_network',
    method='random',
    n_assemblies=5,
    prob_in_assembly=1.0
)
```

**Property-Based Assembly** (group by a node attribute):
```python
sb.create_assemblies(
    name='cell_type_groups',
    network_name='my_network',
    method='property',
    property_name='pulse_group_id'  # Group by this node attribute
)
```

**Grid Assembly** (group by spatial location):
```python
import numpy as np

# Define a 2x2 grid
grid_id = np.array([[0, 1], [2, 3]])
grid_size = [[0.0, 500.0], [0.0, 500.0]]  # [[x_min, x_max], [y_min, y_max]]

sb.create_assemblies(
    name='spatial_groups',
    network_name='my_network',
    method='grid',
    grid_id=grid_id,
    grid_size=grid_size
)
```

### Generate Stimulus

Once you have assemblies, generate a stimulus file:

```python
sb.generate_stimulus(
    output_path='stimulus.h5',
    pattern_type='long',               # Firing pattern type
    assembly_name='random_groups',     # Use this assembly group
    population='stimulus',             # Population name in BMTK
    firing_rate=(0.0, 50.0, 0.0),    # (off_rate, burst_rate, silent_rate)
    t_start=1.0,                       # Start time (seconds)
    t_stop=15.0,                       # End time (seconds)
    on_time=1.0,                       # Duration of active period
    off_time=0.5                       # Duration of silent period
)
```

## Firing Patterns

Each pattern controls how firing rates vary over time across your assemblies. All patterns use a firing rate tuple: `(off_rate, burst_rate, silent_rate)`.

### `'long'` - Contiguous Bursts

One assembly burst during each cycle. Assemblies take turns with sequential bursts lasting the full on-time period.

**Use case**: Testing effects of sustained input to single populations or stimulus units.

```python
sb.generate_stimulus(
    output_path='long.h5',
    pattern_type='long',
    assembly_name='thalamus_groups',
    population='thalamus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=1.0,
    t_stop=15.0,
    on_time=1.0,
    off_time=0.5
)
```

### `'short'` - Sequential Bursts

Multiple brief bursts delivered sequentially to each assembly within a single cycle.

**Use case**: Testing rapid sequential stimulation or exploring temporal patterns.

```python
sb.generate_stimulus(
    output_path='short.h5',
    pattern_type='short',
    assembly_name='stimulus_groups',
    population='stimulus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=0.0,
    t_stop=10.0,
    on_time=1.0,
    off_time=0.5,
    n_rounds=2  # Each assembly gets 2 bursts per cycle
)
```

### `'ramp'` - Linear Rate Transitions

Firing rate increases or decreases linearly over a specified duration.

**Use case**: Testing response to gradually increasing/decreasing stimulus intensity.

```python
sb.generate_stimulus(
    output_path='ramp.h5',
    pattern_type='ramp',
    assembly_name='stim_groups',
    population='stimulus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=1.0,
    t_stop=15.0,
    ramp_up=2.0,      # Duration of ramp increase (seconds)
    ramp_down=2.0,    # Duration of ramp decrease (seconds)
    hold_time=5.0     # Duration at peak rate (seconds)
)
```

### `'join'` - Gradual Recruitment

Neurons gradually join or leave the active pool over multiple substeps.

**Use case**: Testing population coding and recruitment dynamics.

```python
sb.generate_stimulus(
    output_path='join.h5',
    pattern_type='join',
    assembly_name='stim_groups',
    population='stimulus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=1.0,
    t_stop=15.0,
    n_steps=5,        # Number of substeps for gradual recruitment
    on_time=3.0,
    off_time=0.5
)
```

### `'fade'` - Smooth Transitions

Smooth fade from one assembly firing pattern to another, allowing overlap.

**Use case**: Testing transitions between stimulus conditions or population switching.

```python
sb.generate_stimulus(
    output_path='fade.h5',
    pattern_type='fade',
    assembly_name='paired_groups',
    population='stimulus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=1.0,
    t_stop=15.0,
    fade_time=0.5,    # Duration of cross-fade between assemblies
    on_time=2.0,
    off_time=1.0
)
```

### `'loop'` - Cycling Patterns

Assemblies cycle through active periods with variable on-times and fixed off-times.

**Use case**: Testing repeating stimulus patterns with different cycle lengths.

```python
sb.generate_stimulus(
    output_path='loop.h5',
    pattern_type='loop',
    assembly_name='stim_groups',
    population='stimulus',
    firing_rate=(0.0, 50.0, 0.0),
    t_start=1.0,
    t_stop=15.0,
    on_times=[1.0, 2.0, 1.5],  # Variable on-time for each assembly
    off_time=0.5
)
```

## Assembly Methods

### Random Assembly

Randomly distribute nodes across `n_assemblies` groups with optional membership probability.

**Parameters**:
- `n_assemblies` (int): Number of groups to create
- `prob_in_assembly` (float, 0-1): Probability that each node joins the assembly (default: 1.0)

```python
sb.create_assemblies(
    name='random_assemblies',
    network_name='cortex',
    method='random',
    n_assemblies=10,
    prob_in_assembly=0.9  # 90% of nodes in each assembly
)
```

### Property-Based Assembly

Group nodes by the value of a specified attribute (e.g., node location, cell type, pulse_group_id).

**Parameters**:
- `property_name` (str): Column name in nodes dataframe to group by
- `probability` (float, 0-1): Optional; include each group with this probability

```python
sb.create_assemblies(
    name='by_type',
    network_name='cortex',
    method='property',
    property_name='pop_name'  # Group by cell type
)
```

### Grid-Based Assembly

Group nodes into a 2D spatial grid based on x, y coordinates.

**Parameters**:
- `grid_id` (ndarray): 2D array where each element is an assembly ID
- `grid_size` (list): [[x_min, x_max], [y_min, y_max]] spatial bounds

```python
grid_layout = np.array([[0, 1, 2],
                         [3, 4, 5]])
grid_bounds = [[0.0, 1000.0], [0.0, 500.0]]

sb.create_assemblies(
    name='spatial',
    network_name='cortex',
    method='grid',
    grid_id=grid_layout,
    grid_size=grid_bounds
)
```

## Advanced Features

### Generate Baseline Activity

Create background activity across all nodes in a network with a specified firing rate distribution.

```python
sb.generate_baseline(
    output_path='baseline.h5',
    network_name='cortex',
    distribution='lognormal',  # 'lognormal', 'normal', or 'constant'
    mean=20.0,                 # Mean firing rate (Hz)
    stdev=2.0,                 # Standard deviation
    t_start=0.0,
    t_stop=15.0
)
```

### Generate Population-Specific Shell Input

Create background activity with different firing rates for each population.

```python
shell_params = {
    'PN': (20.0, 2.0),   # (mean, stdev) Hz for PN cells
    'PV': (30.0, 3.0),   # (mean, stdev) Hz for PV cells
    'SST': (15.0, 1.5)   # (mean, stdev) Hz for SST cells
}

sb.generate_shell_input(
    output_path='shell.h5',
    network_name='input',
    shell_params=shell_params,
    distribution='lognormal',
    t_start=0.0,
    t_stop=15.0
)
```

## Best Practices

1. **Seed Management**: Set `net_seed` and `psg_seed` in the constructor for reproducible results:
   ```python
   sb = StimulusBuilder(config='config.json', net_seed=42, psg_seed=42)
   ```

2. **Sanity Checks**: Always verify stimulus generation by analyzing the output:
   ```python
   from bmtool.analysis.spikes import load_spikes_to_df
   df = load_spikes_to_df('stimulus.h5', network_name='stimulus', config='config.json')
   print(f"Total spikes: {len(df)}, Time range: {df['timestamps'].min()}-{df['timestamps'].max()}")
   ```

3. **Parameter Validation**: Ensure `on_time` + `off_time` aligns with your total simulation time and that `t_start < t_stop`.

4. **Population Naming**: Match the `population` parameter in `generate_stimulus()` with your BMTK configuration to ensure correct integration.

---

See the [Stimulus Tutorial](../examples/notebooks/stimulus/stimulus_tutorial.ipynb) for a complete working example demonstrating baseline generation, assembly creation, and multiple stimulus patterns.
