# Stimulus Examples

## Overview

The stimulus module provides powerful tools for generating time-varying spiking inputs to BMTK networks. This page links to practical examples demonstrating baseline generation, assembly creation, and stimulus patterns.

## Stimulus Tutorial

The [Stimulus Tutorial Notebook](notebooks/stimulus/stimulus_tutorial.ipynb) provides a complete, runnable workflow for generating neural stimuli:

### What You'll Learn

- **Initialize StimulusBuilder** from your BMTK configuration
- **Generate Baseline Activity** with lognormal firing rate distributions across all input neurons
- **Create Population-Specific Background** (shell input) with different firing patterns for separate cell types
- **Partition Neurons into Assemblies** using property-based grouping (e.g., by pulse_group_id)
- **Generate Multiple Stimulus Patterns** ('long' and 'short') and compare their effects
- **Visualize Results** using firing rate histograms, raster plots, and time-series analysis

### Workflow Steps

1. Configure paths and random seeds
2. Create a StimulusBuilder instance
3. Generate baseline spiking activity
4. Analyze baseline firing rates with histograms
5. Generate population-specific shell input
6. Verify shell input statistics by population
7. Create node assemblies based on network properties
8. Generate long-duration stimulus pattern
9. Visualize long stimulus with firing rate and raster plots
10. Generate short-duration stimulus pattern
11. Compare short stimulus response across assemblies

### Dataset Requirements

To run the tutorial, you need:
- A BMTK configuration file pointing to your network definition
- Networks defined in the config (e.g., 'baseline', 'shell', 'thalamus')
- Nodes with population and property attributes (e.g., pop_name, pulse_group_id)

Adjust the `config_path` variable in the first code cell to point to your configuration file.

## Quick Start

For a minimal example, import and initialize:

```python
from bmtool.stimulus.core import StimulusBuilder

sb = StimulusBuilder(config='your_config.json', net_seed=123, psg_seed=1)

# Generate background with mixed distribution types
params = {
    'PN': {'mean_firing_rate': 20.0, 'stdev': 2.0},     # lognormal
    'PV': {'mean_firing_rate': 30.0},                    # constant
    'SST': {'mean_firing_rate': 15.0, 'stdev': 1.5}    # lognormal
}
sb.generate_background(output_path='background.h5', network_name='input',
                       population_params=params, t_start=0.0, t_stop=15.0)

# Generate stimulus: create assemblies first
sb.create_assemblies(name='stim_groups', network_name='thalamus', 
                    method='property', property_name='pulse_group_id')

# Then generate stimulus patterns
sb.generate_stimulus(output_path='stimulus.h5', pattern_type='long',
                    assembly_name='stim_groups', population='thalamus',
                    firing_rate=(0.0, 50.0, 0.0), t_start=1.0, t_stop=15.0,
                    on_time=1.0, off_time=0.5)
```

## See Also

- [Stimulus Module Overview](../modules/stimulus.md) - Comprehensive documentation of all features, assembly methods, and firing patterns
- [API Reference](../api/stimulus.md) - Detailed function signatures and docstrings
