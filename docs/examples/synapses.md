# Synapses Tutorials

The Synapses module provides tools for creating and tuning chemical and electrical synapses in NEURON and BMTK models.

## Features

- Interactive tuning of synapse parameters
- Support for both chemical and electrical (gap junction) synapses
- Visualization of synaptic responses
- Parameter fitting to match experimental data

The Synapses module provides two different tutorials for chemical synapse tuning:

The [BMTK Chemical Synapse Tuner](notebooks/synapses/synaptic_tuner/bmtk_chem_syn_tuner.ipynb) tutorial demonstrates how to use BMTool to interactively tune chemical synapses within BMTK networks. In this notebook, you'll learn:

- How to set up and configure chemical synapses in BMTK models
- How to switch between different network connections for tuning
- How to adjust synapse parameters and observe responses in a network context
- How to use the optimizer to automatically fit synaptic parameters

The [Neuron Chemical Synapse Tuner](notebooks/synapses/synaptic_tuner/neuron_chem_syn_tuner.ipynb) tutorial shows how to tune chemical synapses using pure NEURON models. This notebook covers:

- How to set up chemical synapses with detailed configuration
- How to manually tune synapse parameters outside of BMTK
- How to work with different synapse types (facilitating, depressing, etc.)
- How to implement custom synaptic mechanisms

The [Gap Junction Tuner](notebooks/synapses/gap_junction_tuner/gap_junction_tuner.ipynb) tutorial shows how to configure and optimize electrical synapses. This notebook covers:

- Setting up gap junctions in NEURON models
- Adjusting gap junction conductance
- Visualizing current flow through gap junctions
- Implementing gap junctions in network models

## Basic API Usage

If you prefer to use the Synapses module directly in your code, here are some basic examples:

### SynapseTuner with BMTK Networks

```python
from bmtool.synapses import SynapseTuner

# Create a tuner for BMTK networks
tuner = SynapseTuner(
    config='simulation_config.json',  # Path to BMTK config
    current_name='i',                 # Synaptic current to record
    slider_vars=['initW','Dep','Fac','Use','tau1','tau2']  # Parameters for sliders
)

# Display the interactive tuner
tuner.InteractiveTuner()

# Switch between different connections in your network
tuner._switch_connection('PV2Exc')
```

### SynapseTuner with Pure NEURON Models

```python
from bmtool.synapses import SynapseTuner

# Define general settings
general_settings = {
    'vclamp': True,
    'rise_interval': (0.1, 0.9),
    'tstart': 500.,
    'tdur': 100.,
    'threshold': -15.,
    'delay': 1.3,
    'weight': 1.,
    'dt': 0.025,
    'celsius': 20
}

# Define connection-specific settings
conn_settings = {
    'Exc2FSI': {
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp': -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "AMPA_NMDA_STP",
        },
        'spec_syn_param': {
            'initW': 0.76,
            'tau_r_AMPA': 0.45,
            'tau_d_AMPA': 7.5,
            'Use': 0.13,
            'Dep': 0.,
            'Fac': 200.
        },
    }
}

# Create tuner with custom settings
tuner = SynapseTuner(
    general_settings=general_settings,
    conn_type_settings=conn_settings
)

# Display the interactive tuner
tuner.InteractiveTuner()
```

### GapJunctionTuner

```python
from bmtool.synapses import GapJunctionTuner

# Create a tuner for gap junctions
tuner = GapJunctionTuner(
    cell1_template='Interneuron',
    cell2_template='Interneuron',
    template_dir='path/to/templates',
    mod_dir='path/to/mechanisms'
)

# Display the interactive tuner
tuner.show()

# Use the optimizer to find resistance for a target coupling coefficient
optimal_resistance = tuner.optimize(target_cc=0.05)
print(f"Optimal gap junction resistance: {optimal_resistance} MOhm")
```

For more advanced usage, please refer to the Jupyter notebook tutorials above.
