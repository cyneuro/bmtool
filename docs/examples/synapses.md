# Synapses Tutorials

The Synapses module provides tools for creating and tuning chemical and electrical synapses in NEURON and BMTK models.

## Features

- Interactive tuning of synapse parameters
- Support for both chemical and electrical (gap junction) synapses
- Visualization of synaptic responses
- Parameter fitting to match experimental data

The [Synaptic Tuner](notebooks/synapses/synaptic_tuner/synaptic_tuner.ipynb) tutorial demonstrates how to use BMTool to interactively tune chemical synapses. In this notebook, you'll learn:

- How to set up and configure chemical synapses
- How to adjust synapse parameters and observe responses
- How to fit synaptic parameters to target response profiles
- How to implement the tuned synapses in your models

The [Gap Junction Tuner](notebooks/synapses/gap_junction_tuner/gap_junction_tuner.ipynb) tutorial shows how to configure and optimize electrical synapses. This notebook covers:

- Setting up gap junctions in NEURON models
- Adjusting gap junction conductance
- Visualizing current flow through gap junctions
- Implementing gap junctions in network models

## Basic API Usage

If you prefer to use the Synapses module directly in your code, here are some basic examples:

### SynapticTuner

```python
from bmtool.synapses import SynapticTuner

# Create a tuner for an Exp2Syn synapse
tuner = SynapticTuner(
    synapse_type='Exp2Syn',
    pre_template='PyramidalCell',
    post_template='InterneuronCell',
    pre_section='soma',
    post_section='dend[0]',
    template_dir='path/to/templates',
    mod_dir='path/to/mechanisms'
)

# Display the interactive tuner
tuner.show()

# After tuning, export parameters
params = tuner.get_parameters()
print(params)
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
