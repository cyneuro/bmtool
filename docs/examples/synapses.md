# Synapses Examples

This page provides examples of how to use the Synapses module.

## Jupyter Notebook Tutorials

For interactive examples with full context and explanations, check out our Jupyter notebook tutorials:

### Synaptic Tuner

The [Synaptic Tuner](synapses/synaptic_tuner.ipynb) tutorial demonstrates how to use BMTool to interactively tune chemical synapses. In this notebook, you'll learn:

- How to set up a synaptic connection between two cells
- How to use interactive sliders to adjust synapse parameters
- How to visualize the effects of parameter changes in real-time
- How to export optimized parameters for use in network models

### Gap Junction Tuner

The [Gap Junction Tuner](synapses/gap_junction_tuner.ipynb) tutorial shows how to configure and optimize electrical synapses. This notebook covers:

- How to create gap junction connections between cells
- How to measure coupling coefficients
- How to automatically find optimal resistance values
- How to visualize the behavior of coupled cells

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