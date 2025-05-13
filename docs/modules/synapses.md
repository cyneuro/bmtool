# Synapses Module

The Synapses module provides tools for configuring and tuning synaptic connections in NEURON models, including both chemical synapses and electrical synapses (gap junctions).

## Features

- **Synaptic Tuner**: Interactive tuning of synaptic properties via Jupyter notebooks
- **Gap Junction Tuner**: Tools for adjusting gap junction properties with coupling coefficient optimization

## Synaptic Tuner

The SynapticTuner aids in the tuning of chemical synapses by providing an interactive interface with sliders in a Jupyter notebook to adjust synaptic parameters and view the effects in real-time.

### Key Features

- Interactive sliders for all synapse parameters
- Visualization of postsynaptic potentials
- Parameter export for use in network models
- Support for various synapse types (Exp2Syn, AMPA, NMDA, etc.)

### Example Usage

```python
from bmtool.synapses import SynapticTuner

# Create a tuner for an Exp2Syn synapse
tuner = SynapticTuner(
    synapse_type='Exp2Syn',
    pre_template='PreCell',
    post_template='PostCell',
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

## Gap Junction Tuner

The GapJunctionTuner provides tools for tuning electrical synapses (gap junctions) to achieve desired coupling coefficients.

### Key Features

- Interactive sliders for gap junction resistance
- Calculation of coupling coefficient
- Optimization algorithm to automatically find resistance values for desired coupling coefficients
- Visualization of voltage changes in coupled cells

### Example Usage

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

## Advanced Features

### Custom Synaptic Mechanisms

You can use custom synaptic mechanisms by specifying the synapse type and required parameters:

```python
tuner = SynapticTuner(
    synapse_type='MyCustomSynapse',
    parameters={
        'tau1': 0.5,
        'tau2': 5.0,
        'e': 0.0,
        'custom_param': 1.0
    },
    pre_template='PreCell',
    post_template='PostCell'
)
```

### Multiple Connection Testing

Test multiple synaptic connections with different parameters:

```python
from bmtool.synapses import SynapseTest

# Create a test for comparing different synapse configurations
test = SynapseTest(
    synapse_configs=[
        {'synapse_type': 'Exp2Syn', 'tau1': 0.5, 'tau2': 5.0, 'weight': 0.001},
        {'synapse_type': 'Exp2Syn', 'tau1': 1.0, 'tau2': 8.0, 'weight': 0.001},
        {'synapse_type': 'AMPA', 'weight': 0.001}
    ],
    pre_template='PreCell',
    post_template='PostCell'
)

# Run the test and plot results
test.run()
test.plot_comparison()
```
