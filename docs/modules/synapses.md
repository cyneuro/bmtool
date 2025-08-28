# Synapses Module

The Synapses module provides tools for configuring and tuning synaptic connections in NEURON models, including both chemical synapses and electrical synapses (gap junctions).

## Features

- **Synaptic Tuner**: Interactive tuning of synaptic properties via Jupyter notebooks
- **Gap Junction Tuner**: Tools for adjusting gap junction properties with coupling coefficient optimization

## Synaptic Tuner

The SynapseTuner provides two main usage modes: one for BMTK networks and one for pure NEURON models. It offers an interactive interface with sliders in a Jupyter notebook to adjust synaptic parameters and view the effects in real-time.

### Key Features

- Interactive sliders for all synapse parameters
- Visualization of postsynaptic responses
- Support for BMTK network configurations
- Support for pure NEURON model tuning
- Parameter optimization algorithms
- Support for various synapse types (Exp2Syn, AMPA, NMDA, STP mechanisms, etc.)

### Example Usage with BMTK

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

### Example Usage with Pure NEURON

```python
from bmtool.synapses import SynapseTuner

# Define general and connection-specific settings
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

conn_type_settings = {
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
    conn_type_settings=conn_type_settings
)

# Display the interactive tuner
tuner.InteractiveTuner()
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

### Synapse Optimization

Use the SynapseOptimizer to automatically tune synapse parameters:

```python
from bmtool.synapses import SynapseOptimizer

# Create the optimizer
optimizer = SynapseOptimizer(tuner)

# Define parameter bounds
param_bounds = {
    'Dep': (0, 200.0),
    'Fac': (0, 400.0),
    'Use': (0.1, 1.0),
    'tau1': (1, 4),
    'tau2': (5, 20)
}

# Define target metrics
target_metrics = {
    'max_amp': 5.0,  # Target maximum amplitude (mV)
    'half_width': 10.0,  # Target half-width (ms)
    'rise_time': 2.0  # Target rise time (ms)
}

# Run optimization
result = optimizer.optimize_parameters(param_bounds, target_metrics)
print(result)
```

### Short-Term Plasticity Analysis

Analyze frequency response characteristics of synapses with short-term plasticity:

```python
# Analyze STP frequency response
frequencies, responses = tuner.stp_frequency_response(
    frequencies=[1, 5, 10, 20, 50, 100],  # Hz
    duration=1000  # ms
)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(frequencies, responses)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Steady-state Response')
plt.title('STP Frequency Response')
plt.show()
```
