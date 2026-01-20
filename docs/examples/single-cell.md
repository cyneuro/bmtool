# Single Cell Examples

This page provides examples of how to use the Single Cell module for various analyses.

## Python API Examples

The examples below demonstrate how to use the Single Cell module through its Python API:

### Basic Setup

All single cell examples require initializing the Profiler:

```python
from bmtool.singlecell import Profiler

# Initialize with paths to templates and mechanisms
profiler = Profiler(template_dir='templates', mechanism_dir='mechanisms', dt=0.1)
```

### Passive Properties Analysis

Calculating passive properties (V-rest, input resistance, and time constant):

```python
from bmtool.singlecell import Passive, run_and_plot
import matplotlib.pyplot as plt

# Create a Passive simulation object
sim = Passive('Cell_Cf', inj_amp=-100., inj_delay=1500., inj_dur=1000.,
              tstop=2500., method='exp2')

# Run the simulation and plot the results
title = 'Passive Cell Current Injection'
xlabel = 'Time (ms)'
ylabel = 'Membrane Potential (mV)'
X, Y = run_and_plot(sim, title, xlabel, ylabel, plot_injection_only=True)

# Plot the double exponential fit
plt.gca().plot(*sim.double_exponential_fit(), 'r:', label='double exponential fit')
plt.legend()
plt.show()
```

This will output the passive properties:

```
Injection location: Cell_Cf[0].soma[0](0.5)
Recording: Cell_Cf[0].soma[0](0.5)._ref_v
Running simulation for passive properties...

V Rest: -70.21 (mV)
Resistance: 128.67 (MOhms)
Membrane time constant: 55.29 (ms)
```

### Current Clamp

Running a current clamp to observe spiking behavior:

```python
from bmtool.singlecell import CurrentClamp, run_and_plot
import matplotlib.pyplot as plt

# Create a CurrentClamp simulation object
sim = CurrentClamp('Cell_Cf', inj_amp=350., inj_delay=1500., inj_dur=1000.,
                   tstop=3000., threshold=-15.)

# Run the simulation and plot the results
X, Y = run_and_plot(sim, title='Current Injection', xlabel='Time (ms)',
                    ylabel='Membrane Potential (mV)', plot_injection_only=True)
plt.show()
```

### FI Curve

Generating a frequency-current (FI) curve:

```python
from bmtool.singlecell import FI, run_and_plot
import matplotlib.pyplot as plt

# Create an FI simulation object
sim = FI('Cell_Cf', i_start=0., i_stop=1000., i_increment=50.,
          tstart=1500., threshold=-15.)

# Run the simulation and plot the results
X, Y = run_and_plot(sim, title='FI Curve', xlabel='Injection (nA)',
                    ylabel='# Spikes')
plt.show()
```

### ZAP Protocol

Analyzing frequency response using a chirp current (ZAP):

```python
from bmtool.singlecell import ZAP, run_and_plot
import matplotlib.pyplot as plt

# Create a ZAP simulation object
sim = ZAP('Cell_Cf')

# Run the simulation and plot the results
X, Y = run_and_plot(sim)
plt.show()
```

## Jupyter Notebook Tutorials

For more detailed examples with rich output and visualizations, check out our Jupyter notebook tutorials:

### Comprehensive Single Cell Analysis Tutorial

The [Comprehensive Single Cell Analysis Tutorial](notebooks/single_cell/single_cell_analysis.ipynb) provides a thorough guide to using BMTool's single cell analysis module. This tutorial covers:

- Multiple methods for loading neurons: Allen Database, NEURON HOC templates, and Python class-based models
- Various electrophysiological analysis techniques: Passive properties, Current clamp, Impedance (ZAP), and Frequency-Intensity (FI) curves
- Consistent analysis methods that work regardless of how the cell was loaded

### Other Examples

You can also access other examples from the command line:

```bash
# Cell Tuning via CLI
bmtool util cell tune --builder

# VHalf Segregation
bmtool util cell vhseg
```
