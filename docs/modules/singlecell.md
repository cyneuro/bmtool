# Single Cell Module

The Single Cell module provides tools for analyzing and tuning biophysical cell models. It works with any neuron HOC object and can also turn Allen Institute database SWC and JSON files into HOC objects for analysis.

## Features

- **Passive Properties**: Calculate resting membrane potential, input resistance, and membrane time constant
- **Current Injection**: Run current clamp simulations to observe spiking behavior
- **FI Curves**: Generate frequency-current curves to characterize neuronal excitability
- **ZAP Protocol**: Analyze frequency response characteristics using chirp current injections
- **Cell Tuning**: Interactive interface for tuning cell parameters
- **VHalf Segregation**: Simplify channel tuning by separating channel activation based on Alturki et al. (2016)

## Getting Started

First, initialize the Profiler with paths to your templates and mechanisms:

```python
from bmtool.singlecell import Profiler
profiler = Profiler(template_dir='templates', mechanism_dir='mechanisms', dt=0.1)
```

For Allen Institute cell models, load them using:

```python
from bmtool.singlecell import load_allen_database_cells
cell = load_allen_database_cells(path_to_SWC_file, path_to_json_file)
```

## Passive Properties

Calculate passive membrane properties (V-rest, input resistance, and time constant):

```python
from bmtool.singlecell import Passive, run_and_plot
import matplotlib.pyplot as plt

sim = Passive('Cell_Cf', inj_amp=-100., inj_delay=1500., inj_dur=1000.,
              tstop=2500., method='exp2')
title = 'Passive Cell Current Injection'
xlabel = 'Time (ms)'
ylabel = 'Membrane Potential (mV)'
X, Y = run_and_plot(sim, title, xlabel, ylabel, plot_injection_only=True)
plt.gca().plot(*sim.double_exponential_fit(), 'r:', label='double exponential fit')
plt.legend()
plt.show()
```

## Current Clamp

Run a current clamp simulation:

```python
from bmtool.singlecell import CurrentClamp
sim = CurrentClamp('Cell_Cf', inj_amp=350., inj_delay=1500., inj_dur=1000.,
                   tstop=3000., threshold=-15.)
X, Y = run_and_plot(sim, title='Current Injection', xlabel='Time (ms)',
                    ylabel='Membrane Potential (mV)', plot_injection_only=True)
plt.show()
```

## FI Curve

Generate a frequency-current (FI) curve:

```python
from bmtool.singlecell import FI
sim = FI('Cell_Cf', i_start=0., i_stop=1000., i_increment=50.,
          tstart=1500., threshold=-15.)
X, Y = run_and_plot(sim, title='FI Curve', xlabel='Injection (nA)',
                    ylabel='# Spikes')
plt.show()
```

## ZAP Protocol

Analyze frequency response using a chirp current (ZAP):

```python
from bmtool.singlecell import ZAP
sim = ZAP('Cell_Cf')
X, Y = run_and_plot(sim)
plt.show()
```

## Cell Tuning

The cell tuning interface can be accessed via the command line:

```bash
# For BMTK models with a simulation_config.json file
bmtool util cell tune --builder

# For non-BMTK cell tuning
bmtool util cell --template TemplateFile.hoc --mod-folder ./ tune --builder
```

## VHalf Segregation

The VHalf Segregation module helps simplify channel tuning:

```bash
# Interactive wizard mode
bmtool util cell vhseg

# Command mode
bmtool util cell --template CA3PyramidalCell vhseg --othersec dend[0],dend[1] \
  --infvars inf_im --segvars gbar_im --gleak gl_ichan2CA3 --eleak el_ichan2CA3
```

For building simple models:

```bash
bmtool util cell --hoc cell_template.hoc vhsegbuild --build
bmtool util cell --hoc segmented_template.hoc vhsegbuild
```
