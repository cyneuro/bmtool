# bmtool
A collection of scripts to make developing networks in BMTK easier.
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/tjbanks/bmtool/blob/master/LICENSE) 

## Table of Contents
- [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [CLI](#cli)
  - [Single Cell](#single-cell)
  - [Connectors](#Connectors)
  - [bmplot](#bmplot)
  - [Examples]()




## Getting Started

**Installation**

```bash
pip install bmtool
```
For developers who will be pulling down additional updates to this repository regularly use the following instead.
```bash
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
python setup.py develop
```
Then download updates (from this directory) with
```
git pull
```

## CLI
```bash
> cd your_bmtk_model_directory
> bmtool
Usage: bmtool [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose  Verbose printing
  --help     Show this message and exit.

Commands:
  debug
  plot
  util

>  
> bmtool plot 
Usage: bmtool plot [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH  Configuration file to use, default: "simulation_config.json"
  --no-display   When set there will be no plot displayed, useful for saving
                 plots
  --help         Show this message and exit.

Commands:
  connection  Display information related to neuron connections
  positions   Plot cell positions for a given set of populations
  raster      Plot the spike raster for a given population
  report      Plot the specified report using BMTK's default report plotter
>
```
## Single Cell 
### The single cell module can use HOC template file or Allen database json SWC and json files.
### can do Passive properties, Current clamp, FI, ZAP
#### First step is it init the Profiler.
```python
from bmtool.singlecell import Profiler
profiler = Profiler(template_dir='./', mechanism_dir = './', dt=0.1) #init profiler
```
#### For Allen database cells an addtional step is needed. This function will create a neuron hoc object which the single cell module can then use.
# Currently this has Allen single cell way of using either need to make simplier way to load allen_cell or change how to use hoc tmeplate cell to check how HOC template is done check [here](https://github.com/GregGlickert/Amygdala-Cells). We need a common framework for both templates and allen types
```python
from bmtool.singlecell import load_allen_database_cells
cell = load_allen_database_cells(morph_path,dynam_path)
```
#### Passive properties
```python
sim = Passive(Cell, inj_amp=-100., inj_delay=1500., inj_dur=1000., tstop=2500., method='exp2')
title = 'Passive Cell Current Injection'
xlabel = 'Time (ms)'
ylabel = 'Membrane Potential (mV)'
X, Y = run_and_plot(sim, title, xlabel, ylabel, plot_injection_only=True)
plt.gca().plot(*sim.double_exponential_fit(), 'r:', label='double exponential fit')
plt.legend()
plt.show()
```
#### Current clamp
```python
from bmtool.singlecell import run_and_plot
sim = CurrentClamp(cell, inj_amp=350., inj_delay=1500., inj_dur=1000., tstop=3000., threshold=-15.)
X, Y = run_and_plot(sim, title='Current Injection', xlabel='Time (ms)',
                    ylabel='Membrane Potential (mV)', plot_injection_only=True)
plt.show()
```
#### FI curve
```python
sim = FI(Cell, i_start=0., i_stop=1000., i_increment=50., tstart=1500.,threshold=-15.)
X, Y = run_and_plot(sim, title='FI Curve', xlabel='Injection (nA)', ylabel='# Spikes')
plt.show()
```
#### ZAP
```python
sim = ZAP(Cell, chirp_type='linear')
X, Y = run_and_plot(sim)
plt.show()
```

# need part about command line Single cell tuner


## Connectors 
### connector functions support both distance depended and homogenous networks. Check out the examples folder for a more detailed notebook going over connectors
#### Unidirectional connector - unidirectional connections in bmtk network model with given probability within a single population (or between two populations)
```python
from bmtool.connectors  import UnidirectionConnector
connector = UnidirectionConnector(p=0.15, n_syn=1)
connector.setup_nodes(source=net.nodes(pop_name = 'PopA'), target=net.nodes(pop_name = 'PopB'))
net.add_edges(**connector.edge_params())
```
#### Recipical connector - buiilding connections in bmtk network model with reciprocal probability within a single population (or between two populations
```python
from bmtool.connectors  import ReciprocalConnector
connector_reci = ReciprocalConnector(p0=0.15, pr=0.06767705087, n_syn0=1, n_syn1=1,estimate_rho=False)
connector_reci.setup_nodes(source=net.nodes(pop_name = 'PopA'), target=net.nodes(pop_name = 'PopA'))
net.add_edges(**connector_reci.edge_params())
```
#### CorrelatedGapJunction
```python
putt code example here
```

#### OneToOneSequentialConnector
```python
putt code example here
```

## bmplot
## All bmplot functions require the same inputs with some requiring additional infomation
  * #### config: A BMTK simulation config 
  * #### sources: network name(s) to plot
  * #### targets: network name(s) to plot
  * #### sids: source node identifier 
  * #### tids: target node identifier
  * #### no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
  * #### save_file: If plot should be saved

# need some list to display all bmplot codes
# all options we want 
* ## gap junction connections (DONT THINK WE HAVE ANYTHING ABOUT THIS)
* ## connection total 
* ## connection percentages
* ## convergence
* ## divergence
* ## distance prob connections 
* ## uni vs bi directions
* ## connection table
* ## connection distrobution  

