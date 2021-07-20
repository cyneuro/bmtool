# bmtools
A collection of scripts to make developing networks in BMTK easier.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/tjbanks/bmtool/blob/master/LICENSE) 

## Getting Started

**Installation**

```bash
pip install bmtool
```
For developers who will be pulling down additional updates to this repository regularly use the following instead.
```bash
git clone https://github.com/tjbanks/bmtools
cd bmtools
python setup.py develop
```
Then download updates (from this directory) with
```
git pull
```

**Example Use**

```bash
> cd your_bmtk_model_directory
> bmtools
Usage: bmtools [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose  Verbose printing
  --help     Show this message and exit.

Commands:
  debug
  plot
  util

>  
> bmtools plot 
Usage: bmtools plot [OPTIONS] COMMAND [ARGS]...

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
> bmtools plot positions
```
![bmtools](./figure.png "Positions Figure")

## Plotting Configuration

BMTools utilizes the default `simulation-config.json` file to know which data files built by BMTK to read. to change this, specify the config after the `plot` command. Eg:

```
bmtool plot --config simulation-config-23.json [FUNCTION] 
```

## Ploting Connections

All connection tools can be customized by supplying additional arguments. 

```
Options:
  --title TEXT      change the plot's title
  --save-file TEXT  save plot to path supplied
  --sources TEXT    comma separated list of source node types [default:all]
  --targets TEXT    comma separated list of target node types [default:all]
  --sids TEXT       comma separated list of source node identifiers
                    [default:node_type_id]
  --tids TEXT       comma separated list of target node identifiers
                    [default:node_type_id]
  --no-prepend-pop  When set don't prepend the population name to the unique
                    ids [default:False]
```

#### `--sources`  and `--targets`
Are supplied as comma separated lists and corrospond with the population name specified in your model. Eg:
```
#initialize the networks in build_network.py
net = NetworkBuilder('hippocampus')
exp0net = NetworkBuilder('exp0input')
```
Default behavior is to plot connections between all populations but you can specify only a few to simplify your plots.

#### `--sids` and `--tids`
Comma separated lists of node identifiers replace the default `cell_id` automatically given to a cell population by BMTK. Any parameter passed to `NetworkBuilder.add_nodes` is stored in network `.h5` files and can be used to identify cells while connecting or producing plots. Eg:

```
# Adding nodes in build_network.py
net.add_nodes(N=inpTotal, pop_name='EC',
    positions=p_EC,
    model_type='biophysical',
    model_template='hoc:IzhiCell_EC2',
    morphology='blank.swc'
    )
```
We could then use the pop_name to alter the output of our connection plots.

```
bmtool plot connection --sids pop_name --tids pop_name [FUNCTION]
```
#### `--no-prepend-pop`

Default behavior of bmtools is to print the population name before the cell id (or sid/tid) followed by an underscore. Eg: `hippocampus_100`. By supplying `--no-prepend-pop` the cell name becomes `100` unless specified otherwise.

#### `All together`

Using these optional switches we can see the difference in our plot output below.

```
bmtool plot connection total
```
vs.
```
bmtool plot connection --sources hippocampus --targets hippocampus --sids pop_name --tids pop_name --no-prepend-pop --title 'Hippocampus Total Connections' total
```
![bmtools](./connection.png "Connection Figure")

### Plot Total Connections

To plot the total number of connections between two populations of cells run 
```
bmtool plot connection total
```
Remember to customize the output using the instructions above.

#### `--synfo`
This is an additional flag that can be used in the total connections plot. By default it is set to '0' which plots total connections. 
If it is specified as '1', it plots the mean and standard deviation number of connections. If it is '2', it plots the .mod files used for that connection type.
Finally if it is '3', it plots the parameter file (.json) used for the connection.

![bmtools](./connection_total.png "Connection Total Figure")

### Plot Average Convergence/Divergence

To plot the average convergence or divergence of a single cell excute one of the following commands:

```
bmtool plot connection convergence
bmtool plot connection divergence
```
![bmtools](./connection_con.png "Connection Convergence Figure")

### Plot Connection Diagram

To plot a rough sketch of cell type connectivity and the type of synapse used between cells run:

```
bmtool plot connection network-graph
```
![bmtools](./connection_graph.png "Connection Graph Figure")


`--edge-property` is an option available to change the synapse name if supplied to `NetworkBuilder.add_edges` when building the network. Default: `model_template`

### Edge Property Histograms

To view the distribution of an edge property between cell types run:

```
bmtool plot connection property-histogram-matrix
```

The following figure was generated using 
```
bmtool plot connection --sources hippocampus --targets hippocampus --sids pop_name --tids pop_name --no-prepend-pop --title 'Synaptic Weight Distribution between Cell Types' property-histogram-matrix
```

![bmtools](./connection_hist.png "Connection Histogram Figure")

By default the `property-histogram-matrix` looks at the `syn_weight` value specified in the `NetworkBuilder.add_edges` function when building your network. You can change this by specifying the `--edge-property`. Eg: 
```
bmtool plot connection property-histogram-matrix --edge-property [PROPERTY]
```

#### Plotting edge values during/after runtime

BMTools is capable of plotting connection properties obtained after runtime from reports. This is useful for synaptic weights that change over time. 

First, you must explicitly record the connection property in your `simulation_config.json`

```
  "reports": {
    "syn_report": {
      "cells": "hippocampus",
      "variable_name": "W_nmda",
      "module": "netcon_report",
      "sections": "soma",
      "syn_type": "pyr2pyr",
      "file_name": "syns.h5"
    }
  }
```
Where `pyr2pyr` is the `POINT_PROCESS` name for the synapse you're attempting to record, and the `variable_name` is a `RANGE` variable listed int the `NEURON` block of the synapse `.mod` file.

Once the simulation has been run un the following referencing the report specified above:

```
bmtool plot connection property-histogram-matrix --edge-property pyr2pyr_w --report output/syns.h5 --time 9999
```

The `--time-compare` option can be be used to show the weight distribution change between the specified times. Eg: ` --time 0 --time-compare 10000`

See the [BMTK Commit](https://github.com/AllenInstitute/bmtk/pull/67/files) for more details.

### Plotting Distance Probability Matrix between cell types

![bmtools](./connection_dist.png "Connection Histogram Figure")

To show the probability of a cell type being connected to another cell type based on distance run:

```
bmtool plot connection prob
```

Full summary of options:

```
> bmtool plot connection prob --help
Usage: bmtool plot connection prob [OPTIONS]

  Probabilities for a connection between given populations. Distance and
  type dependent

Options:
  --axis TEXT  comma separated list of axis to use for distance measure eg:
               x,y,z or x,y
  --bins TEXT  number of bins to separate distances into (resolution) -
               default: 8
  --line       Create a line plot instead of a binned bar plot
  --verbose    Print plot values for use in another script
  --help       Show this message and exit.
```

A more complete command (used for image above) may look similar to

```
bmtools plot connection --sources hippocampus --targets hippocampus --no-prepend-pop --sids pop_name --tids pop_name prob --bins 10 --line --verbose
```

This will plot cells in the `hippocampus` network, using the `pop_name` as the cell identifier. There will be `10` bins created to group the cell distances. A `line` plot will be generated instead of the default `bar` chart. All values for each plot will be printed to the console due to the `verbose` flag.

All  `point_process` cell types will be ignored since they do not have physical locations.

### Plotting Current Clamp and Spike Train Info
To plot all current clamp info involved in a simulation, use the following command (uses 'simulation_config.json' as default)
```
bmtools plot --config simulation_config_foo.json iclamp
```

To plot all spike trains and their target cells,
```
bmtools plot --config simulation_config_foo.json input
```

### Printing basic cell information involved in a simulation
```
bmtools plot --config simulation_config_foo.json cells
```

### Simulation Summary

Using previous functions, plots connection probability as a function of distance, total connections, cell information, current clamp information, input spike train information, and a 3D plot of the network if specified. 
```
bmtools plot --config simulation_config_foo.json summary
```

## Cell Tuning

### Single Cell Tuning

From a BMTK Model directory containing a `simulation_config.json` file:
```
bmtools util cell tune --builder
```

For non-BMTK cell tuning:
```
bmtools util cell --template TemplateFile.hoc --mod-folder ./ tune --builder
```
![bmtools](./figure2.png "Tuning Figure")

### FIR Curve plotting

```
> bmtools util cell fi --help
Usage: bmtools util cell fi [OPTIONS]

  Creates a NEURON GUI window with FI curve and passive properties

Options:
  --title TEXT
  --min-pa INTEGER   Min pA for injection
  --max-pa INTEGER   Max pA for injection
  --increment FLOAT  Increment the injection by [i] pA
  --tstart INTEGER   Injection start time
  --tdur INTEGER     Duration of injection default:1000ms
  --advanced         Interactive dialog to select injection and recording
                     points
  --help             Show this message and exit.

> bmtools util cell fi
? Select a cell:  (Use arrow keys)
 » CA3PyramidalCell
   DGCell
   IzhiCell
   IzhiCell_BC
   IzhiCell_EC
   IzhiCell_EC2
   IzhiCell_EC_BIO
   IzhiCell_EmoExcitatory
   IzhiCell_EmoInhibitory
   IzhiCell_OLM
   IzhiCell_int
```

![bmtools](./figure3.png "FIR Figure")

### VHalf Segregation Module

Based on the Alturki et al. (2016) paper.

Segregate your channel activation for an easier time tuning your cells.


```
> bmtools util cell vhseg --help

Usage: bmtools util cell vhseg [OPTIONS]

  Alturki et al. (2016) V1/2 Automated Segregation Interface, simplify
  tuning by separating channel activation

Options:
  --title TEXT
  --tstop INTEGER
  --outhoc TEXT         Specify the file you want the modified cell template
                        written to
  --outfolder TEXT      Specify the directory you want the modified cell
                        template and mod files written to (default: _seg)
  --outappend           Append out instead of overwriting (default: False)
  --debug               Print all debug statements
  --fminpa INTEGER      Starting FI Curve amps (default: 0)
  --fmaxpa INTEGER      Ending FI Curve amps (default: 1000)
  --fincrement INTEGER  Increment the FI Curve amps by supplied pA (default:
                        100)
  --infvars TEXT        Specify the inf variables to plot, skips the wizard.
                        (Comma separated, eg: inf_mech,minf_mech2,ninf_mech2)
  --segvars TEXT        Specify the segregation variables to globally set,
                        skips the wizard. (Comma separated, eg:
                        mseg_mech,nseg_mech2)
  --eleak TEXT          Specify the eleak var manually
  --gleak TEXT          Specify the gleak var manually
  --othersec TEXT       Specify other sections that a window should be
                        generated for (Comma separated, eg: dend[0],dend[1])
  --help                Show this message and exit.

```

#### Examples 

Wizard Mode (Interactive)

```
> bmtool util cell vhseg

? Select a cell:  CA3PyramidalCell
Using section dend[0]
? Show other sections? (default: No)  Yes
? Select other sections (space bar to select):  done (2 selections)
? Select inf variables to plot (space bar to select):   done (5 selections)
? Select segregation variables [OR VARIABLES YOU WANT TO CHANGE ON ALL SEGMENTS at the same time] (space bar to select):  done (2 selections)
```

Command Mode (Non-interactive)

```
bmtool util cell --template CA3PyramidalCell vhseg --othersec dend[0],dend[1] --infvars inf_im --segvars gbar_im --gleak gl_ichan2CA3 --eleak el_ichan2CA3
```

Example:

![bmtools](./figure4.png "Seg Figure")

Simple models can utilize 
``` 
bmtool util cell --hoc cell_template.hoc vhsegbuild --build
bmtool util cell --hoc segmented_template.hoc vhsegbuild
```
ex: [https://github.com/tjbanks/two-cell-hco](https://github.com/tjbanks/two-cell-hco)

## Planned future features
```
bmtools build
    Create a starting point network
    Download sample networks

bmtools plot
    Plot variable traces
    Plot spike rasters
    X Plot cell positions
    X Plot connection matricies
    
bmtools debug 
    X list cell types available for single debug
    X Run a single cell in the network
    Isolate a single cell in the network
```
