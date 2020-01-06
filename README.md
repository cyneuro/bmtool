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
