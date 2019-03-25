# bmtools
A collection of scripts to make developing networks in BMTK easier.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/tjbanks/bmtool/blob/master/LICENSE) 

## Getting Started

**Installation**

```bash
git clone https://github.com/tjbanks/bmtools
cd bmtools
python setup.py install
```
For developers who will be pulling down additional updates to this repository regularly use the following instead.
```bash
python setup.py develop
```

**Example Use**

```bash
> cd your_bmtk_model_directory
> python -m bmtools
Build, plot or debug BMTK models easily.

python -m bmtools.build
python -m bmtools.plot
python -m bmtools.debug
python -m bmtools.util

optional arguments:
  -h, --help  show this help message and exit
>  
> python -m bmtools.plot
Plot BMTK models easily.

python -m bmtools.plot

positional arguments:
  {positions,positions-old,connection-total,connection-divergence,connection-convergence,network-graph,raster,raster-old}
    positions           Plot cell positions for a given set of populations
    positions-old       Plot cell positions for hipp model (to be removed/model specific for testing)
    connection-total    Plot the total connection matrix for a given set of populations
    connection-divergence
                        Plot the connection percentage matrix for a given set of populations
    connection-convergence
                        Plot the connection convergence matrix for a given set of populations
    network-graph       Plot the connection graph for supplied targets (default:all)
    raster              Plot the spike raster for a given population
    raster-old          Plot the spike raster for hipp model (to be removed/model specific for testing)

optional arguments:
  -h, --help            show this help message and exit
>
> python -m bmtools.plot positions
```
![bmtools](./figure.png "Positions Figure")

## Planned future features
```
python -m bmtools.build
    Create a starting point network
    Download sample networks

python -m bmtools.plot
    Plot variable traces
    Plot spike rasters
    Plot cell positions
    Plot connection matricies
    
python -m bmtools.debug 
    list cell types available for single debug
    Run a single cell in the network
    Isolate a single cell in the network
```
