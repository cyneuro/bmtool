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
  {positions,connection_total,connection_percent,connection_divergence,connection_convergence,raster}
    positions           Plot cell positions for a given set of populations
    connection_total    Plot the total connection matrix for a given set of populations
    connection_percent  Plot the connection percentage matrix for a given set of populations
    connection_divergence
                        Plot the connection percentage matrix for a given set of populations
    connection_convergence
                        Plot the connection convergence matrix for a given set of populations
    raster              Plot the spike raster for a given set of populations

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       simulation config file (default: simulation_config.json) [MUST be first argument]
  --no-display          When set there will be no plot displayed, useful for saving plots
>
> python -m bmtools.plot positions
```
![bmtools](./figure.png "Positions Figure")
