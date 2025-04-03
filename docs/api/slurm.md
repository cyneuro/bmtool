# SLURM API Reference

This page provides API reference documentation for the SLURM module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.SLURM

## BlockRunner

::: bmtool.SLURM.BlockRunner

## ParameterSweep

::: bmtool.SLURM.ParameterSweep

## SLURMRunner

::: bmtool.SLURM.SLURMRunner

## JobArray

::: bmtool.SLURM.JobArray

## SweepAnalyzer

::: bmtool.SLURM.SweepAnalyzer
-->

The SLURM module provides utilities for running BMTK simulations on SLURM-based high-performance computing clusters.

## Key Components

### BlockRunner

The `BlockRunner` class manages simulation workflows:

- Break large simulations into smaller blocks
- Submit jobs to SLURM
- Monitor job progress
- Collect and combine results

### ParameterSweep

The `ParameterSweep` class enables parameter exploration:

- Define parameter ranges to explore
- Generate configuration files
- Submit sweep jobs to SLURM
- Organize results by parameter values

### SLURMRunner

The `SLURMRunner` class provides direct job submission capabilities:

- Configure SLURM job parameters
- Submit single jobs to SLURM
- Load environment modules
- Specify resource requirements

### JobArray

The `JobArray` class creates and manages SLURM job arrays:

- Submit multiple related jobs as an array
- Share resources efficiently
- Monitor all jobs in the array

### SweepAnalyzer

The `SweepAnalyzer` class processes parameter sweep results:

- Summarize results across parameter combinations
- Generate plots showing parameter effects
- Identify optimal parameter values 