# Single Cell API Reference

This page provides API reference documentation for the Single Cell module, which contains functions and classes for working with individual neuron models.

## Utility Functions

::: bmtool.singlecell.load_biophys1
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.singlecell.load_allen_database_cells
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.singlecell.get_target_site
    options:
      show_root_heading: true
      heading_level: 3

## Current Clamp

::: bmtool.singlecell.CurrentClamp
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - setup
        - execute

## Passive Properties

::: bmtool.singlecell.Passive
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - tau_simple
        - tau_single_exponential
        - tau_double_exponential
        - double_exponential_fit
        - single_exponential_fit
        - execute

## Frequency-Current (F-I) Analysis

::: bmtool.singlecell.FI
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - setup
        - execute

## Impedance Analysis

::: bmtool.singlecell.ZAP
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - linear_chirp
        - exponential_chirp
        - zap_current
        - get_impedance
        - execute

## Cell Profiler

::: bmtool.singlecell.Profiler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - load_templates
        - passive_properties
        - current_injection
        - fi_curve
        - impedance_amplitude_profile

## Helper Functions

::: bmtool.singlecell.run_and_plot
    options:
      show_root_heading: true
      heading_level: 3
