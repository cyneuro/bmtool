# Synapses API Reference

This page provides API reference documentation for the Synapses module, which contains tools for creating, tuning and optimizing synaptic connections in NEURON models.

## Synapse Tuning

::: bmtool.synapses.SynapseTuner
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - _update_spec_syn_param
        - _set_up_cell
        - _set_up_synapse
        - _set_up_recorders
        - SingleEvent
        - _get_syn_prop
        - _set_syn_prop
        - _simulate_model
        - InteractiveTuner
        - stp_frequency_response

## Gap Junction Tuning

::: bmtool.synapses.GapJunctionTuner
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - model
        - plot_model
        - coupling_coefficient
        - InteractiveTuner

## Optimization Results

::: bmtool.synapses.SynapseOptimizationResult
    options:
      show_root_heading: true
      heading_level: 3

## Synapse Optimization

::: bmtool.synapses.SynapseOptimizer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - _normalize_params
        - _denormalize_params
        - _calculate_metrics
        - _default_cost_function
        - _objective_function
        - optimize_parameters
        - plot_optimization_results

## Gap Junction Optimization Results

::: bmtool.synapses.GapOptimizationResult
    options:
      show_root_heading: true
      heading_level: 3

## Gap Junction Optimization

::: bmtool.synapses.GapJunctionOptimizer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __init__
        - _objective_function
        - optimize_resistance
        - plot_optimization_results
        - parameter_sweep
