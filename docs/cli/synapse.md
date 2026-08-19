# Synapse Utility Commands

The `util synapse` commands provide a desktop GUI workflow for tuning chemical synapses without using Jupyter.

```bash
bmtool util synapse --help
```

## Interactive Synapse Tuner (Desktop GUI)

Launch a Tk desktop interface backed by `bmtool.synapses.SynapseTuner`.

```bash
bmtool util synapse tune --config ./simulation_config.json
```

By default, BMTool will prompt you to choose which synapse variables should appear as sliders before opening the GUI.

### Common Usage

```bash
# Start with a specific network
bmtool util synapse tune --config ./simulation_config.json --network network_to_network

# Start with a specific connection
bmtool util synapse tune --config ./simulation_config.json --connection Exc2PV

# Record additional synaptic variables
bmtool util synapse tune --config ./simulation_config.json --other-vars Use,tau_f,tau_d

# Skip the interactive picker and provide slider vars directly
bmtool util synapse tune --config ./simulation_config.json --slider-vars tau1,tau2,Use --no-select-sliders
```

### Options

- `--config`: BMTK simulation config file. If omitted, BMTool uses the util-level config value.
- `--network`: Optional initial network from the config.
- `--connection`: Optional initial connection name from the config.
- `--current-name`: Synaptic current variable name to record (default: `i`).
- `--other-vars`: Comma-separated list of additional synapse variables to record and plot.
- `--slider-vars`: Comma-separated list of variables to use as sliders.
- `--select-sliders/--no-select-sliders`: Enable/disable the interactive slider selection prompt.

### Notes

- This command is **config-driven** in v1 and focuses on chemical synapse tuning.
- If your environment cannot open desktop windows (for example, headless servers), run this command from a machine with GUI support.
