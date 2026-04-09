# Cell Utility Commands

The `util cell` commands provide tools for building, tuning, and identifying properties of single cell models.

Many of the following examples can be tested using the example data in [docs/examples/cli/single_cell](docs/examples/cli/single_cell). To test the commands first make sure you have these example files, then set your terminal to be in this single_cell directory. 

```bash
bmtool util cell --help
```

## Shared Options

Most cell commands allow overriding the cell's location and loading parameters:

- `--hoc-folder`: Override the HOC location (defaults to current directory or config).
- `--mod-folder`: Override the MOD file location (defaults to current directory or config).
- `--template`: Specify the NEURON template name.
- `--hoc`: Specify a single HOC file to load (when multiple files are present).

---

## Interactive Tuning (GUI)

Open interactive NEURON GUI windows for visual tuning of cell behavior and F-I curves.

### Cell Tuner
Open a general-purpose tuning interface for testing different stimulation protocols.

```bash
# Quick start using "easy" mode
bmtool util cell --template ET_Cell --hoc-folder . --mod-folder ./modfiles tune --easy
```

For advanced users, the `--builder` mode allows you to construct custom interfaces with specific widgets and plots:
```bash
bmtool util cell --template ET_Cell --hoc-folder . --mod-folder ./modfiles tune --builder
```

### FI Curve Interface
Open an interface dedicated to calculating FI curves (Frequency vs. Current) and passive property summaries.

```bash
bmtool util cell --template ET_Cell --hoc-folder . --mod-folder ./modfiles fi 
```

---

## Characterization (Passive & ZAP)

These commands allow for quick, non-interactive characterization of cell properties from the command line.

### Passive Properties
Calculate $V_{rest}$, $R_{in}$ (input resistance), and $\tau$ (membrane time constant).

```bash
# Calculate passive properties with default values
bmtool util cell --template ET_Cell --hoc-folder . --mod-folder ./modfiles passive --plot
```

**Options:**
- `--inj-amp`: Injection amplitude in pA (default: -100).
- `--inj-delay`: Start time of injection (default: 200ms).
- `--inj-dur`: Duration of injection (default: 1000ms).
- `--tstop`: Total simulation time (default: 1200ms).
- `--method`: `simple`, `exp`, or `exp2` for $\tau$ fitting.
- `--plot`: Display/Save the membrane potential trace.

### ZAP (Impedance & Resonance)
Calculate frequency-dependent impedance and resonant frequency using a chirp current injection.

```bash
# Run a ZAP simulation from 0 to 100 Hz
bmtool util cell --template ET_Cell --hoc-folder . --mod-folder ./modfiles zap --plot
```

**Options:**
- `--inj-amp`: Chirp amplitude in pA (default: 100.0).
- `--fstart`: Start frequency in Hz (default: 0.0).
- `--fend`: End frequency in Hz (default: 100.0).
- `--inj-delay`: Injection delay in ms (default: 200.0).
- `--inj-dur`: Chirp duration in ms (default: 1000.0).
- `--tstop`: Total simulation time in ms (default: 1200.0).
- `--plot`: Display/Save the ZAP results.

---

## VHalf Segregation (Alturki et al. 2016)

The VHalf Segregation tool provides an automated interface to simplify tuning by separating channel activation (V1/2). This implements the methodology described in Alturki et al. (2016).

```bash
# Launch the interactive wizard
bmtool util cell vhseg
```

**Command-line mode (skipping the wizard):**
```bash
bmtool util cell --template CA3PyramidalCell vhseg --othersec dend[0],dend[1] \
  --infvars inf_im --segvars gbar_im --gleak gl_ichan2CA3 --eleak el_ichan2CA3
```

