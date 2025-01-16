import os
import json
import neuron
import numpy as np
from neuron import h
from typing import List, Dict, Callable, Optional,Tuple
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from neuron.units import ms, mV
from dataclasses import dataclass
# scipy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit,minimize_scalar,minimize
# widgets
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import HBox, VBox

class SynapseTuner:
    def __init__(self, mechanisms_dir: str, templates_dir: str, conn_type_settings: dict, connection: str,
                 general_settings: dict, json_folder_path: str = None, current_name: str = 'i',
                 other_vars_to_record: list = None, slider_vars: list = None) -> None:
        """
        Initialize the SynapseModule class with connection type settings, mechanisms, and template directories.
        
        Parameters:
        -----------
        mechanisms_dir : str
            Directory path containing the compiled mod files needed for NEURON mechanisms.
        templates_dir : str
            Directory path containing cell template files (.hoc or .py) loaded into NEURON.
        conn_type_settings : dict
            A dictionary containing connection-specific settings, such as synaptic properties and details.
        connection : str
            Name of the connection type to be used from the conn_type_settings dictionary.
        general_settings : dict
            General settings dictionary including parameters like simulation time step, duration, and temperature.
        json_folder_path : str, optional
            Path to folder containing JSON files with additional synaptic properties to update settings.
        current_name : str, optional
            Name of the synaptic current variable to be recorded (default is 'i').
        other_vars_to_record : list, optional
            List of additional synaptic variables to record during the simulation (e.g., 'Pr', 'Use').
        slider_vars : list, optional
            List of synaptic variables you would like sliders set up for the STP sliders method by default will use all parameters in spec_syn_param.

        """
        neuron.load_mechanisms(mechanisms_dir)
        h.load_file(templates_dir)
        self.conn_type_settings = conn_type_settings
        if json_folder_path:
            print(f"updating settings from json path {json_folder_path}")
            self._update_spec_syn_param(json_folder_path)
        self.general_settings = general_settings
        self.conn = self.conn_type_settings[connection]
        self.synaptic_props = self.conn['spec_syn_param']
        self.vclamp = general_settings['vclamp']
        self.current_name = current_name
        self.other_vars_to_record = other_vars_to_record
        self.ispk = None

        if slider_vars:
            # Start by filtering based on keys in slider_vars
            self.slider_vars = {key: value for key, value in self.synaptic_props.items() if key in slider_vars}
            # Iterate over slider_vars and check for missing keys in self.synaptic_props
            for key in slider_vars:
                # If the key is missing from synaptic_props, get the value using getattr
                if key not in self.synaptic_props:
                    try:
                        # Get the alternative value from getattr dynamically
                        self._set_up_cell()
                        self._set_up_synapse()
                        value = getattr(self.syn,key)
                        #print(value)
                        self.slider_vars[key] = value
                    except AttributeError as e:
                        print(f"Error accessing '{key}' in syn {self.syn}: {e}")

        else:
            self.slider_vars = self.synaptic_props


        h.tstop = general_settings['tstart'] + general_settings['tdur']
        h.dt = general_settings['dt']  # Time step (resolution) of the simulation in ms
        h.steps_per_ms = 1 / h.dt
        h.celsius = general_settings['celsius']

    def _update_spec_syn_param(self, json_folder_path):
        """
        Update specific synaptic parameters using JSON files located in the specified folder.
        
        Parameters:
        -----------
        json_folder_path : str
            Path to folder containing JSON files, where each JSON file corresponds to a connection type.
        """
        for conn_type, settings in self.conn_type_settings.items():
            json_file_path = os.path.join(json_folder_path, f"{conn_type}.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    settings['spec_syn_param'].update(json_data)
            else:
                print(f"JSON file for {conn_type} not found.")


    def _set_up_cell(self):
        """
        Set up the neuron cell based on the specified connection settings.
        """
        self.cell = getattr(h, self.conn['spec_settings']['post_cell'])()


    def _set_up_synapse(self):
        """
        Set up the synapse on the target cell according to the synaptic parameters in `conn_type_settings`.
        
        Notes:
        ------
        - `_set_up_cell()` should be called before setting up the synapse.
        - Synapse location, type, and properties are specified within `spec_syn_param` and `spec_settings`.
        """
        self.syn = getattr(h, self.conn['spec_settings']['level_of_detail'])(list(self.cell.all)[self.conn['spec_settings']['sec_id']](self.conn['spec_settings']['sec_x']))
        for key, value in self.conn['spec_syn_param'].items():
            if isinstance(value, (int, float)):  # Only create sliders for numeric values
                if hasattr(self.syn, key):
                    setattr(self.syn, key, value)
                else:
                    print(f"Warning: {key} cannot be assigned as it does not exist in the synapse. Check your mod file or spec_syn_param.")


    def _set_up_recorders(self):
        """
        Set up recording vectors to capture simulation data.
        
        The method sets up recorders for:
        - Synaptic current specified by `current_name`
        - Other specified synaptic variables (`other_vars_to_record`)
        - Time, soma voltage, and voltage clamp current for all simulations.
        """
        self.rec_vectors = {}
        for var in self.other_vars_to_record:
            self.rec_vectors[var] = h.Vector()
            ref_attr = f'_ref_{var}'
            if hasattr(self.syn, ref_attr):
                self.rec_vectors[var].record(getattr(self.syn, ref_attr))
            else:
                print(f"Warning: {ref_attr} not found in the syn object. Use vars() to inspect available attributes.")

        # Record synaptic current
        self.rec_vectors[self.current_name] = h.Vector()
        ref_attr = f'_ref_{self.current_name}'
        if hasattr(self.syn, ref_attr):
            self.rec_vectors[self.current_name].record(getattr(self.syn, ref_attr))
        else:
            print("Warning: Synaptic current recorder not set up correctly.")

        # Record time, synaptic events, soma voltage, and voltage clamp current
        self.t = h.Vector()
        self.tspk = h.Vector()
        self.soma_v = h.Vector()
        self.ivcl = h.Vector()

        self.t.record(h._ref_t)
        self.nc.record(self.tspk)
        self.nc2.record(self.tspk)
        self.soma_v.record(self.cell.soma[0](0.5)._ref_v)
        self.ivcl.record(self.vcl._ref_i)


    def SingleEvent(self):
        """
        Simulate a single synaptic event by delivering an input stimulus to the synapse.
        
        The method sets up the neuron cell, synapse, stimulus, and voltage clamp, 
        and then runs the NEURON simulation for a single event. The single synaptic event will occur at general_settings['tstart']
        Will display graphs and synaptic properies works best with a jupyter notebook
        """
        self._set_up_cell()
        self._set_up_synapse()

        # Set up the stimulus
        self.nstim = h.NetStim()
        self.nstim.start = self.general_settings['tstart']
        self.nstim.noise = 0
        self.nstim2 = h.NetStim()
        self.nstim2.start = h.tstop
        self.nstim2.noise = 0
        self.nc = h.NetCon(self.nstim, self.syn, self.general_settings['threshold'], self.general_settings['delay'], self.general_settings['weight'])
        self.nc2 = h.NetCon(self.nstim2, self.syn, self.general_settings['threshold'], self.general_settings['delay'], self.general_settings['weight'])
        
        # Set up voltage clamp
        self.vcl = h.VClamp(self.cell.soma[0](0.5))
        vcldur = [[0, 0, 0], [self.general_settings['tstart'], h.tstop, 1e9]]
        for i in range(3):
            self.vcl.amp[i] = self.conn['spec_settings']['vclamp_amp']
            self.vcl.dur[i] = vcldur[1][i]

        self._set_up_recorders()

        # Run simulation
        h.tstop = self.general_settings['tstart'] + self.general_settings['tdur']
        self.nstim.interval = self.general_settings['tdur']
        self.nstim.number = 1
        self.nstim2.start = h.tstop
        h.run()
        self._plot_model([self.general_settings['tstart'] - 5, self.general_settings['tstart'] + self.general_settings['tdur']])
        syn_props = self._get_syn_prop(rise_interval=self.general_settings['rise_interval'])     
        for prop in syn_props.items():
            print(prop)


    def _find_first(self, x):
        """
        Find the index of the first non-zero element in a given array.
        
        Parameters:
        -----------
        x : np.array
            The input array to search.
        
        Returns:
        --------
        int
            Index of the first non-zero element, or None if none exist.
        """
        x = np.asarray(x)
        idx = np.nonzero(x)[0]
        return idx[0] if idx.size else None


    def _get_syn_prop(self, rise_interval=(0.2, 0.8), dt=h.dt, short=False):
        """
        Calculate synaptic properties such as peak amplitude, latency, rise time, decay time, and half-width.
        
        Parameters:
        -----------
        rise_interval : tuple of floats, optional
            Fractional rise time interval to calculate (default is (0.2, 0.8)).
        dt : float, optional
            Time step of the simulation (default is NEURON's `h.dt`).
        short : bool, optional
            If True, only return baseline and sign without calculating full properties.
        
        Returns:
        --------
        dict
            A dictionary containing the synaptic properties: baseline, sign, peak amplitude, latency, rise time, 
            decay time, and half-width.
        """
        if self.vclamp:
            isyn = self.ivcl
        else:
            isyn = self.rec_vectors['i']
        isyn = np.asarray(isyn)
        tspk = np.asarray(self.tspk)
        if tspk.size:
            tspk = tspk[0]
            
        ispk = int(np.floor(tspk / dt))
        baseline = isyn[ispk]
        isyn = isyn[ispk:] - baseline
        # print(np.abs(isyn))
        # print(np.argmax(np.abs(isyn)))
        # print(isyn[np.argmax(np.abs(isyn))])
        # print(np.sign(isyn[np.argmax(np.abs(isyn))]))  
        sign = np.sign(isyn[np.argmax(np.abs(isyn))])  
        if short:
            return {'baseline': baseline, 'sign': sign}
        isyn *= sign
        # print(isyn)
        # peak amplitude
        ipk, _ = find_peaks(isyn)
        ipk = ipk[0]
        peak = isyn[ipk]
        # latency
        istart = self._find_first(np.diff(isyn[:ipk + 1]) > 0)
        latency = dt * (istart + 1)
        # rise time
        rt1 = self._find_first(isyn[istart:ipk + 1] > rise_interval[0] * peak)
        rt2 = self._find_first(isyn[istart:ipk + 1] > rise_interval[1] * peak)
        rise_time = (rt2 - rt1) * dt
        # decay time
        iend = self._find_first(np.diff(isyn[ipk:]) > 0)
        iend = isyn.size - 1 if iend is None else iend + ipk
        decay_len = iend - ipk + 1
        popt, _ = curve_fit(lambda t, a, tau: a * np.exp(-t / tau), dt * np.arange(decay_len),
                            isyn[ipk:iend + 1], p0=(peak, dt * decay_len / 2))
        decay_time = popt[1]
        # half-width
        hw1 = self._find_first(isyn[istart:ipk + 1] > 0.5 * peak)
        hw2 = self._find_first(isyn[ipk:] < 0.5 * peak)
        hw2 = isyn.size if hw2 is None else hw2 + ipk
        half_width = dt * (hw2 - hw1)
        output = {'baseline': baseline, 'sign': sign, 'latency': latency,
            'amp': peak, 'rise_time': rise_time, 'decay_time': decay_time, 'half_width': half_width}
        return output


    def _plot_model(self, xlim):
        """
        Plots the results of the simulation, including synaptic current, soma voltage,
        and any additional recorded variables.

        Parameters:
        -----------
        xlim : tuple
            A tuple specifying the limits of the x-axis for the plot (start_time, end_time).
        
        Notes:
        ------
        - The function determines how many plots to generate based on the number of variables recorded.
        - Synaptic current and either voltage clamp or soma voltage will always be plotted.
        - If other variables are provided in `other_vars_to_record`, they are also plotted.
        - The function adjusts the plot layout and removes any extra subplots that are not needed.
        """
        # Determine the number of plots to generate (at least 2: current and voltage)
        num_vars_to_plot = 2 + (len(self.other_vars_to_record) if self.other_vars_to_record else 0)
        
        # Set up figure based on number of plots (2x2 grid max)
        num_rows = (num_vars_to_plot + 1) // 2  # This ensures we have enough rows
        fig, axs = plt.subplots(num_rows, 2, figsize=(12, 7))
        axs = axs.ravel()
        
        # Plot synaptic current (always included)
        current = self.rec_vectors[self.current_name]
        syn_prop = self._get_syn_prop(short=True)
        current = (current - syn_prop['baseline']) 
        current = current * 1000
        
        axs[0].plot(self.t, current)
        if self.ispk !=None:
            for num in range(len(self.ispk)):
                axs[0].text(self.t[self.ispk[num]],current[self.ispk[num]],f"{str(num+1)}")

        axs[0].set_ylabel('Synaptic Current (pA)')
        
        # Plot voltage clamp or soma voltage (always included)
        ispk = int(np.round(self.tspk[0] / h.dt))
        if self.vclamp:
            baseline = self.ivcl[ispk]
            ivcl_plt = np.array(self.ivcl) - baseline
            ivcl_plt[:ispk] = 0
            axs[1].plot(self.t, 1000 * ivcl_plt)
            axs[1].set_ylabel('VClamp Current (pA)')
        else:
            soma_v_plt = np.array(self.soma_v)
            soma_v_plt[:ispk] = soma_v_plt[ispk]

            axs[1].plot(self.t, soma_v_plt)
            axs[1].set_ylabel('Soma Voltage (mV)')
        
        # Plot any other variables from other_vars_to_record, if provided
        if self.other_vars_to_record:
            for i, var in enumerate(self.other_vars_to_record, start=2):
                if var in self.rec_vectors:
                    axs[i].plot(self.t, self.rec_vectors[var])
                    axs[i].set_ylabel(f'{var.capitalize()}')

        # Adjust the layout
        for i, ax in enumerate(axs[:num_vars_to_plot]):
            ax.set_xlim(*xlim)
            if i >= num_vars_to_plot - 2:  # Add x-label to the last row
                ax.set_xlabel('Time (ms)')
        
        # Remove extra subplots if less than 4 plots are present
        if num_vars_to_plot < len(axs):
            for j in range(num_vars_to_plot, len(axs)):
                fig.delaxes(axs[j])

        #plt.tight_layout()
        plt.show()


    def _set_drive_train(self,freq=50., delay=250.):
        """
        Configures trains of 12 action potentials at a specified frequency and delay period
        between pulses 8 and 9.

        Parameters:
        -----------
        freq : float, optional
            Frequency of the pulse train in Hz. Default is 50 Hz.
        delay : float, optional
            Delay period in milliseconds between the 8th and 9th pulses. Default is 250 ms.

        Returns:
        --------
        tstop : float
            The time at which the last pulse stops.
        
        Notes:
        ------
        - This function is based on experiments from the Allen Database.
        """
        n_init_pulse = 8
        n_ending_pulse = 4
        self.nstim.start = self.general_settings['tstart']
        self.nstim.interval = 1000 / freq
        self.nstim2.interval = 1000 / freq
        self.nstim.number = n_init_pulse
        self.nstim2.number = n_ending_pulse
        self.nstim2.start = self.nstim.start + (n_init_pulse - 1) * self.nstim.interval + delay
        tstop = self.nstim2.start + n_ending_pulse * self.nstim2.interval
        return tstop
 

    def _response_amplitude(self):
        """
        Calculates the amplitude of the synaptic response by analyzing the recorded synaptic current.

        Returns:
        --------
        amp : list
            A list containing the peak amplitudes for each segment of the recorded synaptic current.
        
        """
        isyn = np.array(self.rec_vectors[self.current_name].to_python())
        tspk = np.append(np.asarray(self.tspk), h.tstop)
        syn_prop = self._get_syn_prop(short=True)
        # print("syn_prp[sign] = " + str(syn_prop['sign']))
        isyn = (isyn - syn_prop['baseline']) 
        isyn *= syn_prop['sign']
        ispk = np.floor((tspk + self.general_settings['delay']) / h.dt).astype(int)

        
        try:        
            amp = [isyn[ispk[i]:ispk[i + 1]].max() for i in range(ispk.size - 1)]
            # indexs of where the max of the synaptic current is at. This is then plotted     
            self.ispk = [np.argmax(isyn[ispk[i]:ispk[i + 1]]) + ispk[i] for i in range(ispk.size - 1)]
        # Sometimes the sim can cutoff at the peak of synaptic activity. So we just reduce the range by 1 and ingore that point
        except:
            amp = [isyn[ispk[i]:ispk[i + 1]].max() for i in range(ispk.size - 2)]  
            self.ispk = [np.argmax(isyn[ispk[i]:ispk[i + 1]]) + ispk[i] for i in range(ispk.size - 2)]
        
        return amp


    def _find_max_amp(self, amp, normalize_by_trial=True):
        """
        Determines the maximum amplitude from the response data.

        Parameters:
        -----------
        amp : array-like
            Array containing the amplitudes of synaptic responses.
        normalize_by_trial : bool, optional
            If True, normalize the maximum amplitude within each trial. Default is True.
        
        Returns:
        --------
        max_amp : float
            The maximum or minimum amplitude based on the sign of the response.
        """
        max_amp = amp.max(axis=1 if normalize_by_trial else None)
        min_amp = amp.min(axis=1 if normalize_by_trial else None)
        if(abs(min_amp) > max_amp):
            return min_amp
        return max_amp


    def _calc_ppr_induction_recovery(self,amp, normalize_by_trial=True,print_math=True):
        """
        Calculates induction and recovery metrics from the synaptic response amplitudes.

        Parameters:
        -----------
        amp : array-like
            Array containing the amplitudes of synaptic responses.
        normalize_by_trial : bool, optional
            If True, normalize the amplitudes within each trial. Default is True.

        Returns:
        --------
        induction : float
            The calculated induction value (difference between pulses 6-8 and 1st pulse).
        recovery : float
            The calculated recovery value (difference between pulses 9-12 and pulses 1-4).
        maxamp : float
            The maximum amplitude in the response.
        """
        amp = np.array(amp)
        amp = (amp * 1000) # scale up
        amp = amp.reshape(-1, amp.shape[-1])
        maxamp = amp.max(axis=1 if normalize_by_trial else None)

        def format_array(arr):
            """Format an array to 2 significant figures for cleaner output."""
            return np.array2string(arr, precision=2, separator=', ', suppress_small=True)
        
        if print_math:
            print("\n" + "="*40)
            print("Short Term Plasticity Results")
            print("="*40)
            print("PPR: Above 1 is facilitating, below 1 is depressing.")
            print("Induction: Above 0 is facilitating, below 0 is depressing.")
            print("Recovery: A measure of how fast STP decays.\n")

            # PPR Calculation
            ppr = amp[:, 1:2] / amp[:, 0:1]
            print("Paired Pulse Response (PPR)")
            print("Calculation: 2nd pulse / 1st pulse")
            print(f"Values: ({format_array(amp[:, 1:2])}) / ({format_array(amp[:, 0:1])}) = {format_array(ppr)}\n")

            # Induction Calculation
            induction = np.mean((amp[:, 5:8].mean(axis=1) - amp[:, :1].mean(axis=1)) / maxamp)
            print("Induction")
            print("Calculation: (avg(6th, 7th, 8th pulses) - 1st pulse) / max amps")
            print(f"Values: avg({format_array(amp[:, 5:8])}) - {format_array(amp[:, :1])} / {format_array(maxamp)}")
            print(f"({format_array(amp[:, 5:8].mean(axis=1))}) - ({format_array(amp[:, :1].mean(axis=1))}) / {format_array(maxamp)} = {induction:.3f}\n")

            # Recovery Calculation
            recovery = np.mean((amp[:, 8:12].mean(axis=1) - amp[:, :4].mean(axis=1)) / maxamp)
            print("Recovery")
            print("Calculation: (avg(9th, 10th, 11th, 12th pulses) - avg(1st to 4th pulses)) / max amps")
            print(f"Values: avg({format_array(amp[:, 8:12])}) - avg({format_array(amp[:, :4])}) / {format_array(maxamp)}")
            print(f"({format_array(amp[:, 8:12].mean(axis=1))}) - ({format_array(amp[:, :4].mean(axis=1))}) / {format_array(maxamp)} = {recovery:.3f}\n")

            print("="*40 + "\n")

        recovery = np.mean((amp[:, 8:12].mean(axis=1) - amp[:, :4].mean(axis=1)) / maxamp)
        induction = np.mean((amp[:, 5:8].mean(axis=1) - amp[:, :1].mean(axis=1)) / maxamp)
        ppr = amp[:, 1:2] / amp[:, 0:1]
        # maxamp = max(amp, key=lambda x: abs(x[0]))
        maxamp = maxamp.max()
        
        return ppr, induction, recovery


    def _set_syn_prop(self, **kwargs):
        """
        Sets the synaptic parameters based on user inputs from sliders.
        
        Parameters:
        -----------
        **kwargs : dict
            Synaptic properties (such as weight, Use, tau_f, tau_d) as keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self.syn, key, value)


    def _simulate_model(self,input_frequency, delay, vclamp=None):
        """
        Runs the simulation with the specified input frequency, delay, and voltage clamp settings.

        Parameters:
        -----------
        input_frequency : float
            Frequency of the input drive train in Hz.
        delay : float
            Delay period in milliseconds between the 8th and 9th pulses.
        vclamp : bool or None, optional
            Whether to use voltage clamp. If None, the current setting is used. Default is None.

        """
        if self.input_mode == False:
            self.tstop = self._set_drive_train(input_frequency, delay)
            h.tstop = self.tstop

            vcldur = [[0, 0, 0], [self.general_settings['tstart'], self.tstop, 1e9]]
            for i in range(3):
                self.vcl.amp[i] = self.conn['spec_settings']['vclamp_amp']
                self.vcl.dur[i] = vcldur[1][i]
            h.finitialize(self.cell.Vinit * mV)
            h.continuerun(self.tstop * ms)
        else:
            self.tstop = self.general_settings['tstart'] + self.general_settings['tdur']
            self.nstim.interval = 1000 / input_frequency
            self.nstim.number = np.ceil(self.w_duration.value / 1000 * input_frequency + 1)
            self.nstim2.number = 0
            self.tstop = self.w_duration.value + self.general_settings['tstart']
            
            h.finitialize(self.cell.Vinit * mV)
            h.continuerun(self.tstop * ms)
        
    
    def InteractiveTuner(self):
        """
        Sets up interactive sliders for short-term plasticity (STP) experiments in a Jupyter Notebook.
        
        """
        # Widgets setup (Sliders)
        freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 50, 100, 200]
        delays = [125, 250, 500, 1000, 2000, 4000]
        durations = [300, 500, 1000, 2000, 5000, 10000]
        freq0 = 50
        delay0 = 250
        duration0 = 300
        vlamp_status = self.vclamp

        w_run = widgets.Button(description='Run', icon='history', button_style='primary')
        w_vclamp = widgets.ToggleButton(value=vlamp_status, description='Voltage Clamp', icon='fast-backward', button_style='warning')
        w_input_mode = widgets.ToggleButton(value=False, description='Continuous input', icon='eject', button_style='info')
        w_input_freq = widgets.SelectionSlider(options=freqs, value=freq0, description='Input Freq')

        # Sliders for delay and duration
        self.w_delay = widgets.SelectionSlider(options=delays, value=delay0, description='Delay')
        self.w_duration = widgets.SelectionSlider(options=durations, value=duration0, description='Duration')

        # Generate sliders dynamically based on valid numeric entries in self.slider_vars
        dynamic_sliders = {}
        print("Setting up slider! The sliders ranges are set by their init value so try changing that if you dont like the slider range!")
        for key, value in self.slider_vars.items():
            if isinstance(value, (int, float)):  # Only create sliders for numeric values
                if hasattr(self.syn, key):
                    if value == 0:
                        print(f'{key} was set to zero, going to try to set a range of values, try settings the {key} to a nonzero value if you dont like the range!')
                        slider = widgets.FloatSlider(value=value, min=0, max=1000, step=1, description=key)
                    else:
                        slider = widgets.FloatSlider(value=value, min=0, max=value*20, step=value/5, description=key)
                    dynamic_sliders[key] = slider
                else:
                    print(f"skipping slider for {key} due to not being a synaptic variable")

        # Function to update UI based on input mode
        def update_ui(*args):
            clear_output()
            display(ui)
            self.vclamp = w_vclamp.value
            self.input_mode = w_input_mode.value
            # Update synaptic properties based on slider values
            syn_props = {var: slider.value for var, slider in dynamic_sliders.items()}
            self._set_syn_prop(**syn_props)
            if self.input_mode == False:
                self._simulate_model(w_input_freq.value, self.w_delay.value, w_vclamp.value)
            else:
                self._simulate_model(w_input_freq.value, self.w_duration.value, w_vclamp.value)
            amp = self._response_amplitude()
            self._plot_model([self.general_settings['tstart'] - self.nstim.interval / 3, self.tstop])
            _ = self._calc_ppr_induction_recovery(amp)
            # print('Single trial ' + ('PSC' if self.vclamp else 'PSP'))
            # print(f'Induction: {induction_single:.2f}; Recovery: {recovery:.2f}')
            #print(f'Rest Amp: {amp[0]:.2f}; Maximum Amp: {maxamp:.2f}')

        # Function to switch between delay and duration sliders
        def switch_slider(*args):
            if w_input_mode.value:
                self.w_delay.layout.display = 'none'  # Hide delay slider
                self.w_duration.layout.display = ''   # Show duration slider
            else:
                self.w_delay.layout.display = ''      # Show delay slider
                self.w_duration.layout.display = 'none'  # Hide duration slider

        # Link input mode to slider switch
        w_input_mode.observe(switch_slider, names='value')

        # Hide the duration slider initially until the user selects it
        self.w_duration.layout.display = 'none'  # Hide duration slider

        w_run.on_click(update_ui)

        # Add the dynamic sliders to the UI
        slider_widgets = [slider for slider in dynamic_sliders.values()]

        # Divide sliders into two columns
        half = len(slider_widgets) // 2
        col1 = VBox(slider_widgets[:half])  # First half of sliders
        col2 = VBox(slider_widgets[half:])  # Second half of sliders
        
        # Create a two-column layout with HBox
        slider_columns = HBox([col1, col2])

        ui = VBox([HBox([w_run, w_vclamp, w_input_mode]), HBox([w_input_freq, self.w_delay, self.w_duration]), slider_columns])

        display(ui)
        # run model with default parameters 
        update_ui()

class GapJunctionTuner:
    def __init__(self, mechanisms_dir: str, templates_dir: str, general_settings: dict, conn_type_settings: dict):
        neuron.load_mechanisms(mechanisms_dir)
        h.load_file(templates_dir)
        
        self.general_settings = general_settings
        self.conn_type_settings = conn_type_settings
        
        h.tstop = general_settings['tstart'] + general_settings['tdur'] + 100.
        h.dt = general_settings['dt']  # Time step (resolution) of the simulation in ms
        h.steps_per_ms = 1 / h.dt
        h.celsius = general_settings['celsius']

        self.cell_name = conn_type_settings['cell']
        
        # set up gap junctions
        pc = h.ParallelContext()

        self.cell1 = getattr(h, self.cell_name)()
        self.cell2 = getattr(h, self.cell_name)()
        
        self.icl = h.IClamp(self.cell1.soma[0](0.5))
        self.icl.delay = self.general_settings['tstart'] 
        self.icl.dur = self.general_settings['tdur'] 
        self.icl.amp = self.conn_type_settings['iclamp_amp'] # nA
        
        sec1 = list(self.cell1.all)[conn_type_settings['sec_id']]
        sec2 = list(self.cell2.all)[conn_type_settings['sec_id']]

        pc.source_var(sec1(conn_type_settings['sec_x'])._ref_v, 0, sec=sec1)
        self.gap_junc_1 = h.Gap(sec1(0.5))
        pc.target_var(self.gap_junc_1 ._ref_vgap, 1)

        pc.source_var(sec2(conn_type_settings['sec_x'])._ref_v, 1, sec=sec2)
        self.gap_junc_2 = h.Gap(sec2(0.5))
        pc.target_var(self.gap_junc_2._ref_vgap, 0)

        pc.setup_transfer()
    
    def model(self,resistance):

        self.gap_junc_1.g = resistance
        self.gap_junc_2.g = resistance
        
        t_vec = h.Vector()
        soma_v_1 = h.Vector()
        soma_v_2 = h.Vector()
        t_vec.record(h._ref_t)
        soma_v_1.record(self.cell1.soma[0](0.5)._ref_v)
        soma_v_2.record(self.cell2.soma[0](0.5)._ref_v)
        
        self.t_vec = t_vec
        self.soma_v_1 = soma_v_1
        self.soma_v_2 = soma_v_2
        
        h.finitialize(-70 * mV)
        h.continuerun(h.tstop * ms)
 
 
    def plot_model(self):
        t_range = [self.general_settings['tstart'] - 100., self.general_settings['tstart']+self.general_settings['tdur'] + 100.]
        t = np.array(self.t_vec)
        v1 = np.array(self.soma_v_1)
        v2 = np.array(self.soma_v_2)
        tidx = (t >= t_range[0]) & (t <= t_range[1])

        plt.figure()
        plt.plot(t[tidx], v1[tidx], 'b', label=f'{self.cell_name} 1')
        plt.plot(t[tidx], v2[tidx], 'r', label=f'{self.cell_name} 2')
        plt.title(f"{self.cell_name} gap junction")
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        plt.legend()
        plt.show()        


    def coupling_coefficient(self,t, v1, v2, t_start, t_end, dt=h.dt):
        t = np.asarray(t)
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        idx1 = np.nonzero(t < t_start)[0][-1]
        idx2 = np.nonzero(t < t_end)[0][-1]
        return (v2[idx2] - v2[idx1]) / (v1[idx2] - v1[idx1])

    
    def InteractiveTuner(self):
        w_run = widgets.Button(description='Run', icon='history', button_style='primary')
        values = [i * 10**-4 for i in range(1, 101)]  # From 1e-4 to 1e-2

        # Create the SelectionSlider widget with appropriate formatting
        resistance = widgets.SelectionSlider(
            options=[("%g"%i,i) for i in values],  # Use scientific notation for display
            value=10**-3,  # Default value
            description='Resistance: ',
            continuous_update=True
)

        ui = VBox([w_run,resistance])
        display(ui)
        def on_button(*args):
            clear_output()
            display(ui)
            resistance_for_gap = resistance.value
            self.model(resistance_for_gap)
            self.plot_model()
            cc = self.coupling_coefficient(self.t_vec, self.soma_v_1, self.soma_v_2, 500, 1000)
            print(f"coupling_coefficient is {cc:0.4f}")

        on_button()    
        w_run.on_click(on_button)
        
        
# optimizers!       
        
@dataclass
class SynapseOptimizationResult:
    """Container for synaptic parameter optimization results"""
    optimal_params: Dict[str, float]
    achieved_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    error: float
    optimization_path: List[Dict[str, float]]

class SynapseOptimizer:
    def __init__(self, tuner):
        """
        Initialize the synapse optimizer with parameter scaling
        
        Parameters:
        -----------
        tuner : SynapseTuner
            Instance of the SynapseTuner class
        """
        self.tuner = tuner
        self.optimization_history = []
        self.param_scales = {}
        
    def _normalize_params(self, params: np.ndarray, param_names: List[str]) -> np.ndarray:
        """Normalize parameters to similar scales"""
        return np.array([params[i] / self.param_scales[name] for i, name in enumerate(param_names)])
        
    def _denormalize_params(self, normalized_params: np.ndarray, param_names: List[str]) -> np.ndarray:
        """Convert normalized parameters back to original scale"""
        return np.array([normalized_params[i] * self.param_scales[name] for i, name in enumerate(param_names)])
        
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate standard metrics from the current simulation"""
        self.tuner._simulate_model(50, 250) # 50 Hz with 250ms Delay
        amp = self.tuner._response_amplitude()
        ppr, induction, recovery = self.tuner._calc_ppr_induction_recovery(amp, print_math=False)
        return {
            'induction': float(induction),  # Ensure these are scalar values
            'ppr': float(ppr),
            'recovery': float(recovery),
            'amplitudes': amp
        }
        
    def _default_cost_function(self, metrics: Dict[str, float], target_metrics: Dict[str, float]) -> float:
        """Default cost function that targets induction"""
        return float((metrics['induction'] - target_metrics['induction']) ** 2)
        
    def _objective_function(self, 
                          normalized_params: np.ndarray, 
                          param_names: List[str], 
                          cost_function: Callable,
                          target_metrics: Dict[str, float]) -> float:
        """
        Calculate error using provided cost function
        """
        # Denormalize parameters
        params = self._denormalize_params(normalized_params, param_names)
        
        # Set parameters
        for name, value in zip(param_names, params):
            setattr(self.tuner.syn, name, value)
        
        # Calculate metrics and error
        metrics = self._calculate_metrics()
        error = float(cost_function(metrics, target_metrics))  # Ensure error is scalar
        
        # Store history with denormalized values
        history_entry = {
            'params': dict(zip(param_names, params)),
            'metrics': metrics,
            'error': error
        }
        self.optimization_history.append(history_entry)
        
        return error
    
    def optimize_parameters(self, 
                          target_metrics: Dict[str, float],
                          param_bounds: Dict[str, Tuple[float, float]],
                          cost_function: Optional[Callable] = None,
                          method: str = 'SLSQP',init_guess='random') -> SynapseOptimizationResult:
        """
        Optimize synaptic parameters using custom cost function
        """
        self.optimization_history = []
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]
        
        if cost_function is None:
            cost_function = self._default_cost_function
        
        # Calculate scaling factors
        self.param_scales = {
            name: max(abs(bounds[i][0]), abs(bounds[i][1]))
            for i, name in enumerate(param_names)
        }
        
        # Normalize bounds
        normalized_bounds = [
            (b[0]/self.param_scales[name], b[1]/self.param_scales[name])
            for name, b in zip(param_names, bounds)
        ]
        
        # picks with method of init value we want to use
        if init_guess=='random':
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        elif init_guess=='middle_guess':
            x0 = [(b[0] + b[1])/2 for b in bounds]
        else:
            raise Exception("Pick a vaid init guess method either random or midde_guess")
        normalized_x0 = self._normalize_params(np.array(x0), param_names)
        
        
        # Run optimization
        result = minimize(
            self._objective_function,
            normalized_x0,
            args=(param_names, cost_function, target_metrics),
            method=method,
            bounds=normalized_bounds
        )
        
        # Get final parameters and metrics
        final_params = dict(zip(param_names, self._denormalize_params(result.x, param_names)))
        for name, value in final_params.items():
            setattr(self.tuner.syn, name, value)
        final_metrics = self._calculate_metrics()
        
        return SynapseOptimizationResult(
            optimal_params=final_params,
            achieved_metrics=final_metrics,
            target_metrics=target_metrics,
            error=result.fun,
            optimization_path=self.optimization_history
        )
    
    def plot_optimization_results(self, result: SynapseOptimizationResult):
        """Plot optimization results including convergence and final traces."""
        # Ensure errors are properly shaped for plotting
        iterations = range(len(result.optimization_path))
        errors = np.array([float(h['error']) for h in result.optimization_path]).flatten()
        
        # Plot error convergence
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(iterations, errors, label='Error')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error')
        ax1.set_title('Error Convergence')
        ax1.set_yscale('log')
        ax1.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot parameter convergence
        param_names = list(result.optimal_params.keys())
        num_params = len(param_names)
        fig2, axs = plt.subplots(nrows=num_params, ncols=1, figsize=(8, 5 * num_params))
        
        if num_params == 1:
            axs = [axs]
            
        for ax, param in zip(axs, param_names):
            values = [float(h['params'][param]) for h in result.optimization_path]
            ax.plot(iterations, values, label=f'{param}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Convergence of {param}')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print final results
        print("Optimization Results:")
        print(f"Final Error: {float(result.error):.2e}\n")
        print("Target Metrics:")
        for metric, value in result.target_metrics.items():
            achieved = result.achieved_metrics.get(metric)
            if achieved is not None and metric != 'amplitudes':  # Skip amplitude array
                print(f"{metric}: {float(achieved):.3f} (target: {float(value):.3f})")
        
        print("\nOptimal Parameters:")
        for param, value in result.optimal_params.items():
            print(f"{param}: {float(value):.3f}")
        
        # Plot final model response
        self.tuner._plot_model([self.tuner.general_settings['tstart'] - self.tuner.nstim.interval / 3, self.tuner.tstop])
        amp = self.tuner._response_amplitude()
        self.tuner._calc_ppr_induction_recovery(amp)        
        
        
# dataclass means just init the typehints as self.typehint. looks a bit cleaner
@dataclass
class GapOptimizationResult:
    """Container for gap junction optimization results"""
    optimal_resistance: float
    achieved_cc: float
    target_cc: float
    error: float
    optimization_path: List[Dict[str, float]]

class GapJunctionOptimizer:
    def __init__(self, tuner):
        """
        Initialize the gap junction optimizer
        
        Parameters:
        -----------
        tuner : GapJunctionTuner
            Instance of the GapJunctionTuner class
        """
        self.tuner = tuner
        self.optimization_history = []
        
    def _objective_function(self, resistance: float, target_cc: float) -> float:
        """
        Calculate error between achieved and target coupling coefficient
        
        Parameters:
        -----------
        resistance : float
            Gap junction resistance to try
        target_cc : float
            Target coupling coefficient to match
            
        Returns:
        --------
        float : Error between achieved and target coupling coefficient
        """
        # Run model with current resistance
        self.tuner.model(resistance)
        
        # Calculate coupling coefficient
        achieved_cc = self.tuner.coupling_coefficient(
            self.tuner.t_vec, 
            self.tuner.soma_v_1, 
            self.tuner.soma_v_2,
            self.tuner.general_settings['tstart'],
            self.tuner.general_settings['tstart'] + self.tuner.general_settings['tdur']
        )
        
        # Calculate error
        error = (achieved_cc - target_cc) ** 2 #MSE
        
        # Store history
        self.optimization_history.append({
            'resistance': resistance,
            'achieved_cc': achieved_cc,
            'error': error
        })
        
        return error
    
    def optimize_resistance(self, target_cc: float, 
                          resistance_bounds: tuple = (1e-4, 1e-2),
                          method: str = 'bounded') -> GapOptimizationResult:
        """
        Optimize gap junction resistance to achieve target coupling coefficient
        
        Parameters:
        -----------
        target_cc : float
            Target coupling coefficient to achieve
        resistance_bounds : tuple, optional
            (min, max) bounds for resistance search
        method : str, optional
            Optimization method to use (default: 'bounded')
            
        Returns:
        --------
        GapOptimizationResult
            Container with optimization results
        """
        self.optimization_history = []
        
        # Run optimization
        result = minimize_scalar(
            self._objective_function,
            args=(target_cc,),
            bounds=resistance_bounds,
            method=method
        )
        
        # Run final model with optimal resistance
        self.tuner.model(result.x)
        final_cc = self.tuner.coupling_coefficient(
            self.tuner.t_vec,
            self.tuner.soma_v_1,
            self.tuner.soma_v_2,
            self.tuner.general_settings['tstart'],
            self.tuner.general_settings['tstart'] + self.tuner.general_settings['tdur']
        )
        
        # Package up our results
        optimization_result = GapOptimizationResult(
            optimal_resistance=result.x,
            achieved_cc=final_cc,
            target_cc=target_cc,
            error=result.fun,
            optimization_path=self.optimization_history
        )
        
        return optimization_result
    
    def plot_optimization_results(self, result: GapOptimizationResult):
        """
        Plot optimization results including convergence and final voltage traces
        
        Parameters:
        -----------
        result : GapOptimizationResult
            Results from optimization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot voltage traces
        t_range = [
            self.tuner.general_settings['tstart'] - 100.,
            self.tuner.general_settings['tstart'] + self.tuner.general_settings['tdur'] + 100.
        ]
        t = np.array(self.tuner.t_vec)
        v1 = np.array(self.tuner.soma_v_1)
        v2 = np.array(self.tuner.soma_v_2)
        tidx = (t >= t_range[0]) & (t <= t_range[1])
        
        ax1.plot(t[tidx], v1[tidx], 'b', label=f'{self.tuner.cell_name} 1')
        ax1.plot(t[tidx], v2[tidx], 'r', label=f'{self.tuner.cell_name} 2')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Voltage (mV)')
        ax1.legend()
        ax1.set_title('Optimized Voltage Traces')
        
        # Plot error convergence
        errors = [h['error'] for h in result.optimization_path]
        ax2.plot(errors)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Error')
        ax2.set_title('Error Convergence')
        ax2.set_yscale('log')
        
        # Plot resistance convergence
        resistances = [h['resistance'] for h in result.optimization_path]
        ax3.plot(resistances)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Resistance')
        ax3.set_title('Resistance Convergence')
        ax3.set_yscale('log')
        
        # Print final results
        result_text = (
            f'Optimal Resistance: {result.optimal_resistance:.2e}\n'
            f'Target CC: {result.target_cc:.3f}\n'
            f'Achieved CC: {result.achieved_cc:.3f}\n'
            f'Final Error: {result.error:.2e}'
        )
        ax4.text(0.1, 0.7, result_text, transform=ax4.transAxes, fontsize=10)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

    def parameter_sweep(self, resistance_range: np.ndarray) -> dict:
        """
        Perform a parameter sweep across different resistance values
        
        Parameters:
        -----------
        resistance_range : np.ndarray
            Array of resistance values to test
            
        Returns:
        --------
        dict : Results of parameter sweep including coupling coefficients
        """
        results = {
            'resistance': [],
            'coupling_coefficient': []
        }
        
        for resistance in tqdm(resistance_range, desc="Sweeping resistance values"):
            self.tuner.model(resistance)
            cc = self.tuner.coupling_coefficient(
                self.tuner.t_vec,
                self.tuner.soma_v_1,
                self.tuner.soma_v_2,
                self.tuner.general_settings['tstart'],
                self.tuner.general_settings['tstart'] + self.tuner.general_settings['tdur']
            )
            
            results['resistance'].append(resistance)
            results['coupling_coefficient'].append(cc)
            
        return results