import os
import json
import numpy as np
import neuron
from neuron import h
from neuron.units import ms, mV
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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
            self.update_spec_syn_param(json_folder_path)
        self.general_settings = general_settings
        self.conn = self.conn_type_settings[connection]
        self.synaptic_props = self.conn['spec_syn_param']
        self.vclamp = general_settings['vclamp']
        self.current_name = current_name
        self.other_vars_to_record = other_vars_to_record

        if slider_vars:
            self.slider_vars = {key: value for key, value in self.synaptic_props.items() if key in slider_vars} # filters dict to have only the entries that have a key in the sliders var
        else:
            self.slider_vars = self.synaptic_props

        h.tstop = general_settings['tstart'] + general_settings['tdur']
        h.dt = general_settings['dt']  # Time step (resolution) of the simulation in ms
        h.steps_per_ms = 1 / h.dt
        h.celsius = general_settings['celsius']

    def update_spec_syn_param(self, json_folder_path):
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


    def set_up_cell(self):
        """
        Set up the neuron cell based on the specified connection settings.
        """
        self.cell = getattr(h, self.conn['spec_settings']['post_cell'])()


    def set_up_synapse(self):
        """
        Set up the synapse on the target cell according to the synaptic parameters in `conn_type_settings`.
        
        Notes:
        ------
        - `set_up_cell()` should be called before setting up the synapse.
        - Synapse location, type, and properties are specified within `spec_syn_param` and `spec_settings`.
        """
        self.syn = getattr(h, self.conn['spec_syn_param']['level_of_detail'])(list(self.cell.all)[self.conn['spec_settings']['sec_id']](self.conn['spec_settings']['sec_x']))
        for key, value in self.conn['spec_syn_param'].items():
            if isinstance(value, (int, float)):  # Only create sliders for numeric values
                if hasattr(self.syn, key):
                    setattr(self.syn, key, value)
                else:
                    print(f"Warning: {key} cannot be assigned as it does not exist in the synapse. Check your mod file or spec_syn_param.")


    def set_up_recorders(self):
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
        self.set_up_cell()
        self.set_up_synapse()

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

        self.set_up_recorders()

        # Run simulation
        h.tstop = self.general_settings['tstart'] + self.general_settings['tdur']
        self.nstim.interval = self.general_settings['tdur']
        self.nstim.number = 1
        self.nstim2.start = h.tstop
        h.run()
        self.plot_model([self.general_settings['tstart'] - 5, self.general_settings['tstart'] + self.general_settings['tdur']])
        syn_props = self.get_syn_prop(rise_interval=self.general_settings['rise_interval'])     
        for prop in syn_props.items():
            print(prop)


    def find_first(self, x):
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


    def get_syn_prop(self, rise_interval=(0.2, 0.8), dt=h.dt, short=False):
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
        istart = self.find_first(np.diff(isyn[:ipk + 1]) > 0)
        latency = dt * (istart + 1)
        # rise time
        rt1 = self.find_first(isyn[istart:ipk + 1] > rise_interval[0] * peak)
        rt2 = self.find_first(isyn[istart:ipk + 1] > rise_interval[1] * peak)
        rise_time = (rt2 - rt1) * dt
        # decay time
        iend = self.find_first(np.diff(isyn[ipk:]) > 0)
        iend = isyn.size - 1 if iend is None else iend + ipk
        decay_len = iend - ipk + 1
        popt, _ = curve_fit(lambda t, a, tau: a * np.exp(-t / tau), dt * np.arange(decay_len),
                            isyn[ipk:iend + 1], p0=(peak, dt * decay_len / 2))
        decay_time = popt[1]
        # half-width
        hw1 = self.find_first(isyn[istart:ipk + 1] > 0.5 * peak)
        hw2 = self.find_first(isyn[ipk:] < 0.5 * peak)
        hw2 = isyn.size if hw2 is None else hw2 + ipk
        half_width = dt * (hw2 - hw1)
        output = {'baseline': baseline, 'sign': sign, 'latency': latency,
            'amp': peak, 'rise_time': rise_time, 'decay_time': decay_time, 'half_width': half_width}
        return output


    def plot_model(self, xlim):
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
        axs[0].plot(self.t, 1000 * self.rec_vectors[self.current_name])
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

        plt.tight_layout()
        plt.show()


    def set_drive_train(self,freq=50., delay=250.):
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
 

    def response_amplitude(self):
        """
        Calculates the amplitude of the synaptic response by analyzing the recorded synaptic current.

        Returns:
        --------
        amp : list
            A list containing the peak amplitudes for each segment of the recorded synaptic current.
        
        """
        isyn = np.asarray(self.rec_vectors['i'])
        tspk = np.append(np.asarray(self.tspk), h.tstop)
        syn_prop = self.get_syn_prop(short=True)
        # print("syn_prp[sign] = " + str(syn_prop['sign']))
        isyn = (isyn - syn_prop['baseline']) 
        isyn *= syn_prop['sign']
        # print(isyn)
        ispk = np.floor((tspk + self.general_settings['delay']) / h.dt).astype(int)
        amp = [isyn[ispk[i]:ispk[i + 1]].max() for i in range(ispk.size - 1)]
        return amp


    def find_max_amp(self, amp, normalize_by_trial=True):
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


    def induction_recovery(self,amp, normalize_by_trial=True):
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
        amp = amp.reshape(-1, amp.shape[-1])
        
        
        maxamp = amp.max(axis=1 if normalize_by_trial else None)
        induction = np.mean((amp[:, 5:8].mean(axis=1) - amp[:, :1].mean(axis=1)) / maxamp)
        recovery = np.mean((amp[:, 8:12].mean(axis=1) - amp[:, :4].mean(axis=1)) / maxamp)

        # maxamp = max(amp, key=lambda x: abs(x[0]))
        maxamp = maxamp.max()
        return induction, recovery, maxamp


    def paired_pulse_ratio(self, dt=h.dt):
        """
        Computes the paired-pulse ratio (PPR) based on the recorded synaptic current or voltage.

        Parameters:
        -----------
        dt : float, optional
            Time step in milliseconds. Default is the NEURON simulation time step.

        Returns:
        --------
        ppr : float
            The ratio between the second and first pulse amplitudes.

        Notes:
        ------
        - The function handles both voltage-clamp and current-clamp conditions.
        - A minimum of two spikes is required to calculate PPR.
        """
        if self.vclamp:
            isyn = self.ivcl
        else:
            isyn = self.rec_vectors['i']
        isyn = np.asarray(isyn)
        tspk = np.asarray(self.tspk)
        if tspk.size < 2:
            raise ValueError("Need at least two spikes.")
        syn_prop = self.get_syn_prop()
        isyn = (isyn - syn_prop['baseline']) * syn_prop['sign']
        ispk2 = int(np.floor(tspk[1] / dt))
        ipk, _ = find_peaks(isyn[ispk2:])
        ipk2 = ipk[0] + ispk2
        peak2 = isyn[ipk2]
        return peak2 / syn_prop['amp']


    def set_syn_prop(self, **kwargs):
        """
        Sets the synaptic parameters based on user inputs from sliders.
        
        Parameters:
        -----------
        **kwargs : dict
            Synaptic properties (such as weight, Use, tau_f, tau_d) as keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self.syn, key, value)


    def simulate_model(self,input_frequency, delay, vclamp=None):
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
            self.tstop = self.set_drive_train(input_frequency, delay)
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
        
        Notes:
        ------
        - The sliders allow control over synaptic properties dynamically based on slider_vars.
        - Additional buttons allow running the simulation and configuring voltage clamp settings.
        """
        # Widgets setup (Sliders)
        freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]
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
            self.set_syn_prop(**syn_props)
            if self.input_mode == False:
                self.simulate_model(w_input_freq.value, self.w_delay.value, w_vclamp.value)
            else:
                self.simulate_model(w_input_freq.value, self.w_duration.value, w_vclamp.value)
            self.plot_model([self.general_settings['tstart'] - self.nstim.interval / 3, self.tstop])
            amp = self.response_amplitude()
            induction_single, recovery, maxamp = self.induction_recovery(amp)
            ppr = self.paired_pulse_ratio()
            print('Paired Pulse Ratio using ' + ('PSC' if self.vclamp else 'PSP') + f': {ppr:.3f}')
            print('Single trial ' + ('PSC' if self.vclamp else 'PSP'))
            print(f'Induction: {induction_single:.2f}; Recovery: {recovery:.2f}')
            print(f'Rest Amp: {amp[0]:.2f}; Maximum Amp: {maxamp:.2f}')

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

        # Hide the duration slider initially
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




    