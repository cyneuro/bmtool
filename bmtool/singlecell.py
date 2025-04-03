import glob
import os
import json
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import neuron
from neuron import h
import pandas as pd


def load_biophys1():
    """
    Load the Biophys1 template from BMTK if it hasn't been loaded yet.
    
    This function checks if the Biophys1 object exists in NEURON's h namespace.
    If not, it loads the necessary HOC files for Allen Cell Types Database models.
    
    Notes:
    ------
    This is primarily used for working with cell models from the Allen Cell Types Database.
    """
    if not hasattr(h, 'Biophys1'):
        from bmtk import utils
        module_dir = os.path.dirname(os.path.abspath(utils.__file__))
        hoc_file = os.path.join(module_dir, 'scripts', 'bionet', 'templates', 'Biophys1.hoc')
        h.load_file("import3d.hoc")
        h.load_file(hoc_file)


def load_allen_database_cells(morphology, dynamic_params, model_processing='aibs_perisomatic'):
    """
    Create a cell model from the Allen Cell Types Database.
    
    Parameters:
    -----------
    morphology : str
        Path to the morphology file (SWC or ASC format).
    dynamic_params : str
        Path to the JSON file containing biophysical parameters.
    model_processing : str, optional
        Model processing type from the AllenCellType database.
        Default is 'aibs_perisomatic'.
    
    Returns:
    --------
    callable
        A function that, when called, creates and returns a NEURON cell object
        with the specified morphology and biophysical properties.
        
    Notes:
    ------
    This function creates a closure that loads and returns a cell when called.
    The cell is created using the Allen Institute's modeling framework.
    """
    from bmtk.simulator.bionet.default_setters import cell_models
    load_biophys1()
    model_processing = getattr(cell_models, model_processing)
    with open(dynamic_params) as f:
        dynamics_params = json.load(f)
    def create_cell():
        hobj = h.Biophys1(morphology)
        hobj = model_processing(hobj, cell=None, dynamics_params=dynamics_params)
        return hobj
    return create_cell


def get_target_site(cell, sec=('soma', 0), loc=0.5, site=''):
    """
    Get a segment and its section from a cell model using flexible section specification.
    
    Parameters:
    -----------
    cell : NEURON cell object
        The cell object to access sections from.
    sec : str, int, or tuple, optional
        Section specification, which can be:
        - str: Section name (defaults to index 0 if multiple sections)
        - int: Index into the 'all' section list
        - tuple: (section_name, index) for accessing indexed sections
        Default is ('soma', 0).
    loc : float, optional
        Location along the section (0-1), default is 0.5 (middle of section).
    site : str, optional
        Name of the site for error messages (e.g., 'injection', 'recording').
    
    Returns:
    --------
    tuple
        (segment, section) at the specified location
        
    Raises:
    -------
    ValueError
        If the section cannot be found or accessed.
    """
    if isinstance(sec, str):
        sec = (sec, 0)
    elif isinstance(sec, int):
        if not hasattr(cell, 'all'):
            raise ValueError("Section list named 'all' does not exist in the template.")
        sec = ('all', sec)
    loc = float(loc)
    try:
        section = next(s for i, s in enumerate(getattr(cell, sec[0])) if i == sec[1])
        seg = section(loc)
    except Exception as e0:
        try:
            section = eval("cell." + sec[0])
            seg = section(loc)
        except Exception as e:
            print(e0)
            print(e)
            raise ValueError("Hint: Are you selecting the correct " + site + " location?")
    return seg, section


class CurrentClamp(object):
    def __init__(self, template_name, post_init_function=None, record_sec='soma', record_loc=0.5, threshold=None,
                 inj_sec='soma', inj_loc=0.5, inj_amp=100., inj_delay=100., inj_dur=1000., tstop=1000.):
        """
        Initialize a current clamp simulation environment.
        
        Parameters:
        -----------
        template_name : str or callable
            Either the name of the cell template located in HOC or
            a function that creates and returns a cell object.
        post_init_function : str, optional
            Function of the cell to be called after initialization.
        record_sec : str, int, or tuple, optional
            Section to record from. Can be:
            - str: Section name (defaults to index 0 if multiple sections)
            - int: Index into the 'all' section list
            - tuple: (section_name, index) for accessing indexed sections
            Default is 'soma'.
        record_loc : float, optional
            Location (0-1) within section to record from. Default is 0.5.
        threshold : float, optional
            Spike threshold (mV). If specified, spikes are detected and counted.
        inj_sec : str, int, or tuple, optional
            Section for current injection. Same format as record_sec. Default is 'soma'.
        inj_loc : float, optional
            Location (0-1) within section for current injection. Default is 0.5.
        inj_amp : float, optional
            Current injection amplitude (pA). Default is 100.0.
        inj_delay : float, optional
            Start time for current injection (ms). Default is 100.0.
        inj_dur : float, optional
            Duration of current injection (ms). Default is 1000.0.
        tstop : float, optional
            Total simulation time (ms). Default is 1000.0.
            Will be extended if necessary to include the full current injection.
        """
        self.create_cell = getattr(h, template_name) if isinstance(template_name, str) else template_name
        self.record_sec = record_sec
        self.record_loc = record_loc
        self.inj_sec = inj_sec
        self.inj_loc = inj_loc
        self.threshold = threshold

        self.tstop = max(tstop, inj_delay + inj_dur)
        self.inj_delay = inj_delay # use x ms after start of inj to calculate r_in, etc
        self.inj_dur = inj_dur
        self.inj_amp = inj_amp * 1e-3 # pA to nA

        # sometimes people may put a hoc object in for the template name 
        if callable(template_name):
            self.cell = template_name()
        else:    
            self.cell = self.create_cell()
        if post_init_function:
            eval(f"self.cell.{post_init_function}")

        self.setup()

    def setup(self):
        """
        Set up the simulation environment for current clamp experiments.
        
        This method:
        1. Creates the current clamp stimulus at the specified injection site
        2. Sets up voltage recording at the specified recording site
        3. Creates vectors to store time and voltage data
        
        Notes:
        ------
        Sets self.cell_src as the current clamp object that can be accessed later.
        """
        inj_seg, _ = get_target_site(self.cell, self.inj_sec, self.inj_loc, 'injection')
        self.cell_src = h.IClamp(inj_seg)
        self.cell_src.delay = self.inj_delay
        self.cell_src.dur = self.inj_dur
        self.cell_src.amp = self.inj_amp

        rec_seg, rec_sec = get_target_site(self.cell, self.record_sec, self.record_loc, 'recording')
        self.v_vec = h.Vector()
        self.v_vec.record(rec_seg._ref_v)

        self.t_vec = h.Vector()
        self.t_vec.record(h._ref_t)

        if self.threshold is not None:
            self.nc = h.NetCon(rec_seg._ref_v, None, sec=rec_sec)
            self.nc.threshold = self.threshold
            self.tspk_vec = h.Vector()
            self.nc.record(self.tspk_vec)

        print(f'Injection location: {inj_seg}')
        print(f'Recording: {rec_seg}._ref_v')

    def execute(self) -> Tuple[list, list]:
        """
        Run the current clamp simulation and return recorded data.
        
        This method:
        1. Sets up the simulation duration
        2. Initializes and runs the NEURON simulation
        3. Converts recorded vectors to Python lists
        
        Returns:
        --------
        tuple
            (time_vector, voltage_vector) where:
            - time_vector: List of time points (ms)
            - voltage_vector: List of membrane potentials (mV) at those time points
        """
        print("Current clamp simulation running...")
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        if self.threshold is not None:
            self.nspks = len(self.tspk_vec)
            print()
            print(f'Number of spikes: {self.nspks:d}')
            print()
        return self.t_vec.to_python(), self.v_vec.to_python()


class Passive(CurrentClamp):
    def __init__(self, template_name, inj_amp=-100., inj_delay=200., inj_dur=1000.,
                 tstop=1200., method=None, **kwargs):
        """
        Initialize a passive membrane property simulation environment.
        
        Parameters:
        -----------
        template_name : str or callable
            Either the name of the cell template located in HOC or
            a function that creates and returns a cell object.
        inj_amp : float, optional
            Current injection amplitude (pA). Default is -100.0 (negative to measure passive properties).
        inj_delay : float, optional
            Start time for current injection (ms). Default is 200.0.
        inj_dur : float, optional
            Duration of current injection (ms). Default is 1000.0.
        tstop : float, optional
            Total simulation time (ms). Default is 1200.0.
        method : str, optional
            Method to estimate membrane time constant:
            - 'simple': Find the time to reach 0.632 of voltage change
            - 'exp': Fit a single exponential curve
            - 'exp2': Fit a double exponential curve
            Default is None, which uses 'simple' when calculations are performed.
        **kwargs : 
            Additional keyword arguments to pass to the parent CurrentClamp constructor.
            
        Notes:
        ------
        This class is designed for measuring passive membrane properties including
        input resistance and membrane time constant.
        
        Raises:
        -------
        AssertionError
            If inj_amp is zero (must be non-zero to measure passive properties).
        """
        assert(inj_amp != 0)
        super().__init__(template_name=template_name, tstop=tstop,
                         inj_amp=inj_amp, inj_delay=inj_delay, inj_dur=inj_dur, **kwargs)
        self.inj_stop = inj_delay + inj_dur
        self.method = method
        self.tau_methods = {'simple': self.tau_simple, 'exp2': self.tau_double_exponential, 'exp': self.tau_single_exponential}

    def tau_simple(self):
        """
        Calculate membrane time constant using the simple 0.632 criterion method.
        
        This method calculates the membrane time constant by finding the time it takes
        for the membrane potential to reach 63.2% (1-1/e) of its final value after 
        a step current injection.
        
        Returns:
        --------
        callable
            A function that prints the calculation details when called.
        
        Notes:
        ------
        Sets the following attributes:
        - tau: The calculated membrane time constant in ms
        """
        v_t_const = self.cell_v_final - self.v_diff / np.e
        index_v_tau = next(x for x, val in enumerate(self.v_vec_inj) if val <= v_t_const)
        self.tau = self.t_vec[self.index_v_rest + index_v_tau] - self.v_rest_time  # ms

        def print_calc():
            print()
            print('Tau Calculation: time until 63.2% of dV')
            print('v_rest + 0.632*(v_final-v_rest)')
            print(f'{self.v_rest:.2f} + 0.632*({self.cell_v_final:.2f}-({self.v_rest:.2f})) = {v_t_const:.2f} (mV)')
            print(f'Time where V = {v_t_const:.2f} (mV) is {self.v_rest_time + self.tau:.2f} (ms)')
            print(f'{self.v_rest_time + self.tau:.2f} - {self.v_rest_time:g} = {self.tau:.2f} (ms)')
            print()
        return print_calc

    @staticmethod
    def single_exponential(t, a0, a, tau):
        """
        Single exponential function for fitting membrane potential response.
        
        Parameters:
        -----------
        t : array-like
            Time values
        a0 : float
            Offset (steady-state) value
        a : float
            Amplitude of the exponential component
        tau : float
            Time constant of the exponential decay
            
        Returns:
        --------
        array-like
            Function values at the given time points
        """
        return a0 + a * np.exp(-t / tau)

    def tau_single_exponential(self):
        """
        Calculate membrane time constant by fitting a single exponential curve.
        
        This method:
        1. Identifies the peak response (for sag characterization)
        2. Falls back to simple method for initial estimate
        3. Fits a single exponential function to the membrane potential response
        4. Sets tau to the exponential time constant
        
        Returns:
        --------
        callable
            A function that prints the calculation details when called.
            
        Notes:
        ------
        Sets the following attributes:
        - tau: The calculated membrane time constant in ms
        - t_peak, v_peak: Time and voltage of peak response
        - v_sag: Sag potential (difference between peak and steady-state)
        - v_max_diff: Maximum potential difference from rest
        - sag_norm: Normalized sag ratio
        - popt: Optimized parameters from curve fitting
        - pcov: Covariance matrix of the optimization
        """
        index_v_peak = (np.sign(self.inj_amp) * self.v_vec_inj).argmax()
        self.t_peak = self.t_vec_inj[index_v_peak]
        self.v_peak = self.v_vec_inj[index_v_peak]
        self.v_sag = self.v_peak - self.cell_v_final
        self.v_max_diff = self.v_diff + self.v_sag
        self.sag_norm = self.v_sag / self.v_max_diff

        self.tau_simple()

        p0 = (self.v_diff, self.tau)  # initial estimate
        v0 = self.v_rest
        def fit_func(t, a, tau):
            return self.single_exponential(t, a0=v0 - a, a=a, tau=tau)
        bounds = ((-np.inf, 1e-3), np.inf)
        popt, self.pcov = curve_fit(fit_func, self.t_vec_inj, self.v_vec_inj, p0=p0, bounds=bounds, maxfev=10000)
        self.popt = np.insert(popt, 0, v0 - popt[0])
        self.tau = self.popt[2]

        def print_calc():
            print()
            print('Tau Calculation: Fit a single exponential curve to the membrane potential response')
            print('f(t) = a0 + a*exp(-t/tau)')
            print(f'Fit parameters: (a0, a, tau) = ({self.popt[0]:.2f}, {self.popt[1]:.2f}, {self.popt[2]:.2f})')
            print(f'Membrane time constant is determined from the exponential term: {self.tau:.2f} (ms)')
            print()
            print('Sag potential: v_sag = v_peak - v_final = %.2f (mV)' % self.v_sag)
            print('Normalized sag potential: v_sag / (v_peak - v_rest) = %.3f' % self.sag_norm)
            print()
        return print_calc

    @staticmethod
    def double_exponential(t, a0, a1, a2, tau1, tau2):
        """
        Double exponential function for fitting membrane potential response.
        
        This function is particularly useful for modeling cells with sag responses,
        where the membrane potential shows two distinct time constants.
        
        Parameters:
        -----------
        t : array-like
            Time values
        a0 : float
            Offset (steady-state) value
        a1 : float
            Amplitude of the first exponential component
        a2 : float
            Amplitude of the second exponential component
        tau1 : float
            Time constant of the first exponential component
        tau2 : float
            Time constant of the second exponential component
            
        Returns:
        --------
        array-like
            Function values at the given time points
        """
        return a0 + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    def tau_double_exponential(self):
        """
        Calculate membrane time constant by fitting a double exponential curve.
        
        This method is useful for cells with sag responses that cannot be
        fitted well with a single exponential.
        
        Returns:
        --------
        callable
            A function that prints the calculation details when called.
            
        Notes:
        ------
        Sets the following attributes:
        - tau: The calculated membrane time constant (the slower of the two time constants)
        - t_peak, v_peak: Time and voltage of peak response
        - v_sag: Sag potential (difference between peak and steady-state)
        - v_max_diff: Maximum potential difference from rest
        - sag_norm: Normalized sag ratio
        - popt: Optimized parameters from curve fitting
        - pcov: Covariance matrix of the optimization
        """
        index_v_peak = (np.sign(self.inj_amp) * self.v_vec_inj).argmax()
        self.t_peak = self.t_vec_inj[index_v_peak]
        self.v_peak = self.v_vec_inj[index_v_peak]
        self.v_sag = self.v_peak - self.cell_v_final
        self.v_max_diff = self.v_diff + self.v_sag
        self.sag_norm = self.v_sag / self.v_max_diff

        self.tau_simple()
        p0 = (self.v_sag, -self.v_max_diff, self.t_peak, self.tau)  # initial estimate
        v0 = self.v_rest
        def fit_func(t, a1, a2, tau1, tau2):
            return self.double_exponential(t, v0 - a1 - a2, a1, a2, tau1, tau2)
        bounds = ((-np.inf, -np.inf, 1e-3, 1e-3), np.inf)
        popt, self.pcov = curve_fit(fit_func, self.t_vec_inj, self.v_vec_inj, p0=p0, bounds=bounds, maxfev=10000)
        self.popt = np.insert(popt, 0, v0 - sum(popt[:2]))
        self.tau = max(self.popt[-2:])

        def print_calc():
            print()
            print('Tau Calculation: Fit a double exponential curve to the membrane potential response')
            print('f(t) = a0 + a1*exp(-t/tau1) + a2*exp(-t/tau2)')
            print('Constrained by initial value: f(0) = a0 + a1 + a2 = v_rest')
            print('Fit parameters: (a0, a1, a2, tau1, tau2) = (' + ', '.join(f'{x:.2f}' for x in self.popt) + ')')
            print(f'Membrane time constant is determined from the slowest exponential term: {self.tau:.2f} (ms)')
            print()
            print('Sag potential: v_sag = v_peak - v_final = %.2f (mV)' % self.v_sag)
            print('Normalized sag potential: v_sag / (v_peak - v_rest) = %.3f' % self.sag_norm)
            print()
        return print_calc

    def double_exponential_fit(self):
        """
        Get the double exponential fit values for plotting.
        
        Returns:
        --------
        tuple
            (time_vector, fitted_values) where:
            - time_vector: Time points starting from rest time
            - fitted_values: Membrane potential values predicted by the double exponential function
        """
        t_vec = self.v_rest_time + self.t_vec_inj
        v_fit = self.double_exponential(self.t_vec_inj, *self.popt)
        return t_vec, v_fit
    
    def single_exponential_fit(self):
        """
        Get the single exponential fit values for plotting.
        
        Returns:
        --------
        tuple
            (time_vector, fitted_values) where:
            - time_vector: Time points starting from rest time
            - fitted_values: Membrane potential values predicted by the single exponential function
        """
        t_vec = self.v_rest_time + self.t_vec_inj
        v_fit = self.single_exponential(self.t_vec_inj, *self.popt)
        return t_vec, v_fit

    def execute(self):
        """
        Run the simulation and calculate passive membrane properties.
        
        This method:
        1. Runs the NEURON simulation
        2. Extracts membrane potential at rest and steady-state
        3. Calculates input resistance from the step response
        4. Calculates membrane time constant using the specified method
        5. Prints detailed calculations for educational purposes
        
        Returns:
        --------
        tuple
            (time_vector, voltage_vector) from the simulation
        
        Notes:
        ------
        Sets several attributes including:
        - v_rest: Resting membrane potential
        - r_in: Input resistance in MOhms
        - tau: Membrane time constant in ms
        """
        print("Running simulation for passive properties...")
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        self.index_v_rest = int(self.inj_delay / h.dt)
        self.index_v_final = int(self.inj_stop / h.dt)
        self.v_rest = self.v_vec[self.index_v_rest]
        self.v_rest_time = self.t_vec[self.index_v_rest]
        self.cell_v_final = self.v_vec[self.index_v_final]
        self.v_final_time = self.t_vec[self.index_v_final]

        t_idx = slice(self.index_v_rest, self.index_v_final + 1)
        self.v_vec_inj = np.array(self.v_vec)[t_idx]
        self.t_vec_inj = np.array(self.t_vec)[t_idx] - self.v_rest_time

        self.v_diff = self.cell_v_final - self.v_rest
        self.r_in = self.v_diff / self.inj_amp  # MegaOhms

        print_calc = self.tau_methods.get(self.method, self.tau_simple)()

        print()
        print(f'V Rest: {self.v_rest:.2f} (mV)')
        print(f'Resistance: {self.r_in:.2f} (MOhms)')
        print(f'Membrane time constant: {self.tau:.2f} (ms)')
        print()
        print(f'V_rest Calculation: Voltage taken at time {self.v_rest_time:.1f} (ms) is')
        print(f'{self.v_rest:.2f} (mV)')
        print()
        print('R_in Calculation: dV/dI = (v_final-v_rest)/(i_final-i_start)')
        print(f'({self.cell_v_final:.2f} - ({self.v_rest:.2f})) / ({self.inj_amp:g} - 0)')
        print(f'{np.sign(self.inj_amp) * self.v_diff:.2f} (mV) / {np.abs(self.inj_amp)} (nA) = {self.r_in:.2f} (MOhms)')
        print_calc()

        return self.t_vec.to_python(), self.v_vec.to_python()


class FI(object):
    def __init__(self, template_name, post_init_function=None,
                 i_start=0., i_stop=1050., i_increment=100., tstart=50., tdur=1000., threshold=0.,
                 record_sec='soma', record_loc=0.5, inj_sec='soma', inj_loc=0.5):
        """
        Initialize a frequency-current (F-I) curve simulation environment.
        
        Parameters:
        -----------
        template_name : str or callable
            Either the name of the cell template located in HOC or
            a function that creates and returns a cell object.
        post_init_function : str, optional
            Function of the cell to be called after initialization.
        i_start : float, optional
            Initial current injection amplitude (pA). Default is 0.0.
        i_stop : float, optional
            Maximum current injection amplitude (pA). Default is 1050.0.
        i_increment : float, optional
            Amplitude increment between trials (pA). Default is 100.0.
        tstart : float, optional
            Current injection start time (ms). Default is 50.0.
        tdur : float, optional
            Current injection duration (ms). Default is 1000.0.
        threshold : float, optional
            Spike threshold (mV). Default is 0.0.
        record_sec : str, int, or tuple, optional
            Section to record from. Same format as in CurrentClamp. Default is 'soma'.
        record_loc : float, optional
            Location (0-1) within section to record from. Default is 0.5.
        inj_sec : str, int, or tuple, optional
            Section for current injection. Same format as record_sec. Default is 'soma'.
        inj_loc : float, optional
            Location (0-1) within section for current injection. Default is 0.5.
            
        Notes:
        ------
        This class creates multiple instances of the cell model, one for each
        current amplitude to be tested, allowing all simulations to be run
        in a single call to NEURON's run() function.
        """
        self.create_cell = getattr(h, template_name) if isinstance(template_name, str) else template_name
        self.post_init_function = post_init_function
        self.i_start = i_start * 1e-3 # pA to nA
        self.i_stop = i_stop * 1e-3
        self.i_increment = i_increment * 1e-3
        self.tstart = tstart
        self.tdur = tdur
        self.tstop = tstart + tdur
        self.threshold = threshold

        self.record_sec = record_sec
        self.record_loc = record_loc
        self.inj_sec = inj_sec
        self.inj_loc = inj_loc

        self.cells = []
        self.sources = []
        self.ncs = []
        self.tspk_vecs = []
        self.nspks = []

        self.ntrials = int((self.i_stop - self.i_start) // self.i_increment + 1)
        self.amps = (self.i_start + np.arange(self.ntrials) * self.i_increment).tolist()
        for _ in range(self.ntrials):
            # Cell definition
            cell = self.create_cell()
            if post_init_function:
                eval(f"cell.{post_init_function}")
            self.cells.append(cell)

        self.setup()

    def setup(self):
        """
        Set up the simulation environment for frequency-current (F-I) analysis.
        
        For each current amplitude to be tested, this method:
        1. Creates a current source at the injection site
        2. Sets up spike detection at the recording site
        3. Creates vectors to record spike times
        
        Notes:
        ------
        This preparation allows multiple simulations to be run with different
        current amplitudes in a single call to h.run().
        """
        for cell, amp in zip(self.cells, self.amps):
            inj_seg, _ = get_target_site(cell, self.inj_sec, self.inj_loc, 'injection')
            src = h.IClamp(inj_seg)
            src.delay = self.tstart
            src.dur = self.tdur
            src.amp = amp
            self.sources.append(src)

            rec_seg, rec_sec = get_target_site(cell, self.record_sec, self.record_loc, 'recording')
            nc = h.NetCon(rec_seg._ref_v, None, sec=rec_sec)
            nc.threshold = self.threshold
            spvec = h.Vector()
            nc.record(spvec)
            self.ncs.append(nc)
            self.tspk_vecs.append(spvec)

        print(f'Injection location: {inj_seg}')
        print(f'Recording: {rec_seg}._ref_v')

    def execute(self):
        """
        Run the simulation and count spikes for each current amplitude.
        
        This method:
        1. Initializes and runs a single NEURON simulation that evaluates all current amplitudes
        2. Counts spikes for each current amplitude
        3. Prints a summary of results in tabular format
        
        Returns:
        --------
        tuple
            (current_amplitudes, spike_counts) where:
            - current_amplitudes: List of current injection amplitudes (nA)
            - spike_counts: List of spike counts corresponding to each amplitude
        """
        print("Running simulations for FI curve...")
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        self.nspks = [len(v) for v in self.tspk_vecs]
        print()
        print("Results")
        # lets make a df so the results line up nice
        data = {'Injection (nA):':self.amps,'number of spikes':self.nspks}
        df = pd.DataFrame(data)
        print(df)
        #print(f'Injection (nA): ' + ', '.join(f'{x:g}' for x in self.amps))
        #print(f'Number of spikes: ' + ', '.join(f'{x:d}' for x in self.nspks))
        print()

        return self.amps, self.nspks


class ZAP(CurrentClamp):
    def __init__(self, template_name, inj_amp=100., inj_delay=200., inj_dur=15000.,
                 tstop=15500., fstart=0., fend=15., chirp_type=None, **kwargs):
        """
        Initialize a ZAP (impedance amplitude profile) simulation environment.
        
        Parameters:
        -----------
        template_name : str or callable
            Either the name of the cell template located in HOC or
            a function that creates and returns a cell object.
        inj_amp : float, optional
            Current injection amplitude (pA). Default is 100.0.
        inj_delay : float, optional
            Start time for current injection (ms). Default is 200.0.
        inj_dur : float, optional
            Duration of current injection (ms). Default is 15000.0.
        tstop : float, optional
            Total simulation time (ms). Default is 15500.0.
        fstart : float, optional
            Starting frequency of the chirp current (Hz). Default is 0.0.
        fend : float, optional
            Ending frequency of the chirp current (Hz). Default is 15.0.
        chirp_type : str, optional
            Type of chirp current determining how frequency increases over time:
            - 'linear': Linear increase in frequency (default if None)
            - 'exponential': Exponential increase in frequency
        **kwargs : 
            Additional keyword arguments to pass to the parent CurrentClamp constructor.
            
        Notes:
        ------
        This class is designed for measuring the frequency-dependent impedance profile
        of a neuron using a chirp current that sweeps through frequencies.
        
        Raises:
        -------
        AssertionError
            - If inj_amp is zero
            - If chirp_type is 'exponential' and either fstart or fend is <= 0
        """
        assert(inj_amp != 0)
        super().__init__(template_name=template_name, tstop=tstop,
                         inj_amp=inj_amp, inj_delay=inj_delay, inj_dur=inj_dur, **kwargs)
        self.inj_stop = inj_delay + inj_dur
        self.fstart = fstart
        self.fend = fend
        self.chirp_type = chirp_type
        self.chirp_func = {'linear': self.linear_chirp, 'exponential': self.exponential_chirp}
        if chirp_type=='exponential':
            assert(fstart > 0 and fend > 0)

    def linear_chirp(self, t, f0, f1):
        """
        Generate a chirp current with linearly increasing frequency.
        
        Parameters:
        -----------
        t : ndarray
            Time vector (ms)
        f0 : float
            Start frequency (kHz)
        f1 : float
            End frequency (kHz)
            
        Returns:
        --------
        ndarray
            Current values with amplitude self.inj_amp and frequency 
            increasing linearly from f0 to f1 Hz over time t
        """
        return self.inj_amp * np.sin(np.pi * (2 * f0 + (f1 - f0) / t[-1] * t) * t)

    def exponential_chirp(self, t, f0, f1):
        """
        Generate a chirp current with exponentially increasing frequency.
        
        Parameters:
        -----------
        t : ndarray
            Time vector (ms)
        f0 : float
            Start frequency (kHz), must be > 0
        f1 : float
            End frequency (kHz), must be > 0
            
        Returns:
        --------
        ndarray
            Current values with amplitude self.inj_amp and frequency 
            increasing exponentially from f0 to f1 Hz over time t
        
        Notes:
        ------
        For exponential chirp, both f0 and f1 must be positive.
        """
        L = np.log(f1 / f0) / t[-1]
        return self.inj_amp * np.sin(np.pi * 2 * f0 / L * (np.exp(L * t) - 1))

    def zap_current(self):
        """
        Create a frequency-modulated (chirp) current for probing impedance.
        
        This method:
        1. Sets up time vectors for the simulation and current injection
        2. Creates a chirp current based on the specified parameters (linear or exponential)
        3. Prepares the current vector for NEURON playback
        
        Notes:
        ------
        The chirp current increases in frequency from fstart to fend Hz over the duration
        of the injection. This allows frequency-dependent impedance to be measured in
        a single simulation.
        """
        self.dt = dt = h.dt
        self.index_v_rest = int(self.inj_delay / dt)
        self.index_v_final = int(self.inj_stop / dt)

        t = np.arange(int(self.tstop / dt) + 1) * dt
        t_inj = t[:self.index_v_final - self.index_v_rest + 1]
        f0 = self.fstart * 1e-3 # Hz to 1/ms
        f1 = self.fend * 1e-3
        chirp_func = self.chirp_func.get(self.chirp_type, self.linear_chirp)
        self.zap_vec_inj = chirp_func(t_inj, f0, f1)
        i_inj = np.zeros_like(t)
        i_inj[self.index_v_rest:self.index_v_final + 1] = self.zap_vec_inj

        self.zap_vec = h.Vector()
        self.zap_vec.from_python(i_inj)
        self.zap_vec.play(self.cell_src._ref_amp, dt)

    def get_impedance(self, smooth=1):
        """
        Calculate and extract the frequency-dependent impedance profile.
        
        This method:
        1. Filters the impedance to the frequency range of interest
        2. Optionally applies smoothing to reduce noise
        3. Identifies the resonant frequency (peak impedance)
        
        Parameters:
        -----------
        smooth : int, optional
            Window size for smoothing the impedance. Default is 1 (no smoothing).
            
        Returns:
        --------
        tuple
            (frequencies, impedance_values) in the range of interest
            
        Notes:
        ------
        Sets self.peak_freq to the resonant frequency (frequency of maximum impedance).
        """
        f_idx = (self.freq > min(self.fstart, self.fend)) & (self.freq < max(self.fstart, self.fend))
        impedance = self.impedance
        if smooth > 1:
            impedance = np.convolve(impedance, np.ones(smooth) / smooth, mode='same')
        freq, impedance = self.freq[f_idx], impedance[f_idx]
        self.peak_freq = freq[np.argmax(impedance)]
        print(f'Resonant Peak Frequency: {self.peak_freq:.3g} (Hz)')
        return freq, impedance

    def execute(self) -> Tuple[list, list]:
        """
        Run the ZAP simulation and calculate the impedance profile.
        
        This method:
        1. Sets up the chirp current
        2. Runs the NEURON simulation
        3. Calculates the impedance using FFT
        4. Prints a summary of the frequency range and analysis method
        
        Returns:
        --------
        tuple
            (time_vector, voltage_vector) from the simulation
            
        Notes:
        ------
        Sets several attributes including:
        - Z: Complex impedance values (from FFT)
        - freq: Frequency values for the impedance profile
        - impedance: Absolute impedance values
        """
        print("ZAP current simulation running...")
        self.zap_current()
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        self.zap_vec.resize(self.t_vec.size())
        self.v_rest = self.v_vec[self.index_v_rest]
        self.v_rest_time = self.t_vec[self.index_v_rest]

        t_idx = slice(self.index_v_rest, self.index_v_final + 1)
        self.v_vec_inj = np.array(self.v_vec)[t_idx] - self.v_rest
        self.t_vec_inj = np.array(self.t_vec)[t_idx] - self.v_rest_time

        self.cell_v_amp_max = np.abs(self.v_vec_inj).max()
        self.Z = np.fft.rfft(self.v_vec_inj) / np.fft.rfft(self.zap_vec_inj) # MOhms
        self.freq = np.fft.rfftfreq(self.zap_vec_inj.size, d=self.dt * 1e-3) # ms to sec
        self.impedance = np.abs(self.Z)

        print()
        print('Chirp current injection with frequency changing from '
              f'{self.fstart:g} to {self.fend:g} Hz over {self.inj_dur * 1e-3:g} seconds')
        print('Impedance is calculated as the ratio of FFT amplitude '
              'of membrane voltage to FFT amplitude of chirp current')
        print()
        return self.t_vec.to_python(), self.v_vec.to_python()


class Profiler():
    """All in one single cell profiler"""
    def __init__(self, template_dir: str = None, mechanism_dir: str = None, dt=None):
        self.template_dir = None
        self.mechanism_dir = None

        if not self.template_dir:
            self.template_dir = template_dir
        if not self.mechanism_dir:
            self.mechanism_dir = mechanism_dir
        self.templates = None

        self.load_templates()

        h.load_file("stdrun.hoc")
        if dt is not None:
            h.dt = dt
            h.steps_per_ms = 1 / h.dt

    def load_templates(self, hoc_template_file=None):
        if self.templates is None: # Can really only do this once
            if self.mechanism_dir != './' and self.mechanism_dir != '.' and self.mechanism_dir != '././':
                neuron.load_mechanisms(self.mechanism_dir)
            h_base = set(dir(h))

            cwd = os.getcwd()
            os.chdir(self.template_dir)
            if not hoc_template_file:
                self.hoc_templates = glob.glob("*.hoc")
                for hoc_template in self.hoc_templates:
                    h.load_file(str(hoc_template))
            else:
                self.hoc_templates = [hoc_template_file]
                h.load_file(hoc_template_file)

            os.chdir(cwd)

            # h.load_file('biophys_components/hoc_templates/Template.hoc')
            h_loaded = dir(h)

            self.templates = [x for x in h_loaded if x not in h_base]

        return self.templates

    def passive_properties(self, template_name: str, post_init_function: str = None,
                           record_sec: str = 'soma', inj_sec: str = 'soma', plot: bool = True,
                           method=None, **kwargs) -> Tuple[list, list]:
        """
        Calculates passive properties for the specified cell template_name

        Parameters
        ==========
        template_name: str or callable
            name of the cell template located in hoc
            or a function that creates and returns a cell object
        post_init_function: str
            function of the cell to be called after the cell has been initialized (like insert_mechs(123))
        record_sec: str
            section of the cell you want to record spikes from (default: soma)
        inj_sec: str
            section of the cell you want to inject current to (default: soma)
        plot: bool
            automatically plot the cell profile
        method: str
            method to estimate membrane time constant (see Passive)
        **kwargs:
            extra key word arguments for Passive()

        Returns time (ms), membrane voltage (mV)
    """
        passive = Passive(template_name, post_init_function=post_init_function,
                          record_sec=record_sec, inj_sec=inj_sec, method=method, **kwargs)
        time, amp = passive.execute()

        if plot:
            plt.figure()
            plt.plot(time, amp)
            if passive.method == 'exp2':
                plt.plot(*passive.double_exponential_fit(), 'r:', label='double exponential fit')
                plt.legend()
            plt.title("Passive Cell Current Injection")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.show()

        return time, amp

    def current_injection(self, template_name: str, post_init_function: str = None,
                          record_sec: str = 'soma', inj_sec: str = 'soma', plot: bool = True,
                          **kwargs) -> Tuple[list, list]:

        ccl = CurrentClamp(template_name, post_init_function=post_init_function,
                           record_sec=record_sec, inj_sec=inj_sec, **kwargs)
        time, amp = ccl.execute()

        if plot:
            plt.figure()
            plt.plot(time, amp)
            plt.title("Current Injection")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.show()

        return time, amp

    def fi_curve(self, template_name: str, post_init_function: str = None,
                 record_sec: str = 'soma', inj_sec: str = 'soma', plot: bool = True,
                 **kwargs) -> Tuple[list, list]:
        """
        Calculates an FI curve for the specified cell template_name

        Parameters
        ==========
        template_name: str or callable
            name of the cell template located in hoc
            or a function that creates and returns a cell object
        post_init_function: str
            function of the cell to be called after the cell has been initialized (like insert_mechs(123))
        record_sec: str
            section of the cell you want to record spikes from (default: soma)
        inj_sec: str
            section of the cell you want to inject current to (default: soma)
        plot: bool
            automatically plot an fi curve

        Returns the injection amplitudes (nA) used, number of spikes per amplitude supplied
            list(amps), list(# of spikes)
        """
        fi = FI(template_name, post_init_function=post_init_function,
                record_sec=record_sec, inj_sec=inj_sec, **kwargs)
        amp, nspk = fi.execute()

        if plot:
            plt.figure()
            plt.plot(amp, nspk)
            plt.title("FI Curve")
            plt.xlabel('Injection (nA)')
            plt.ylabel('# Spikes')
            plt.show()

        return amp, nspk

    def impedance_amplitude_profile(self, template_name: str, post_init_function: str = None,
                                    record_sec: str = 'soma', inj_sec: str = 'soma', plot: bool = True,
                                    chirp_type=None, smooth: int = 9, **kwargs) -> Tuple[list, list]:
        """
        chirp_type: str
            Type of chirp current (see ZAP)
        smooth: int
            Window size for smoothing the impedance in frequency domain
        **kwargs:
            extra key word arguments for ZAP()
        """
        zap = ZAP(template_name, post_init_function=post_init_function,
                  record_sec=record_sec, inj_sec=inj_sec, chirp_type=chirp_type, **kwargs)
        time, amp = zap.execute()

        if plot:
            plt.figure()
            plt.plot(time, amp)
            plt.title("ZAP Response")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')

            plt.figure()
            plt.plot(time, zap.zap_vec)
            plt.title('ZAP Current')
            plt.xlabel('Time (ms)')
            plt.ylabel('Current Injection (nA)')

            plt.figure()
            plt.plot(*zap.get_impedance(smooth=smooth))
            plt.title('Impedance Amplitude Profile')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Impedance (MOhms)')
            plt.show()

        return time, amp

# Example usage
# profiler = Profiler('./temp/templates', './temp/mechanisms/modfiles')
# profiler.passive_properties('Cell_Cf')
# profiler.fi_curve('Cell_Cf')
# profiler.current_injection('Cell_Cf', post_init_function="insert_mechs(123)", inj_amp=300, inj_delay=100)


def run_and_plot(sim, title=None, xlabel='Time (ms)', ylabel='Membrane Potential (mV)',
                 plot=True, plot_injection_only=False):
    """Helper function for running simulation and plot
    sim: instance of the simulation class in this module
    title, xlabel, ylabel: plot labels
    plot: whether or not to plot
    plot_injection_only: plot only the injection duration
    Return: outputs by sim.execute()
    """
    X, Y = sim.execute()
    X = np.array(X)
    Y = np.array(Y)
    if plot:
        plt.figure()
        if plot_injection_only:
            t_idx = (X >= sim.inj_delay) & (X <= sim.inj_delay + sim.inj_dur)
            plt.plot(X[t_idx], Y[t_idx])
        else:
            plt.plot(X, Y)
        if title is None:
            title = type(sim).__name__
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return X, Y
