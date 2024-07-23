import glob
import os
import json
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import neuron
from neuron import h


def load_biophys1():
    if not hasattr(h, 'Biophys1'):
        from bmtk import utils
        module_dir = os.path.dirname(os.path.abspath(utils.__file__))
        hoc_file = os.path.join(module_dir, 'scripts', 'bionet', 'templates', 'Biophys1.hoc')
        h.load_file("import3d.hoc")
        h.load_file(hoc_file)


def load_allen_database_cells(morphology, dynamic_params, model_processing='aibs_perisomatic'):
    """Create Allen cell model
    morphology: morphology file path
    dynamic_params: dynamic_params file path
    model_processing: model processing type by AllenCellType database
    Return: a function that creates and returns a cell object 
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
        template_name: str, name of the cell template located in hoc
            or callable, a function that creates and returns a cell object
        post_init_function: str, function of the cell to be called after the cell has been initialized
        record_sec: tuple, (section list name, index) to access a section in a hoc template
            If a string of section name is specified, index default to 0
            If an index is specified, section list name default to `all`
        record_loc: float, location within [0, 1] of a segment in a section to record from
        threshold: Optional float, spike threshold (mV), if specified, record and count spikes times
        inj_sec, inj_loc: current injection site, same format as record site
        tstop: time for simulation (ms)
        inj_delay: current injection start time (ms)
        inj_dur: current injection duration (ms)
        inj_amp: current injection amplitude (pA)
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

        self.cell = self.create_cell()
        if post_init_function:
            eval(f"self.cell.{post_init_function}")

        self.setup()

    def setup(self):
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
        method: {'simple', 'exp2', 'exp'}, optional.
            Method to estimate membrane time constant. Default is 'simple'
            that find the time to reach 0.632 of change. 'exp2' fits a double
            exponential curve to the membrane potential response. 'exp' fits
            a single exponential curve.
        """
        assert(inj_amp != 0)
        super().__init__(template_name=template_name, tstop=tstop,
                         inj_amp=inj_amp, inj_delay=inj_delay, inj_dur=inj_dur, **kwargs)
        self.inj_stop = inj_delay + inj_dur
        self.method = method
        self.tau_methods = {'simple': self.tau_simple, 'exp2': self.tau_double_exponential, 'exp': self.tau_single_exponential}

    def tau_simple(self):
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
        return a0 + a * np.exp(-t / tau)

    def tau_single_exponential(self):
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
        return a0 + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    def tau_double_exponential(self):
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
        t_vec = self.v_rest_time + self.t_vec_inj
        v_fit = self.double_exponential(self.t_vec_inj, *self.popt)
        return t_vec, v_fit
    
    def single_exponential_fit(self):
        t_vec = self.v_rest_time + self.t_vec_inj
        v_fit = self.single_exponential(self.t_vec_inj, *self.popt)
        return t_vec, v_fit

    def execute(self):
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
        template_name: str, name of the cell template located in hoc
            or callable, a function that creates and returns a cell object
        i_start: initial current injection amplitude (pA)
        i_stop: maximum current injection amplitude (pA)
        i_increment: amplitude increment each trial (pA)
        tstart: current injection start time (ms)
        tdur: current injection duration (ms)
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
        print("Running simulations for FI curve...")
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        self.nspks = [len(v) for v in self.tspk_vecs]
        print()
        print("Results")
        print(f'Injection (nA): ' + ', '.join(f'{x:g}' for x in self.amps))
        print(f'Number of spikes: ' + ', '.join(f'{x:d}' for x in self.nspks))
        print()

        return self.amps, self.nspks


class ZAP(CurrentClamp):
    def __init__(self, template_name, inj_amp=100., inj_delay=200., inj_dur=15000.,
                 tstop=15500., fstart=0., fend=15., chirp_type=None, **kwargs):
        """
        fstart, fend: float, frequency at the start and end of the chirp current
        chirp_type: {'linear', 'exponential'}, optional.
            Type of chirp current, i.e. how frequency increase over time.
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
        return self.inj_amp * np.sin(np.pi * (2 * f0 + (f1 - f0) / t[-1] * t) * t)

    def exponential_chirp(self, t, f0, f1):
        L = np.log(f1 / f0) / t[-1]
        return self.inj_amp * np.sin(np.pi * 2 * f0 / L * (np.exp(L * t) - 1))

    def zap_current(self):
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
        f_idx = (self.freq > min(self.fstart, self.fend)) & (self.freq < max(self.fstart, self.fend))
        impedance = self.impedance
        if smooth > 1:
            impedance = np.convolve(impedance, np.ones(smooth) / smooth, mode='same')
        freq, impedance = self.freq[f_idx], impedance[f_idx]
        self.peak_freq = freq[np.argmax(impedance)]
        print(f'Resonant Peak Frequency: {self.peak_freq:.3g} (Hz)')
        return freq, impedance

    def execute(self) -> Tuple[list, list]:
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
