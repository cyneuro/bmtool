import glob
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import neuron
from neuron import h


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
    def __init__(self, template_name, post_init_function=None, record_sec='soma', record_loc=0.5, tstop=1000.,
                 inj_sec='soma', inj_loc=0.5, inj_amp=100., inj_delay=100., inj_dur=1000.):
        """
        template_name: str, name of the cell template located in hoc
        post_init_function: str, function of the cell to be called after the cell has been initialized
        record_sec: tuple, (section list name, index) to access a section in a hoc template
            If a string of section name is specified, index default to 0
            If an index is specified, section list name default to `all`
        record_loc: float, location within [0, 1] of a segment in a section to record from
        inj_sec, inj_loc: current injection site, same format as record site
        tstop: time for simulation (ms)
        inj_delay: current injection start time (ms)
        inj_dur: current injection duration (ms)
        inj_amp: current injection amplitude (pA)
        """
        self.template_name = template_name
        self.record_sec = record_sec
        self.record_loc = record_loc
        self.inj_sec = inj_sec
        self.inj_loc = inj_loc

        self.tstop = tstop
        self.inj_delay = inj_delay # use x ms after start of inj to calculate r_in, etc
        self.inj_dur = inj_dur
        self.inj_amp = inj_amp * 1e-3 # pA to nA

        self.cell = getattr(h, self.template_name)()
        if post_init_function:
            eval(f"self.cell.{post_init_function}")
        self.cell_src = None
        self.v_vec = None
        self.t_vec = h.Vector()

        self.setup()

    def setup(self):
        self.t_vec.record(h._ref_t)

        inj_seg, _ = get_target_site(self.cell, self.inj_sec, self.inj_loc, 'injection')
        self.cell_src = h.IClamp(inj_seg)

        self.cell_src.delay = self.inj_delay
        self.cell_src.dur = self.inj_dur
        self.cell_src.amp = self.inj_amp

        rec_seg, _ = get_target_site(self.cell, self.record_sec, self.record_loc, 'recording')
        self.v_vec = h.Vector()
        self.v_vec.record(rec_seg._ref_v)

        print(f'Injection location: {inj_seg}')
        print(f'Recording: {rec_seg}._ref_v')

    def execute(self) -> Tuple[list, list]:
        print("Current clamp simulation running...")
        h.tstop = self.tstop
        h.stdinit()
        h.run()

        return self.t_vec.to_python(), self.v_vec.to_python()


class Passive(CurrentClamp):
    def __init__(self, template_name, tstop=1200., inj_amp=-100., inj_delay=200., inj_dur=1000.,
                 method=None, **kwargs):
        """
        method: {'simple', 'exp2'}, optional.
            Method to estimate membrane time constant. Default is 'simple'
            that find the time to reach 0.632 of change. 'exp2' fits a double
            exponential curve to the membrane potential response.
        """
        assert(inj_amp != 0)
        super(Passive, self).__init__(template_name=template_name, tstop=tstop,
                                      inj_amp=inj_amp, inj_delay=inj_delay, inj_dur=inj_dur, **kwargs)
        self.inj_stop = inj_delay + inj_dur
        self.method = method or 'simple'

    def tau_simple(self):
        v_t_const = self.cell_v_final - self.v_diff / np.e
        index_v_tau = next(x for x, val in enumerate(self.v_vec) if val <= v_t_const)
        self.tau = self.t_vec[index_v_tau] - self.v_rest_time # ms

        def print_calc():
            print()
            print('Tau Calculation: time until 63.2% of dV')
            print('v_rest + 0.632*(v_final-v_rest)')
            print(f'{self.v_rest:.2f} + 0.632*({self.cell_v_final:.2f}-({self.v_rest:.2f})) = {v_t_const:.2f} (mV)')
            print(f'Time where V = {v_t_const:.2f} (mV) is {self.inj_delay + self.tau:.2f} (ms)')
            print(f'{self.inj_delay + self.tau:.2f} - {self.inj_delay:g} = {self.tau:.2f} (ms)')
            print()
        return print_calc

    @staticmethod
    def double_exponential(t, a0, a1, a2, tau1, tau2):
        return a0 + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

    def tau_double_exponential(self):
        t_idx = slice(self.index_v_rest, self.index_v_final + 1)
        v_vec = self.v_vec.as_numpy().copy()[t_idx]
        t_vec = self.t_vec.as_numpy().copy()[t_idx]
        t_vec -= t_vec[0]

        index_v_peak = (np.sign(self.inj_amp) * v_vec).argmax()
        self.t_peak = t_vec[index_v_peak]
        self.v_peak = v_vec[index_v_peak]
        self.v_sag = self.v_peak - self.cell_v_final
        self.v_max_diff = self.v_diff + self.v_sag
        self.sag_norm = self.v_sag / self.v_max_diff

        self.tau_simple()
        p0 = (self.v_sag, -self.v_max_diff, self.t_peak, self.tau) # intial estimate
        v0 = self.v_rest
        def fit_func(t, a1, a2, tau1, tau2):
            return self.double_exponential(t, v0 - a1 - a2, a1, a2, tau1, tau2)
        popt, self.pcov = curve_fit(fit_func, t_vec, v_vec, p0=p0, maxfev=10000)
        self.popt = np.insert(popt, 0, v0 - sum(popt[:2]))
        self.tau = max(self.popt[-2:])

        def print_calc():
            print()
            print('Tau Calculation: Fit a double exponential curve to the membrane potential response')
            print('f(t) = a0 + a1*exp(-t/tau1) + a2*exp(-t/tau2)')
            print('Constained by initial value: f(0) = a0 + a1 + a2 = v_rest')
            print('Fit parameters: (a0, a1, a2, tau1, tau2) = (' + ', '.join(f'{x:.2f}' for x in self.popt) + ')')
            print(f'Membrane time constant is determined from the slowest exponential term: {self.tau:.2f} (ms)')
            print()
            print('Sag potential: v_sag = v_peak - v_final = %.2f (mV)' % self.v_sag)
            print('Normalized sag potential: v_sag / (v_peak - v_rest) = %.3f' % self.sag_norm)
            print()
        return print_calc

    def double_exponential_fit(self):
        t_vec = self.t_vec.as_numpy()[self.index_v_rest:self.index_v_final + 1]
        v_fit = self.double_exponential(t_vec - t_vec[0], *self.popt)
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

        self.v_diff = self.cell_v_final - self.v_rest
        self.r_in = self.v_diff / self.inj_amp # MegaOhms

        print_calc = self.tau_double_exponential() if self.method == 'exp2' else self.tau_simple()

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
        i_start: initial current injection amplitude (pA)
        i_stop: maximum current injection amplitude (pA)
        i_increment: amplitude increment each trial (pA)
        tstart: current injection start time (ms)
        tdur: current injection duration (ms)
        """
        self.template_name = template_name
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
        self.t_vec = h.Vector()
        self.nspks = []

        self.ntrials = int((self.i_stop - self.i_start) // self.i_increment + 1)
        self.amps = (self.i_start + np.arange(self.ntrials) * self.i_increment).tolist()
        for _ in range(self.ntrials):
            # Cell definition
            cell = getattr(h, self.template_name)()
            if post_init_function:
                eval(f"cell.{post_init_function}")
            self.cells.append(cell)

        self.setup()

    def setup(self):
        self.t_vec.record(h._ref_t)

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


class Profiler():
    """All in one single cell profiler"""
    def __init__(self, template_dir: str = None, mechanism_dir: str = None):
        self.template_dir = None
        self.mechanism_dir = None

        if not self.template_dir:
            self.template_dir = template_dir
        if not self.mechanism_dir:
            self.mechanism_dir = mechanism_dir
        self.templates = None

        self.load_templates()

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
        template_name: str
            name of the cell template located in hoc
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
        template_name: str
            name of the cell template located in hoc
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


# Example usage
# profiler = Profiler('./temp/templates', './temp/mechanisms/modfiles')
# profiler.passive_properties('Cell_Cf')
# profiler.fi_curve('Cell_Cf')
# profiler.current_injection('Cell_Cf', post_init_function="insert_mechs(123)", inj_amp=300, inj_delay=100)
