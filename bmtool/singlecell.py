import glob
import os
from typing import Tuple
import random

import matplotlib.pyplot as plt
import neuron
from neuron import h
import numpy as np

class CurrentClamp():
    def __init__(self, template_name, post_init_function=None, record_sec="soma", record_loc="0.5", tstop=1000,
                 inj_sec="soma", inj_loc="0.5", inj_amp=100, inj_delay=100, inj_dur=1000):
        self.template_name = template_name
        self.tstop = tstop
        self.inj_dur = inj_dur
        
        self.record_sec = record_sec
        self.record_loc = record_loc
        self.inj_sec = inj_sec
        self.inj_loc = inj_loc

        self.inj_amp = inj_amp/1e3
        self.cell = eval('h.'+self.template_name+'()')
        if post_init_function:
            eval(f"self.cell.{post_init_function}")
        self.cell_src = None
        self.cell_vec = None
        self.inj_delay = inj_delay #use x ms after start of inj to calculate r_in, etc

        self.t_vec = h.Vector()

        self.setup()

    def setup(self):
        self.t_vec.record(h._ref_t)

        try:
            inj_str = "self.cell." + self.inj_sec + "(" + self.inj_loc + ")"
            self.cell_src = h.IClamp(eval(inj_str))
        except TypeError:
            try:
                inj_str = "self.cell." + self.inj_sec + "[0](" + self.inj_loc + ")"
                self.cell_src = h.IClamp(eval(inj_str))
            except:
                print("Hint: Are you selecting the correct injection location?")
                raise
        self.cell_src.delay = self.inj_delay
        self.cell_src.dur = self.inj_dur
        self.cell_src.amp = self.inj_amp
        
        try:
            rec_str = "self.cell." + self.record_sec + "(" + self.record_loc + ")._ref_v"
            self.cell_nc = h.NetCon(eval(rec_str),None,sec=eval("self.cell." + self.record_sec))
        except TypeError:
            try:
                rec_str = "self.cell." + self.record_sec + "[0](" + self.record_loc + ")._ref_v"
                self.cell_nc = h.NetCon(eval(rec_str),None,sec=eval("self.cell." + self.record_sec + '[0]'))
            except:
                print("Hint: Are you selecting the correct recording location?")
                raise
        self.cell_nc.threshold = 0
        self.cell_vec = h.Vector()
        self.cell_vec.record(eval(rec_str))

        print(f"Injection location: {inj_str}")
        print(f"Recording: {rec_str}")

    def execute(self) -> Tuple[list,list]:
        print("current clamp simulation running...")
        h.tstop = int(self.tstop)
        h.stdinit()
        h.run()

        return (list(self.t_vec), list(self.cell_vec))

class Passive(CurrentClamp):
    def __init__(self, template_name, tstop=1200, inj_amp=-100,inj_delay=200, inj_dur=1000, **kwargs):
        super(Passive, self).__init__(template_name=template_name, tstop=tstop,
                                      inj_amp=inj_amp, inj_delay=inj_delay, inj_dur=inj_dur, **kwargs)

        self.cell_v_final = 0
        self.v_t_const = 0

        self.v_rest_time = 0.0
        self.v_final_time = 0.0

        self.v_rest = 0.0
        self.r_in = 0.0
        self.tau = 0.0

    def execute(self):
        
        print(f"Running passive property simulation...")
        h.tstop = int(self.tstop)
        h.stdinit()
        h.run()

        index_v_rest = int(((1000/h.dt)/1000 * self.inj_delay))
        index_v_final = int(((1000/h.dt)/1000 * (self.tstop)))
        
        self.v_rest = self.cell_vec[index_v_rest]
        self.v_rest_time = index_v_rest / (1/h.dt)

        self.cell_v_final = self.cell_vec[index_v_final]
        self.v_final_time = index_v_final / (1/h.dt)

        v_diff = self.v_rest - self.cell_v_final

        self.v_t_const = self.v_rest - (v_diff *.632)

        index_v_tau = next(x for x, val in enumerate(list(self.cell_vec)) if val < self.v_t_const) 
        time_tau = (index_v_tau / ((1000/h.dt)/1000)) - self.inj_delay
        self.tau = time_tau #/ 1000 (in ms)
        self.r_in = (v_diff)/(0-self.inj_amp) * 1e6 #MegaOhms -> Ohms
        print()
        print(f"V Rest: {self.v_rest:.2f} (mV)")
        print(f"Resistance: {self.r_in/1e6:.2f} (MOhms)")
        print(f"tau: {self.tau:.2f} (ms)")
        print()

        v_rest_calc = "V_rest Calculation: Voltage taken at time " + str(self.v_rest_time) + "(ms)"
        v_rest_calc1 = f"{self.v_rest:.2f} (mV)"
        rin_calc = "R_in Calculation: [(dV/dI)] = (v_start-v_final)/(i_start-i_final)" 
        rin_calc1 = "(" + str(round(self.v_rest,2)) + "-(" + str(round(self.cell_v_final,2))+"))/(0-(" + str(self.inj_amp) + "))"
        rin_calc2 = str(round(self.v_rest-self.cell_v_final,2))+ " (mV) /" + str(0-self.inj_amp) + " (nA)"
        rin_calc3 = str(round(((self.v_rest-self.cell_v_final)/(0-self.inj_amp)),2)) + " (MOhms)"

        tau_calc = "Tau Calculation: [(s) until 63.2% change in mV]"
        tau_calc1 = "(mV at inj_start_time (" + str(self.inj_delay) + ")) - ((mV at inj_time  - mV at inj_final (" + str(self.tstop) + ")) * 0.632)"
        tau_calc2 = "(" + str(round(self.v_rest,2)) + ") - (" + str(round(self.v_rest,2)) +"-" +str(round(self.cell_v_final,2))+")*0.632 = " + str(round(self.v_rest - ((self.v_rest - self.cell_v_final)*0.632),2))
        tau_calc3 = "Time where mV == " + str(round(self.v_t_const,2)) + " = " + str(self.inj_delay+self.tau) + "(ms) | (" + str(round(self.inj_delay+self.tau*1000,2)) + " - v_start_time (" + str(self.inj_delay) +"))/1000"
        tau_calc4 = str(round(self.tau,4)) + " (ms)"

        print(v_rest_calc)
        print(v_rest_calc1)
        print()
        print(rin_calc)
        print(rin_calc1)
        print(rin_calc2)
        print(rin_calc3)
        print()
        print(tau_calc)
        print(tau_calc1)
        print(tau_calc2) 
        print(tau_calc3) 
        print(tau_calc4)
        print()  

        return (list(self.t_vec), list(self.cell_vec))
        #return (self.v_rest, self.r_in/1e6, self.tau)

class FI():
    def __init__(self,template_name, post_init_function=None,
                 i_increment=100, i_start=0, i_stop=1050, tstart=50, tdur=1000,
                 record_sec="soma", record_loc="0.5", inj_sec="soma", inj_loc="0.5"):
        """ Takes in values of pA """
        super(FI, self).__init__()
        self.template_name = template_name
        self.post_init_function = post_init_function
        self.i_increment = float(i_increment)/1e3
        self.i_start = float(i_start)/1e3
        self.i_stop = float(i_stop)/1e3
        self.tstart = tstart
        self.tdur = tdur

        self.record_sec = record_sec
        self.record_loc = record_loc
        self.inj_sec = inj_sec
        self.inj_loc = inj_loc

        self.tstop = tstart+tdur
        self.cells = []
        self.sources = []
        self.ncs = []
        self.vectors = []
        self.t_vec = h.Vector()
        self.plenvec = []

        self.amps = np.arange(self.i_start,self.i_stop,self.i_increment)
        for _ in self.amps:
            #Cell definition
            cell = eval('h.'+self.template_name+'()')
            if post_init_function:
                eval(f"cell.{post_init_function}")
            self.cells.append(cell)

        self.lenvec = None
        self.ampvec = h.Vector([i*1e3 for i in self.amps])

    def execute(self):
        self.t_vec.record(h._ref_t)

        for i, amp in enumerate(self.amps):
            #Injection
            cell = self.cells[i]

            inj_str = "cell." + self.inj_sec + "(" + self.inj_loc + ")"
            try:
                inj_str = "cell." + self.inj_sec + "(" + self.inj_loc + ")"
                src = h.IClamp(eval(inj_str))
            except TypeError:
                try:
                    #print("Not using section " + inj_str)
                    inj_str = "cell." + self.inj_sec + "[0](" + self.inj_loc + ")"
                    #print("Trying " + inj_str)
                    src = h.IClamp(eval(inj_str))
                except:
                    print("Hint: Are you selecting the correct injection location?")
                    raise
            src.delay = self.tstart
            src.dur = self.tdur
            src.amp = amp
            self.sources.append(src)

            #Recording
            try:
                rec_str = "cell." + self.record_sec + "(" + self.record_loc + ")._ref_v"
                nc = h.NetCon(eval(rec_str),None,sec=eval("cell."+self.record_sec))
            except TypeError:
                try:
                    rec_str = "cell." + self.record_sec + "[0](" + self.record_loc + ")._ref_v"
                    nc = h.NetCon(eval(rec_str),None,sec=eval(f"cell.{self.record_sec}[0]"))
                except:
                    print("Hint: Are you selecting the correct recording location?")
                    raise

            nc.threshold = 0
            spvec = h.Vector()
            nc.record(spvec)
            self.ncs.append(nc)
            self.vectors.append(spvec)
        
        print(f"Injection location: {inj_str}")
        print(f"Recording: {rec_str}")
        print(f"Running FI curve simulation...")
        print()
        h.tstop = int(self.tstop)
        h.stdinit()
        h.run()

        self.plenvec = [len(list(i)) for i in self.vectors]
        print("Results")
        print(f"Rates: {list(self.amps)}")
        print(f"Spikes: {self.plenvec}")
        print()

        return (list(self.amps), list(self.plenvec))


class Profiler():
    """All in one single cell profiler
    """
    def __init__(self, template_dir:str=None, mechanism_dir:str=None):
        self.template_dir = None
        self.mechanism_dir = None
        
        if not self.template_dir:
            self.template_dir = template_dir
        if not self.mechanism_dir:
            self.mechanism_dir = mechanism_dir
        self.templates = None

        self.load_templates()
    
    def load_templates(self,hoc_template_file=None):
        if self.templates is None: # Can really only do this once
            if self.mechanism_dir != './' and self.mechanism_dir != '.' and self.mechanism_dir != '././':
                neuron.load_mechanisms(self.mechanism_dir)
            h_base = dir(h)

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

            #h.load_file('biophys_components/hoc_templates/Template.hoc')
            h_loaded = dir(h)

            self.templates = [x for x in h_loaded if x not in h_base]

        return self.templates
    
    def passive_properties(self, template_name:str, post_init_function:str=None, 
                           record_sec:str='soma', inj_sec:str='soma', plot:bool=True,
                           **kwargs) -> Tuple[list,list]:
        """
        Calculates passive properties for the specified cell template_name

        Parameters
        ==========
        template_name:str
            name of the cell template located in hoc
        post_init_function:str
            function of the cell to be called after the cell has been initialized (like insert_mechs(123))
        record_sec:str
            section of the cell you want to record spikes from (default: soma)
        inj_sec:str
            section of the cell you want to inject current (default: soma)
        plot:bool
            automatically plot the cell profile
        **kwargs:
            extra key word arguments for Passive()
    """
        passive = Passive(template_name, post_init_function=post_init_function,
                          record_sec=record_sec, inj_sec=inj_sec, **kwargs)
        time_vec, amp_vec = passive.execute()

        if plot:
            plt.figure()
            plt.plot(time_vec, amp_vec)
            plt.title("Passive Cell Current Injection")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.show()

        return (time_vec, amp_vec)
    
    def current_injection(self, template_name:str, post_init_function:str=None, 
                          record_sec:str='soma', inj_sec:str='soma', plot:bool=True,
                          **kwargs) -> Tuple[list,list]:
        
        ccl = CurrentClamp(template_name, post_init_function=post_init_function, 
                           record_sec=record_sec, inj_sec=inj_sec, **kwargs)
        time_vec, amp_vec = ccl.execute()

        if plot:
            plt.figure()
            plt.plot(time_vec, amp_vec)
            plt.title("Current Injection")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.show()

        return (time_vec, amp_vec)
    
    
    def fi_curve(self, template_name:str, post_init_function=None, 
                 record_sec:str='soma', inj_sec:str='soma', plot:bool=True,
                 **kwargs) -> Tuple[list,list]:
        """
        Calculates an FI curve for the specified cell template_name

        Parameters
        ==========
        template_name:str
            name of the cell template located in hoc
        post_init_function:str
            function of the cell to be called after the cell has been initialized (like insert_mechs(123))
        record_sec:str
            section of the cell you want to record spikes from (default: soma)
        inj_sec:str
            section of the cell you want to inject current (default: soma)
        plot:bool
            automatically plot an fi curve

        Returns the nA used, number of spikes per nA supplied
            list(amps), list(spikes)

        """
        fi = FI(template_name, post_init_function=post_init_function,
                record_sec=record_sec, inj_sec=inj_sec, **kwargs)
        amp_vec, spike_vec = fi.execute()

        if plot:
            plt.figure()
            plt.plot(amp_vec, spike_vec)
            plt.title("FI Curve")
            plt.xlabel('Injection (nA)')
            plt.ylabel('# Spikes')
            plt.show()

        return (amp_vec, spike_vec)


#Example usage
#profiler = Profiler('./temp/templates', './temp/mechanisms/modfiles')
#profiler.passive_properties('Cell_Cf')
#profiler.fi_curve('Cell_Cf')
#profiler.current_injection('Cell_Cf', post_init_function="insert_mechs(123)", inj_amp=300, inj_delay=100)