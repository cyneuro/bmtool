from neuron import h
import neuron
import os
import glob
import numpy as np
from datetime import datetime
import re
import click
from clint.textui import puts, colored, indent
from random import random

class Widget:
    def __init__(self):
        return

    def execute(self):
        raise NotImplementedError

    def hoc_declaration_str_list(self,**kwargs):
        return []

    def hoc_display_str_list(self,**kwargs):
        return []

class ValuePanel:

    def __init__(self, init_val=0, label='',lower_limit=-100,upper_limit=100,slider=True):
        self._val = h.ref(init_val)
        h.xvalue(label, self._val, True, self._bounds_check)
        self.__lower_limit = lower_limit
        self.__upper_limit = upper_limit
        if slider:
            h.xslider(self._val, self.__lower_limit, self.__upper_limit)

    def _bounds_check(self):
        self.val = self.val

    @property
    def val(self):
        return self._val[0]

    @val.setter
    def val(self, new_val):
        new_val = max(self.__lower_limit, new_val)
        self._val[0] = min(new_val, self.__upper_limit)

class SameCellValuePanel:

    def __init__(self, original_cell, other_cells, section, prop, init_val=0, label='',lower_limit=-1e15,upper_limit=1e15,all_sec=False):
        self.original_cell = original_cell
        self.other_cells = other_cells
        self.prop = prop
        #import pdb;pdb.set_trace()
        #self._val = h.ref(init_val)
        
        try:
            self._val_obj = getattr(eval("original_cell."+section)(0.5),"_ref_"+prop)
        except AttributeError as e:
            self._val_obj = getattr(eval("original_cell."+section),"_ref_"+prop)
            


        try:
            self._val =  h.Pointer(self._val_obj)
        except Exception as e:
            import pdb;pdb.set_trace()
        self._vals = []

        all_sections = [sec for sec in h.allsec()]
        primary_section = eval("original_cell."+section)

        if all_sec:
            for osection in all_sections:
                if not osection == primary_section:
                    try:
                        attrptr = getattr(osection(0.5),"_ref_"+prop)
                        self._vals.append(h.Pointer(attrptr))
                    except AttributeError as e:
                        print(e)

        else:
            for cell in other_cells:
                self._vals.append(h.Pointer(getattr(eval("cell."+section)(0.5),"_ref_"+prop)))
                #print(self._vals[-1].val)
        #import pdb;pdb.set_trace()
        h.xvalue(label, self._val_obj, True, self._bounds_check)
        self.__lower_limit = lower_limit
        self.__upper_limit = upper_limit
        #h.xslider(self._val, self.__lower_limit, self.__upper_limit)

    def _bounds_check(self):
        #self.val = self.val
        new_val = max(self.__lower_limit, self.val)
        #self._val[0] = min(new_val, self.__upper_limit)
        self._val.assign(min(new_val, self.__upper_limit))
        #print(self._val.val)
        for _val in self._vals:
            _val.assign(min(new_val, self.__upper_limit))
            #print(_val.val)
        #print("bounds check called")
        #pass

    @property
    def val(self):
        return self._val.val

    @val.setter
    def val(self, new_val):
        new_val = max(self.__lower_limit, new_val)
        #self._val[0] = min(new_val, self.__upper_limit)
        self._val.assign(min(new_val, self.__upper_limit))
        #print(self._val.val)
        for _val in self._vals:
            _val.assign(min(new_val, self.__upper_limit))
            #print(_val.val)

# 1WQ2E -- Morgan 1/20/2020 @ ~9:30pm

class SingleButtonWidget(Widget):
    def __init__(self,text,on_press):
        self.text = text
        self.on_press = on_press

    def execute(self):
        h.xpanel("SingleButton")
        h.xbutton(self.text, self.on_press)
        h.xpanel()

class MultiSecMenuWidget(Widget):
    def __init__(self,cell,other_cells,section,md,label="",x="0.5",vartype=1):
        super(MultiSecMenuWidget, self).__init__()
        self.cell = cell
        self.other_cells = other_cells
        self.section = section
        self.md = md
        self.label = label
        self.panels = []
        self.defined_vars = []
        self.x = x
        self.vartype = vartype

        if self.label == "":
            self.label = self.cell.hname() + "." + section + "(0.5)" + " (Parameters)"
        self.variables = self.get_variables()

    def get_variables(self):
        variables = [("diam","diam (um)"),("cm", "cm (uF/cm2)")]
        cellsec = eval('self.cell.'+self.section)
        mechs = [mech.name() for mech in cellsec(0.5) if not mech.name().endswith("_ion")]
        #ctg.mechanism_dict["kdr"]["NEURON"]["USEION"]["READ"]
        #['eleak']
        # if they're in the useion read then ignore as 
        #ctg.mechanism_dict["leak"]["PARAMETER"]
        #[('gbar', '(siemens/cm2)'), ('eleak', '(mV)')]
        md = self.md
        for mech in mechs:
            if md.get(mech) and \
                md[mech].get("NEURON") and \
                md[mech]["NEURON"].get("USEION") and \
                md[mech]["NEURON"]["USEION"].get("READ"):
                    ri = md[mech]["NEURON"]["USEION"]["READ"]
                    for v in ri:
                        variables.append((v,v))
            if md.get(mech) and md[mech].get("PARAMETER"):
                params = md[mech]["PARAMETER"]
                ions = [v[0] for v in variables]
                for param in params:
                    if param[0] not in ions:
                        v = param[0]+"_"+mech
                        units = ""
                        if param[1]:
                            units = param[1]
                        t = v + ' ' + units
                        variables.append((v,t))
        return variables
    
    def execute(self):
        h.xpanel('xvarlabel')
        h.xlabel(self.label)
        for var in self.variables:
            if var[1] not in self.defined_vars:
                try:
                    panel=SameCellValuePanel(self.cell, self.other_cells, self.section, var[0], label=var[1])
                    self.panels.append(panel)
                    self.defined_vars.append(var[1])
                except AttributeError as e:
                    print(e)
        h.xpanel()
        return

    def add_var(self):
        pass

    def hoc_display_str_list(self,**kwargs):
        ctg = kwargs["ctg"]
        ret = []
        #ret.append("$o2.soma nrnsecmenu(.5,1)")
        cell = ctg.hoc_ref(self.cell)
        ret.append(cell + "."+ self.section + " nrnsecmenu("+str(self.x)+","+str(self.vartype)+")")
        return ret

class SegregationSelectorWidget(Widget):
    def __init__(self,cell, other_cells, section, mechanism_dict, all_sec=False, variables=None):
        super(SegregationSelectorWidget, self).__init__()
        if all_sec:
            self.label = "Segregation Selection (All Segments)"
        else:
            self.label = "Segregation Selection"
        self.cell = cell
        self.other_cells = other_cells
        self.section = section
        self.mechanism_dict = mechanism_dict
        self.all_sec = all_sec

        self.vps=[]#ValuePanels
        self.labels=[]
        cellsec = eval("cell."+section)
        self.mechs = [mech.name() for mech in cellsec(0.5) if not mech.name().endswith("_ion")]
        

        self.variables = variables

    def execute(self):
        def calculate():
            h.stdinit()
            h.run()

        def a(n):
            """function to be called when a radio button is toggled"""
            print(n)
        h.xpanel('xvarlabel')
        h.xlabel(self.label)
        if self.variables:
            for variable in self.variables:
                label = variable #+  ' Segregation (mV)'
                ref = variable
                try:
                    if not label in self.labels:
                        vp = SameCellValuePanel(self.cell, self.other_cells, self.section, ref, label=label,all_sec=self.all_sec)
                        self.vps.append(vp)
                        self.labels.append(label)
                except AttributeError as e:
                    pass
        else:
            for mech in self.mechs:
                if not self.mechanism_dict[mech].get("DERIVATIVE"):
                    continue
                actvars = [line['variable'] for line in self.mechanism_dict[mech]["DERIVATIVE"] if line['is_likely_activation']]
                
                for var in self.mechanism_dict[mech]['state_activation_vars']:
                    if var['var'] in actvars:
                        #{'var': 'n', 'var_inf': 'inf', 'vh': 'nvhalf', 'k': 'nk', 'procedure_set': 'rate', 'line_set': 'inf = 1.0 / (1.0 + (exp((v + nvhalf) / (nk))))'}
                        label = var['var'] + ' (' + mech + ') ' +  'Segregation (mV)'
                        #vp = ValuePanel(label=label)
                        ref = var['var']+'seg_'+mech
                        try:
                            if not label in self.labels:
                                vp = SameCellValuePanel(self.cell, self.other_cells, self.section, ref, label=label,all_sec=self.all_sec)
                                self.vps.append(vp)
                                self.labels.append(label)
                        except AttributeError as e:
                            pass
        h.xbutton('Segregation (Run)', calculate)
        
        #menu = h.xmenu('menu')
        #import pdb;pdb.set_trace()
        #for i in range(1, 11):
        #    h.xradiobutton('item %d' % i, (a, i))

        #h.xmenu()
        h.xpanel()
        
class SegregationPassiveWidget(Widget):
    def __init__(self, fir_widget, cell, other_cells, section, mechanism_dict, gleak_var = None, eleak_var = None):
        super(SegregationPassiveWidget, self).__init__()
        self.label = "Segregation Passive Fitting"
        self.fir_widget = fir_widget
        self.is_calculating = False
        self.v_rest = None
        self.r_in = None
        self.tau = None

        self.cell = cell
        self.other_cells = other_cells
        self.section = section
        self.mechanism_dict = mechanism_dict
        
        self.segment = eval("cell."+section+"(0.5)") #getattr(cell,section)(0.5)
        self.mechs = [mech.name() for mech in self.segment if not mech.name().endswith("_ion")]

        
        self.eleak_var = ""
        self.gleak_var = ""
        self.eleak_not_found = False
        self.gleak_not_found = False
        if not gleak_var:
            if hasattr(self.segment,"gbar_leak"):
                self.gleak_var = "gbar_leak"
            elif hasattr(self.segment,"g_bar_leak"):
                self.gleak_var = "g_bar_leak"
            else:
                #raise AttributeError("Leak channel gbar not found, specify gleak_var in SegregationPassiveWidget")
                self.gleak_not_found = True
        else:
            self.gleak_var = gleak_var
        
        if not eleak_var:
            if hasattr(self.segment,"eleak"):
                self.eleak_var = "eleak"
            elif hasattr(self.segment,"e_leak"):
                self.eleak_var = "e_leak"
            else:
                #raise AttributeError("Leak channel reversal not found, specify eleak_var in SegregationPassiveWidget")
                self.eleak_not_found = True
        else:
            self.eleak_var = eleak_var
        
        #import pdb;pdb.set_trace()

    def execute(self):
        def calculate():
            if not self.is_calculating:
                self.is_calculating = True
                            
                area = self.segment.area()
                g_bar_leak = getattr(self.segment,self.gleak_var)

                v_rest = self.v_rest.val # mV
                r_in = self.r_in.val # Ohms
                tau = self.tau.val # ms

                # Assuming Vrest is within the range for ELeak
                # ELeak = Vrest

                # Rin = 1/(Area*g_bar_leak)
                # g_bar leak = 1/(Rin*Area)

                # tau = (R*C) = R*Area*cm
                # cm = tau*g_bar_leak*Area
                # cm/cm2 = (tau*g_bar_leak*Area)/Area = tau*g_bar_leak

                eleak = v_rest
                setattr(self.segment,self.eleak_var,eleak)

                g_bar_leak = 1/(r_in*area)*1e2 #area m to cm?
                setattr(self.segment,self.gleak_var,g_bar_leak)

                cm = tau*g_bar_leak*1e3 #tau ms->s
                setattr(self.segment,"cm",cm)

                for cell in self.other_cells:
                    segment = eval("cell."+self.section+"(0.5)") #getattr(cell,self.section)(0.5)
                    setattr(segment,self.eleak_var,eleak)
                    setattr(segment,self.gleak_var,g_bar_leak)
                    setattr(segment,"cm",cm)

                h.stdinit()
                h.run()
            
        spw = self
        ctstop = self.fir_widget.tstop
        cvode = h.CVode()

        def commands():
            def start_event():
                return
            cvode.event(0 , start_event)

            def stop_event():
                nonlocal spw
                if not spw.is_calculating:
                    if self.gleak_var and self.eleak_var:
                        spw.v_rest.val = round(self.fir_widget.v_rest,4)
                        spw.r_in.val = round(float(self.fir_widget.r_in/1e6),4)
                        spw.tau.val = round(self.fir_widget.tau,4)  
                else:
                    spw.is_calculating = False           
            cvode.event(ctstop, stop_event)           

        h.xpanel('xvarlabel')
        h.xlabel(self.label)
        if self.gleak_var and self.eleak_var:
            self.v_rest=ValuePanel(label='V Rest (mV)',slider=False,lower_limit=-1000,upper_limit=1000)
            self.r_in=ValuePanel(label='R in (MOhm)',slider=False,lower_limit=0,upper_limit=100000)
            self.tau=ValuePanel(label='Tau (ms)',slider=False,lower_limit=-100,upper_limit=100)
            h.xbutton('Fit Passive Properties (Run)', calculate)
        else:
            if self.gleak_not_found:
                h.xlabel("Leak conductance (gleak) not found")
            if self.eleak_not_found:
                h.xlabel("Leak reversal (eleak) not found")
        h.xpanel()

        return commands

class SegregationFIRFitWidget(Widget):
    def __init__(self, fir_widget):
        super(SegregationFIRFitWidget, self).__init__()
        self.label = "Segregation FIR Fitting"
        self.fir_widget = fir_widget
        self.vps = []
        self.is_calculating = False
        self.valvec = None
        self.plot = None

    def execute(self):
        def calculate():
            self.is_calculating = True
            h.stdinit()
            h.run()

        spw = self
        ctstop = self.fir_widget.tstop
        cvode = h.CVode()

        def commands():
            def start_event():
                vals = [v.val for v in self.vps]
                any_nonzero = next((value for index,value in enumerate(vals) if value != 0), None)
                if any_nonzero:
                    self.valvec = h.Vector(vals)
                    self.fir_widget.graph.color(3)
                    self.fir_widget.graph.label("User FI Curve")
                    self.plot = self.valvec.plot(self.fir_widget.graph,self.fir_widget.ampvec,3,1)
                return
            cvode.event(1 , start_event)

            def stop_event():
                nonlocal spw
                if not spw.is_calculating:
                    pass 
                else:
                    spw.is_calculating = False           
            cvode.event(ctstop, stop_event)           

        h.xpanel('xvarlabel')
        h.xlabel(self.label)
        for amp in self.fir_widget.amps:
            vp = ValuePanel(label=str(int(amp*1000)) + " (pA)",slider=False,lower_limit=0,upper_limit=1000)
            self.vps.append(vp)
        h.xbutton('Fit FIR Curve (Run) [Plots User FIR]', calculate)
        h.xpanel()

        return commands

class AutoVInitWidget(Widget):
    def __init__(self,fir_widget):
        self.fir_widget = fir_widget
        self.vinitcheckbox = 1
    
    def execute(self):
        spw = self
        ctstop = self.fir_widget.tstop
        cvode = h.CVode()

        def commands():
            def start_event():
                return
            cvode.event(0 , start_event)

            def stop_event():
                nonlocal spw
                if spw.vinitcheckbox:
                    h.v_init = spw.fir_widget.v_rest         
            cvode.event(ctstop, stop_event)           

        h.xpanel('xvarlabel')
        h.xcheckbox('Auto-set v_init to v_rest at end of simulation', (self, 'vinitcheckbox'))#, self.vinitcheckboxpressed)
        h.xpanel()

        return commands

class TextWidget(Widget):
    def __init__(self,label=""):
        super(TextWidget, self).__init__()
        self.label = label
        self.mystrs = []

        self.fir_widget = None
        self.fir_print_calc = False
        self.fir_print_fi = False
        return
    
    def add_text(self, text):
        """
        Returns the index for the string you want to set
        Easier way to do newlines
        """
        self.mystrs.append(h.ref(''))
        i = len(self.mystrs)-1
        self.mystrs[i][0] = text
        return i

    def set_text(self, index, text):
        self.mystrs[index][0] = text
        return

    def execute(self):
        h.xpanel('xvarlabel')
        h.xlabel(self.label)
        for mystr in self.mystrs:
            h.xvarlabel(mystr)
        h.xpanel()
        return
    
    def set_to_fir_passive(self, fir_widget, print_calc=True, print_fi=True):
        self.fir_widget = fir_widget
        self.fir_print_calc = print_calc
        self.fir_print_fi = print_fi
        self.label='PASSIVE PROPERTIES:\n'
        self.add_text("V_rest: ")
        self.add_text("R_in: ")
        self.add_text("Tau: ")
        self.add_text("")
        if print_calc:
            self.add_text("V_rest Calculation: ")
            self.add_text("R_in Calculation: ")
            self.add_text("")
            self.add_text("Tau Calculation: ")
            self.add_text("")
            self.add_text("")
        if print_fi:
            self.add_text("FICurve ([nA]:Hz): ")
            self.add_text("")
        return
    
    def update_fir_passive(self):
        
        if self.fir_widget:
            self.set_text(0,"V_rest: " + str(round(self.fir_widget.v_rest,2)) + " (mV) ")
            self.set_text(1,"R_in: " + str(round(self.fir_widget.r_in/1e6,2)) + " (MOhms) ")
            self.set_text(2,"Tau: " + str(round(self.fir_widget.tau,4)) + " (ms) ")

            if self.fir_print_calc:
                v_rest_calc = "V_rest Calculation: Taken at time " + str(self.fir_widget.v_rest_time) + "(ms) on negative injection cell"
                rin_calc = "R_in Calculation: [(dV/dI)] = (v_start-v_final)/(i_start-i_final) = " + str(round(self.fir_widget.v_rest,2)) + \
                        "-(" + str(round(self.fir_widget.passive_v_final,2))+"))/(0-(" + str(self.fir_widget.passive_amp) + "))" + \
                        " = (" + str(round(self.fir_widget.v_rest-self.fir_widget.passive_v_final,2))+ " (mV) /" + str(0-self.fir_widget.passive_amp) + " (nA))" + \
                        " = ("+ str(round(((self.fir_widget.v_rest-self.fir_widget.passive_v_final)/(0-self.fir_widget.passive_amp)),2)) + " (MOhms))"
                tau_calc = "Tau Calculation: [(s) until 63.2% change in mV] = " + \
                        "(mV at inj_start_time (" + str(self.fir_widget.tstart) + ")) - ((mV at inj_time  - mV at inj_final (" + str(self.fir_widget.tstart+self.fir_widget.passive_delay) + ")) * 0.632) = " + \
                        "(" + str(round(self.fir_widget.v_rest,2)) + ") - (" + str(round(self.fir_widget.v_rest,2)) +"-" +str(round(self.fir_widget.passive_v_final,2))+")*0.632 = " + str(round(self.fir_widget.v_rest - ((self.fir_widget.v_rest - self.fir_widget.passive_v_final)*0.632),2))
                tau_calc2 = "Time where mV == " + str(round(self.fir_widget.v_t_const,2)) + " = " + str(self.fir_widget.tstart+self.fir_widget.tau) + "(ms) | (" + str(self.fir_widget.tstart+self.fir_widget.tau*1000) + " - v_start_time (" + str(self.fir_widget.tstart) +"))/1000 = " + str(round(self.fir_widget.tau,4))
                
                self.set_text(4,v_rest_calc)
                self.set_text(5,rin_calc)
                self.set_text(6,"v_start time: " + str(self.fir_widget.v_rest_time) + "(ms) | v_final time: " + str(self.fir_widget.v_final_time) + "(ms)")
                self.set_text(7,tau_calc)
                self.set_text(8,tau_calc2)
            if self.fir_print_fi:
                spikes = [str(round(i,0)) for i in self.fir_widget.plenvec]
                amps = self.fir_widget.amps
                self.set_text(11," | ".join("["+str(round(a,2))+"]:"+n for a,n in zip(amps,spikes)))
        return
        
class PointMenuWidget(Widget):
    def __init__(self,pointprocess):
        super(PointMenuWidget, self).__init__()
        self.pointprocess = pointprocess

        self.is_iclamp = False
        self.iclamp_obj = None
        self.iclamp_sec = None
        self.iclamp_dur = 0
        self.iclamp_amp = 0
        self.iclamp_delay = 0

        self.is_netstim = False
        self.netstim_obj = None
        self.netstim_sec = None
        self.netstim_interval = 0
        self.netstim_number = 0
        self.netstim_start = 0
        self.netstim_noise = 0     
        self.netstim_synapse = None
        self.netstim_netcon = None
        self.netstim_netcon_weight = 1

        self.is_synapse = False
        self.synapse_obj = None
        self.synapse_sec = None
        self.synapse_name = ""
        self.synapse_location = 0.5

        return
    
    def iclamp(self, sec, dur, amp, delay):
        self.is_iclamp = True
        self.iclamp_sec = sec

        iclamp = h.IClamp(sec)
        iclamp.dur = dur
        iclamp.amp = amp
        iclamp.delay = delay

        self.iclamp_dur = dur
        self.iclamp_amp = amp
        self.iclamp_delay = delay

        self.iclamp_obj = iclamp
        self.pointprocess = iclamp

        return self.iclamp_obj

    def netstim(self,interval,number,start,noise,target=None,weight=1,location=0.5):
        self.is_netstim = True

        self.netstim_obj = h.NetStim(location)
        self.netstim_obj.interval = interval #ms (mean) time between spikes
        self.netstim_obj.number = number # (average) number of spikes
        self.netstim_obj.start = start # ms (most likely) start time of first spike
        self.netstim_obj.noise = noise # range 0 to 1. Fractional randomness.
        
        self.netstim_location = location
        self.netstim_interval = interval
        self.netstim_number = number
        self.netstim_start = start
        self.netstim_noise = noise     

        self.netstim_netcon_weight = weight

        if target:
            self.netstim_synapse = target
            self.netstim_netcon = h.NetCon(self.netstim_obj, target,0,0,self.netstim_netcon_weight)

        self.pointprocess = self.netstim_obj

        return self.netstim_obj, self.netstim_netcon

    def synapse(self, sec, location, synapse_name):
        self.is_synapse = True
        #syn = h.AlphaSynapse(soma(0.5))
        self.synapse_sec = sec(float(location))
        self.synapse_name = synapse_name
        self.synapse_obj = getattr(h,synapse_name)(float(location),sec=sec)
        self.pointprocess = self.synapse_obj
        self.synapse_location = location
        
        return self.synapse_obj

    def execute(self):
        h.nrnpointmenu(self.pointprocess)
        return

    def hoc_declaration_str_list(self,**kwargs):
        ctg = kwargs["ctg"]
        ret = []

        if self.is_iclamp:
            cell_ref = ctg.hoc_ref(self.iclamp_sec.sec.cell())
            sec_ref = self.iclamp_sec.sec.hname().split(".")[-1]
            clamp_ref = ctg.hoc_ref(self.iclamp_obj)
            clamp_loc = self.iclamp_sec.x

            ret.append("// current clamp current injection")
            #ret.append("objref " + clamp_ref)
            ret.append(cell_ref + "." + sec_ref + " " + clamp_ref + " = new IClamp(" + str(clamp_loc) +")")
            ret.append(clamp_ref + ".del = " + str(self.iclamp_delay))
            ret.append(clamp_ref + ".dur = " + str(self.iclamp_dur))
            ret.append(clamp_ref + ".amp = " + str(self.iclamp_amp))

        elif self.is_netstim:
            netstim_ref = ctg.hoc_ref(self.netstim_obj)
            netstim_loc = self.netstim_location
            netcon_ref = ctg.hoc_ref(self.netstim_netcon)
            netstim_syn_ref = ctg.hoc_ref(self.netstim_synapse)

            #ret.append("objref " + netstim_ref + "// the code below provides the cell with a spike train")
            #ret.append("objref " + netcon_ref + "// the code below provides the cell with a spike train")

            ret.append(netstim_ref + "=new NetStim(" + str(netstim_loc) + ")")
            ret.append(netstim_ref + ".interval=" + str(self.netstim_interval) + " // ms (mean) time between spikes")
            ret.append(netstim_ref + ".number=" + str(self.netstim_number) + " //(average) number of spikes")
            ret.append(netstim_ref + ".start=" + str(self.netstim_start) + " // ms (most likely) start time of first spike")
            ret.append(netstim_ref + ".noise=" + str(self.netstim_noise) + " // range 0 to 1. Fractional randomness.")
            ret.append(netcon_ref + "=new NetCon("+netstim_ref+","+netstim_syn_ref+",0,0,"+str(self.netstim_netcon_weight)+")")

        elif self.is_synapse:
            cell_ref = ctg.hoc_ref(self.synapse_sec.sec.cell())
            sec_ref = self.synapse_sec.sec.hname().split(".")[-1]
            syn_ref = ctg.hoc_ref(self.synapse_obj)
            syn_loc = self.synapse_location
            
            #ret.append("objref " + syn_ref)
            ret.append(cell_ref + "." + sec_ref + " " + syn_ref + " = new " + \
                        self.synapse_name + "(" + str(syn_loc) +") // build a synapse input into "+cell_ref)
            
        return ret

    def hoc_display_str_list(self,**kwargs):
        ctg = kwargs["ctg"]
        ret = []

        if self.is_iclamp:
            hoc_ref = ctg.hoc_ref(self.iclamp_obj)
            ret.append("nrnpointmenu(" + hoc_ref + ")")
        elif self.is_netstim:
            hoc_ref = ctg.hoc_ref(self.netstim_obj)
            ret.append("nrnpointmenu(" + hoc_ref + ")")
            pass
        elif self.is_synapse:
            hoc_ref = ctg.hoc_ref(self.synapse_obj)
            ret.append("nrnpointmenu(" + hoc_ref + ")")
            pass
        return ret

class VoltagePlotWidget(Widget):
    
    def __init__(self,cell,section="soma",tstop=1150,minx=-80,maxx=50,miny=0,maxy=1):
        super(VoltagePlotWidget, self).__init__()
        self.cell = cell
        self.section = section
        self.tstop = tstop
        self.graph = None
        self.color = 1
        self.expressions = {}
        self.segment = eval("self.cell."+section) #getattr(self.cell,section)
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

        self.vectors = []
        self.v_vector = None
        return

    def advance_color(self):
        #https://www.neuron.yale.edu/neuron/static/py_doc/visualization/graph.html#Graph.color
        self.color = self.color + 1
        if self.color == 10:
            self.color = 1
        

    def reset_color(self):
        self.color = 1

    def add_var(self,variable,text):
        self.expressions[text] = variable
        return

    def add_act_inf(self, mech_dict=None, variables=None):
        if mech_dict:
            mechs = [mech.name() for mech in self.segment(0.5) if not mech.name().endswith("_ion")]
            for mech in mechs:
                if not mech_dict[mech].get("DERIVATIVE"):
                    continue
                actvars = [line['variable'] for line in mech_dict[mech]["DERIVATIVE"] if line['is_likely_activation']]
                
                for var in mech_dict[mech]['state_activation_vars']:
                    if var['var'] in actvars:
                        #{'var': 'n', 'var_inf': 'inf', 'vh': 'nvhalf', 'k': 'nk', 'procedure_set': 'rate', 'line_set': 'inf = 1.0 / (1.0 + (exp((v + nvhalf) / (nk))))'}
                        #label = var['var'] + ' (' + mech + ') ' +  'Segregation (mV)'
                        #vp = ValuePanel(label=label)
                        #ref = var['var']+'seg_'+mech

                        variable = "_ref_" + var['var_inf'] + "_" + mech
                        text = '('+var['var']+') '+var['var_inf'] + "_" + mech
                        
                        if var['k'] in [m[0] for m in mech_dict[mech]["PARAMETER"] if m[2] and float(m[2]) < 0] or not var['k']:            
                            self.add_var(variable,text)
        if variables:
            for variable in variables:
                variable_ref = "_ref_"+variable
                self.add_var(variable_ref,variable)

        #self.add_var("_ref_inf_kdrseg","n inf kdr")
        #self.add_var("_ref_minf_naseg","m inf na")
        return
    
    def execute(self):
        variable = getattr(self.segment(0.5),"_ref_v")
        self.v_vector = h.Vector()
        self.v_vector.record(variable)

        #Recording
        for text, var in self.expressions.items():
            try:
                variable = getattr(self.segment(0.5),var)
            except AttributeError as e:
                variable = getattr(h,var) # Perhaps it's a global?
            spvec = h.Vector()
            spvec.record(variable)
            self.vectors.append((spvec,text))

        cvode = h.CVode()
            
        self.graph = h.Graph()
        
        #self.graph.size(self.tstart,self.tstop,self.miny,self.maxy)
        h.graphList[0].append(self.graph)

        def commands():
            def start_event():
                self.graph.erase_all()
                return
            cvode.event(0 , start_event)

            def stop_event():
                for color, v in enumerate(self.vectors):
                    vec = v[0]
                    text = v[1]
                    #self.graph.addvar(text,variable)
                    self.graph.color(color+1)
                    self.graph.label(text)
                    vec.plot(self.graph,self.v_vector, color+1,1)
                    
                    #self.advance_color()
                    
                return
                
            cvode.event(self.tstop, stop_event)       
        self.graph.size(self.minx,self.maxx,self.miny,self.maxy)
        return commands
    

class PlotWidget(Widget):

    def __init__(self, tstart=0,tstop=50,miny=-80,maxy=50):
        super(PlotWidget, self).__init__()
        self.tstart = tstart
        self.tstop = tstop
        self.miny = miny
        self.maxy = maxy
        self.graph = None
        self.current_index = 0
        self.color = 1
        self.expressions = {}
        self.hoc_expressions = {}
        return

    def advance_color(self):
        #https://www.neuron.yale.edu/neuron/static/py_doc/visualization/graph.html#Graph.color
        self.color = self.color + 1
        if self.color == 10:
            self.color = 1
        

    def reset_color(self):
        self.color = 1

    def add_expr(self,variable,text,hoc_text="",hoc_text_obj=None):
        self.expressions[text] = variable
        if hoc_text != "":
            self.hoc_expressions[text] = (hoc_text,hoc_text_obj)
        return
    
    def execute(self):
        self.graph = h.Graph()
        for text, variable in self.expressions.items():
            #self.graph.addvar('soma(0.5).v', my_cell.soma(0.5)._ref_v)
            self.graph.addvar(text,variable)
            self.advance_color()
            self.graph.color(self.color)
        self.graph.size(self.tstart,self.tstop,self.miny,self.maxy)
        h.graphList[0].append(self.graph)
        return

    def hoc_display_str_list(self,**kwargs):
        ret = []
        ctg = kwargs["ctg"]
        #newPlot(0,tstop,-80,60)        
        #graphItem.save_name("graphList[0].")
        #graphList[0].append(graphItem)
        #sprint(tstr1,"%s.soma.v(.5)",$s1)
        #graphItem.addexpr(tstr1,3,1)
        ret.append("newPlot("+str(self.tstart)+","+str(self.tstop) + \
                            ","+str(self.miny)+","+str(self.maxy)+")")
        ret.append("graphItem.save_name(\"graphList[0].\")")
        ret.append("graphList[0].append(graphItem)")
        for text, hoc_expression in self.hoc_expressions.items():
            #ret.append("sprint(tstr1,\"%s.soma.v(.5)\",$s1)
            
            hoc_ref_str = ctg.hoc_ref(hoc_expression[1]) # Get the hoc equivilent of this object
            ret.append("sprint(tstr1,\""+hoc_expression[0]+"\",\""+hoc_ref_str+"\")")
            ret.append("graphItem.addexpr(tstr1,"+str(self.color)+",1)")
            self.advance_color()
        ret.append("")

        return ret

class FICurveWidget(Widget):
    def __init__(self,template_name,i_increment=100,i_start=0,i_stop=1000,tstart=50,
            tdur=1000,passive_amp=-100,passive_delay=200, record_sec="soma", record_loc="0.5",
            inj_sec="soma", inj_loc="0.5"):
        """ Takes in values of pA
        """
        super(FICurveWidget, self).__init__()
        self.template_name = template_name
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
        self.graph = None
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
            self.cells.append(cell)

        self.lenvec = None
        self.ampvec = h.Vector([i*1e3 for i in self.amps])

        self.passive_amp = passive_amp/1e3
        self.passive_cell = eval('h.'+self.template_name+'()')
        self.passive_src = None
        self.passive_vec = None
        self.passive_delay = passive_delay #use x ms after start of inj to calculate r_in, etc
        self.passive_v_final = 0
        self.v_t_const = 0

        self.v_rest_time = 0.0
        self.v_final_time = 0.0

        self.v_rest = 0.0
        self.r_in = 0.0
        self.tau = 0.0

    def execute(self):
        self.graph = h.Graph()
        self.t_vec.record(h._ref_t)

        self.passive_src = h.IClamp(eval("self.passive_cell." + self.inj_sec + "(" + self.inj_loc + ")"))
        self.passive_src.delay = self.tstart
        self.passive_src.dur = self.tdur
        self.passive_src.amp = self.passive_amp
        
        rec_str = "self.passive_cell." + self.record_sec + "(" + self.record_loc + ")._ref_v"
        self.passive_nc = h.NetCon(eval(rec_str),None,sec=eval("self.passive_cell." + self.record_sec)) 
        self.passive_nc.threshold = 0
        self.passive_vec = h.Vector()
        self.passive_vec.record(eval(rec_str))

        for i, amp in enumerate(self.amps):
            #Injection
            cell = self.cells[i]

            inj_str = "cell." + self.inj_sec + "(" + self.inj_loc + ")"
            src = h.IClamp(eval(inj_str))
            src.delay = self.tstart
            src.dur = self.tdur
            src.amp = amp
            self.sources.append(src)

            #Recording
            rec_str = "cell." + self.record_sec + "(" + self.record_loc + ")._ref_v"
            nc = h.NetCon(eval(rec_str),None,sec=eval("cell."+self.record_sec))
            nc.threshold = 0
            spvec = h.Vector()
            nc.record(spvec)
            self.ncs.append(nc)
            self.vectors.append(spvec)

        ctstop = self.tstart+self.tdur
        cvgraph = self.graph
        cvectors = self.vectors
        ctemplate_name = self.template_name
        lenvec = self.lenvec
        ampvec = self.ampvec
        camps = self.amps
        plenvec = self.plenvec
        cdur = self.tdur
        cfir_widget = self
        cvode = h.CVode()
        def commands():
            def start_event():
                nonlocal cfir_widget, cvgraph
                cvgraph.erase_all()
                cfir_widget.plenvec.clear()
                #print(cfir_widget.vectors[4].as_numpy())
                return
            cvode.event(0 , start_event)

            def stop_event():
                nonlocal ctstop, cvectors, cvgraph, ctemplate_name, ampvec, lenvec,camps,plenvec
                nonlocal cfir_widget
                #print(cfir_widget.vectors[4].as_numpy())    
                #print(cfir_widget.cells[0].soma(0.5)._ref_cm)
                tplenvec = [len(cvec) for cvec in cvectors]
                hzlenvec = [i * (1000/cdur) for i in tplenvec]
                for vec in hzlenvec:
                    plenvec.append(vec)
                lenvec = h.Vector(plenvec)
                #print(lenvec.as_numpy())
                #cvgraph.erase_all()
                cvgraph.color(1)
                cvgraph.label(ctemplate_name + " FI Curve")
                plot = lenvec.plot(cvgraph,ampvec,1,1)
                cvgraph.size(0,max([i*1e3 for i in camps]),0,max(lenvec)+1)
                
                #cfir_widget.passive_vec[int(cfir_widget.tstop)-20]
                index_v_rest = int(((1000/h.dt)/1000 * cfir_widget.tstart))
                index_v_final = int(((1000/h.dt)/1000 * (cfir_widget.tstart+cfir_widget.passive_delay)))
                
                cfir_widget.v_rest = cfir_widget.passive_vec[index_v_rest]
                cfir_widget.v_rest_time = index_v_rest / (1/h.dt)

                cfir_widget.passive_v_final = cfir_widget.passive_vec[index_v_final]
                cfir_widget.v_final_time = index_v_final / (1/h.dt)

                v_diff = cfir_widget.v_rest - cfir_widget.passive_v_final

                cfir_widget.v_t_const = cfir_widget.v_rest - (v_diff *.632)
                #Find index of first occurance where
                #index_v_tau = list(filter(lambda i: i < v_t_const, cfir_widget.passive_vec))[0]
                index_v_tau = next(x for x, val in enumerate(list(cfir_widget.passive_vec)) if val < cfir_widget.v_t_const) 
                time_tau = (index_v_tau / ((1000/h.dt)/1000)) - cfir_widget.tstart
                cfir_widget.tau = time_tau #/ 1000 (in ms)
                cfir_widget.r_in = (v_diff)/(0-cfir_widget.passive_amp) * 1e6 #MegaOhms -> Ohms
                                
            cvode.event(ctstop, stop_event)
        
        h.graphList[0].append(self.graph)

        return commands

    def hoc_display_str_list(self,**kwargs):
        return []

class SecMenuWidget(Widget):
    def __init__(self, sec, x=0.5, vartype=1):
        """
        vartype=1,2,3 shows parameters, assigned, or states respectively.
        0 < x < 1 shows variables at segment containing x changing these variables changes only the values in that segment eg. equivalent to section.v(.2) = -65
        """
        super(SecMenuWidget, self).__init__()
        self.x = x
        self.vartype = vartype
        self.sec = sec
        return

    def execute(self):
        h.nrnsecmenu(self.x,self.vartype,sec=self.sec)
        return

    def hoc_display_str_list(self,**kwargs):
        ctg = kwargs["ctg"]
        ret = []
        #ret.append("$o2.soma nrnsecmenu(.5,1)")
        cell = ctg.hoc_ref(self.sec.cell())
        sec = self.sec.hname().split(".")[-1]
        ret.append(cell + "."+ sec + " nrnsecmenu("+str(self.x)+","+str(self.vartype)+")")
        return ret

class ControlMenuWidget(Widget):

    def __init__(self):
        super(ControlMenuWidget, self).__init__()
        return

    def add_expr(self):
        return
    
    def execute(self):
        h.nrncontrolmenu()
        return

    def hoc_display_str_list(self,**kwargs):
        ret = []
        ret.append("nrncontrolmenu()")
        return ret



class CellTunerGUI:
    """
    Notes:
    from neuron import h
    import neuron
    import os
    import glob

    mechanism_dir = 'biophys_components/mechanisms/'
    template_dir = 'biophys_components/hoc_templates'

    neuron.load_mechanisms(mechanism_dir)
    h_base = dir(h)

    cwd = os.getcwd()
    os.chdir(template_dir)

    hoc_templates = glob.glob("*.hoc")

    for hoc_template in hoc_templates:
        h.load_file(str(hoc_template))

    os.chdir(cwd)

    #h.load_file('biophys_components/hoc_templates/Template.hoc')
    h_loaded = dir(h)

    templates = [x for x in h_loaded if x not in h_base]

    #print(templates)
    template_name_loaded = templates[0]

    template_loaded = eval('h.'+template_name_loaded+'()')

    for sec in template_loaded.allsec():
        print(sec) #is an object
        dir(sec)
        s = sec
    
    #https://www.neuron.yale.edu/neuron/static/new_doc/programming/python.html#neuron.h.Section
    for seg in s:
        for mech in seg:
            print(sec.name(), seg.x, mech.name())

    https://www.neuron.yale.edu/neuron/static/py_doc/programming/gui/layout.html
    https://github.com/tjbanks/two-cell-hco/blob/master/graphic_library.hoc

    """
    def __init__(self, template_dir, mechanism_dir,title='NEURON GUI', tstop=250, dt=.1, print_debug=False, skip_load_mod=False,v_init=-65):
        self.template_dir = template_dir
        self.mechanism_dir = mechanism_dir
        self.title = title
        self.hoc_templates = []
        self.templates = None
        self.template_name = ""

        self.h = h
        self.hboxes = []
        self.fih = []
        
        self.clamps = []
        self.netstims = []
        self.netcons = []
        self.synapses = []
        self.other_templates = []
        

        self.display = [] # Don't feel like dealing with classes

        self.template = None #Template file used for GUI
        self.root_sec = None
        self.sections = []

        self.setup_hoc_text = []
        
        self.tstop = tstop
        self.v_init = v_init

        h.dt = dt
        self.print_debug = print_debug
        
        
        #I'm not entirely pleased with how messy this solution is but it works
        # - in regards to mechanism reading, surely NEURON has an easier/clean way builtin
        self.mechanism_files = []
        self.mechanism_parse = {}
        self.mechanism_dict = {}
        self.mechanism_point_processes = []
        if not skip_load_mod:
            self.parse_mechs()

        self.hoc_ref_template = "Cell"
        self.hoc_ref_clamps = "ccl"
        self.hoc_ref_netstims = "stim"
        self.hoc_ref_netcons = "nc"
        self.hoc_ref_syns = "syn"
        self.hoc_ref_other_templates = "auxcell"
        return 

    def get_all_h_hocobjects(self):
        ret = []
        for i in dir(h):
            try:
                if type(getattr(neuron.h,i)) == neuron.hoc.HocObject:
                    ret.append(i)
            except Exception as e:
                pass
        return ret

    def hoc_ref(self,hobject):
        found = False
        if hobject == self.template or hobject == self.template.hname():
            return self.hoc_ref_template
        else:
            clamps_name = [c.hname() for c in self.clamps]
            netstims_name = [c.hname() for c in self.netstims]
            netcons_name = [c.hname() for c in self.netcons]
            synapses_name = [c.hname() for c in self.synapses]

            if hobject in self.clamps:
                found = True
                return  self.hoc_ref_clamps + "[" + str(self.clamps.index(hobject)) + "]"
            if hobject in clamps_name:
                found = True
                return self.hoc_ref_clamps + "[" + str(clamps_name.index(hobject)) + "]"

            if hobject in self.netstims:
                found = True
                return  self.hoc_ref_netstims + "[" + str(self.netstims.index(hobject)) + "]"
            if hobject in netstims_name:
                found = True
                return self.hoc_ref_netstims + "[" + str(netstims_name.index(hobject)) + "]"

            if hobject in self.netcons:
                found = True
                return  self.hoc_ref_netcons + "[" + str(self.netcons.index(hobject)) + "]"
            if hobject in netcons_name:
                found = True
                return self.hoc_ref_netcons + "[" + str(netcons_name.index(hobject)) + "]"

            if hobject in self.synapses:
                found = True
                return  self.hoc_ref_syns + "[" + str(self.synapses.index(hobject)) + "]"
            if hobject in synapses_name:
                found = True
                return self.hoc_ref_syns + "[" + str(synapses_name.index(hobject)) + "]"

        if not found:
            import pdb;pdb.set_trace()
        
        return 

    def register_iclamp(self,iclamp):
        self.clamps.append(iclamp)

    def register_netstim(self,netstim):
        self.netstims.append(netstim)

    def register_netcon(self,netcon):
        self.netcons.append(netcon)

    def register_synapse(self,synapse):
        self.synapses.append(synapse)

    def set_title(self,window_index,title):
        self.display[window_index]['title'] = title
    def get_title(self,window_index):
        return self.display[window_index]['title']

    def add_window(self,title=None,width=1100,height=600):
        if title is None:
            title = self.template_name + " - BMTool Single Cell Tuner"

        window = {
            'title':title,
            'width':width,
            'height':height,
            'columns':[],
            '_column_objs':[]
            }
        self.display.append(window)

        window_index = len(self.display) - 1
        return window_index

    def add_column(self,window_index):
        column = {
            'widgets' : []
        }
        self.display[window_index]['columns'].append(column)
        column_index = len(self.display[window_index]['columns']) - 1
        return column_index

    def add_widget(self,window_index,column_index,widget):
        self.display[window_index]['columns'][column_index]['widgets'].append(widget)
        return len(self.display[window_index]['columns'][column_index]['widgets'])

    def new_IClamp_Widget(self,sec,dur,amp,delay):
        """
        Safely handles hoc output
        """
        iclamp = h.IClamp(sec)
        iclamp.dur = dur
        iclamp.amp = amp
        iclamp.delay = delay
        self.setup_hoc_text.append("")
        return PointMenuWidget(iclamp), iclamp

    def show(self,auto_run=False, on_complete=None,on_complete_fih=None,run_count=1):
        from neuron import gui
        fih_commands = []
        h.tstop = int(self.tstop)
        h.v_init = int(self.v_init)
        self.hboxes = []
        for window_index,window in enumerate(self.display):
            self.hboxes.append(h.HBox())
            # Instance for each column
            window['_column_objs'] = [h.VBox() for _ in range(len(window['columns']))]

            for column_index, col_vbox_obj in enumerate(window['_column_objs']):
                col_vbox_obj.intercept(True)
                column = window['columns'][column_index]
                for widget in column['widgets']:
                    ret = widget.execute()
                    if ret:
                        fih_commands.append(ret)
                col_vbox_obj.intercept(False)

            self.hboxes[window_index].intercept(True)
            for col in window['_column_objs']:
                col.map()
            self.hboxes[window_index].intercept(False)
            x = window_index * 35 #Degree of separation, will be 35 pixels apart on popup
            y = x
            self.hboxes[window_index].map(window['title'],x,y,window['width'],window['height'])

        
        #https://www.neuron.yale.edu/phpbb/viewtopic.php?f=2&t=2236
        self.fih = []
        for commands in fih_commands:
            self.fih.append(h.FInitializeHandler(0, commands))
        if on_complete_fih:
            tstop = self.tstop
            cvode = h.CVode()
            def commands_complete():
                nonlocal tstop, self
                def pdbtest():
                    nonlocal tstop, self
                    #import pdb;pdb.set_trace()
                    pass
                cvode.event(tstop,on_complete_fih)
                cvode.event(tstop,pdbtest)
                
            self.fih.append(h.FInitializeHandler(0, commands_complete))

        if auto_run:
            for _ in range(run_count):
                h.stdinit()
                h.run()   
        if on_complete:
            on_complete()
        print("Press enter to close the GUI window and continue...")
        input()
        return

    def get_mech_variables(self, sec):

        return

    def write_hoc(self, filename,mechanism_dir=None,template_dir=None,val_set=None):
        if not mechanism_dir:
            mechanism_dir = self.mechanism_dir
        if not template_dir:
            template_dir = self.template_dir

        print("Writing hoc file to " + filename)
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception as e:
                print("Error removing " + filename + ". Continuing but issues may arrise.")
        with open(filename, 'a+') as f:
            now = datetime.now()
            now_str = now.strftime("%m/%d/%Y %H:%M:%S")
            extra_sp = 49-len(now_str)
            f.write("//################################################################//\n")
            f.write("//# GUI Built using BMTool (https://github.com/tjbanks/bmtool) #//\n")
            f.write("//# Tyler Banks (tyler@tylerbanks.net)                           #//\n")
            f.write("//# Neural Engineering Laboratory (Prof. Satish Nair)            #//\n")
            f.write("//# University of Missouri, Columbia                             #//\n")
            f.write("//# Build time: " + now_str + ' '*extra_sp + "#//\n")
            f.write("//################################################################//\n")
            f.write("\n")
            #IMPORT STATEMENTS
            f.write("{load_file(\"stdrun.hoc\")}\n")
            f.write("{load_file(\"nrngui.hoc\")}\n")
            f.write("\n")
            
            #LOAD MECHANISMS
            if mechanism_dir != './' and mechanism_dir != '.':
                f.write("//Loading mechanisms in other folder\n")
                f.write("nrn_load_dll(\""+mechanism_dir+"/x86_64/.libs/libnrnmech.so\")\n")#UNIX
                f.write("nrn_load_dll(\""+mechanism_dir+"/nrnmech.dll\")\n")#WINDOWS
            f.write("\n")

            #LOAD TEMPLATES
            cwd = os.getcwd()
            f.write("// Load Template(s) (some may not be needed if a folder was specified and may cause problems, remove as needed)\n")
            #os.chdir(self.template_dir)
            #hoc_templates = glob.glob("*.hoc")
            #os.chdir(cwd)
            for hoc_template in self.hoc_templates:
                f.write("{load_file(\"" + os.path.join(template_dir,hoc_template).replace('\\','/') + "\")}\n")

            f.write("\n")
            f.write("tstop = " + str(self.tstop) + "\n")
            f.write("v_init = " + str(self.v_init) + "\n")
            f.write("objref Cell // declare the primary cell object\n")
            f.write("Cell = new " + self.template_name + "() // build the neuron from template\n")
            f.write("\n")

            f.write("NumClamps = " + str(len(self.clamps)) + "\n")
            f.write("NumStims = " + str(len(self.netstims)) + "\n")
            f.write("NumNetcons = " + str(len(self.netcons)) + "\n")
            f.write("NumSynapses = " + str(len(self.synapses)) + "\n")
            f.write("NumOtherCells = " + str(len(self.other_templates)) + "\n")
            f.write("\n")

            st = "objref " + self.hoc_ref_clamps + "[NumClamps]\n"
            if len(self.clamps) == 0:
                st = "//"+st
            f.write(st)

            st = "objref " + self.hoc_ref_netstims + "[NumStims]\n"
            if len(self.netstims) == 0:
                st = "//"+st
            f.write(st)

            st = "objref " + self.hoc_ref_netcons + "[NumNetcons]\n"
            if len(self.netcons) == 0:
                st = "//"+st
            f.write(st)

            st = "objref " + self.hoc_ref_syns + "[NumSynapses]\n"
            if len(self.synapses) == 0:
                st = "//"+st
            f.write(st)

            st = "objref " + self.hoc_ref_other_templates + "[NumOtherCells]\n"
            if len(self.other_templates) == 0:
                st = "//"+st
            f.write(st)

            f.write("\n")

            for text in self.setup_hoc_text:
                f.write(text + "\n")
            
            #f.write("\n\n")
            f.write("strdef tstr0, tstr1,tstr2,tstr3\n")
            f.write("\n")
            for window_index, window in enumerate(self.display):
                f.write("//Window " + str(window_index+1) + " variables\n")
                var_prefix = "Window"+str(window_index+1)

                f.write("strdef "+var_prefix+"BoxTitle\n")
                f.write(var_prefix+"SubVBoxNum = " + str(len(window["columns"])) + "\n")
                f.write("objref "+var_prefix+"HBoxObj,"+var_prefix+"SubVBoxObj["+var_prefix+"SubVBoxNum]\n")

                f.write("\n")
                
            f.write("\n")

            for window_index, window in enumerate(self.display):
                for column_index, column in enumerate(window["columns"]):
                    for widget_index, widget in enumerate(column["widgets"]):
                        newline = False
                        for widget_line in widget.hoc_declaration_str_list(ctg=self):
                            f.write(widget_line +"\n")
                            newline = True
                        if newline:
                            f.write("\n")
            
            # Define custom cell parameters here if present
            if val_set:
                for cell,section in val_set.items():
                    cell_prefix = self.hoc_ref(cell)
                    for section,values in section.items():
                        for param,val in values.items():
                            f.write(cell_prefix+"."+section+"."+param+" = " + str(round(val,12))+"\n")
                f.write("\n")
            
            for window_index, window in enumerate(self.display):
                window_method_prefix = "DisplayWindow"
                f.write("proc " + window_method_prefix + str(window_index+1) + "() { local i\n")
                f.write("\n")  
                var_prefix = "Window"+str(window_index+1)
                f.write("    "+var_prefix+"BoxTitle = \"" + window["title"] + "\"\n")
                f.write("    "+var_prefix+"HBoxObj = new HBox()\n")
                f.write("    for i=0,"+var_prefix+"SubVBoxNum-1 "+var_prefix+"SubVBoxObj[i] = new VBox()\n")
                f.write("\n")
            #f.write("\n")
            #for window_index, window in enumerate(self.display):
                var_prefix = "Window"+str(window_index+1)
                #f.write("    // " + var_prefix + "\n")
                for column_index, column in enumerate(window["columns"]):
                    f.write("    // Column" + str(column_index+1) + "\n")
                    f.write("    "+var_prefix+"SubVBoxObj["+str(column_index)+"].intercept(1)\n")
                    for widget_index, widget in enumerate(column["widgets"]):
                        f.write("        // Widget"+str(widget_index+1) + "\n")
                        for widget_line in widget.hoc_display_str_list(ctg=self):
                            f.write("        " + widget_line +"\n")
                    f.write("    "+var_prefix+"SubVBoxObj["+str(column_index)+"].intercept(0)\n")
                    f.write("\n")
                f.write("    "+var_prefix+"HBoxObj.intercept(1)\n")
                f.write("        for i=0,"+var_prefix+"SubVBoxNum-1 "+var_prefix+"SubVBoxObj[i].map()\n")
                f.write("    "+var_prefix+"HBoxObj.intercept(0)\n")
                x = str(window_index*35)#Degree of separation 35 pixels apart for each window so you can see them on popup
                y = x
                f.write("    "+var_prefix+"HBoxObj.map("+var_prefix+"BoxTitle,"+x+","+y+","+str(window["width"])+","+str(window["height"])+")\n")

                f.write("\n")
                f.write("}// end " + window_method_prefix + str(window_index+1) + "()\n")
                f.write("\n")

            f.write("\n")
            for window_index, window in enumerate(self.display):
                f.write(window_method_prefix + str(window_index+1) + "()\n")

        return

    def load_template(self,template_name,hoc_template_file=None):
        for sec in h.allsec():
            h("%s{delete_section()}"%sec.name())
        self.template_name = template_name
        templates = self.get_templates(hoc_template_file=hoc_template_file) #also serves to load templates
        if template_name not in templates:
            raise Exception("NEURON template not found")
        
        self.template = eval('h.'+template_name+'()')
        self.sections = [sec for sec in h.allsec()]
        root_sec = [sec for sec in h.allsec() if sec.parentseg() is None]
        assert len(root_sec) is 1
        self.root_sec = root_sec[0]
        self.other_sec = [sec for sec in h.allsec() if sec.parentseg() is not None]
        return

    def all_sections(self):
        return [sec for sec in h.allsec()]

    def get_sections(self):
        return self.sections

    def get_section_names(self):
        return [sec.name() for sec in self.get_sections()]

    def parse_mechs(self):
        from pynmodl.lems import mod2lems
        from pynmodl.nmodl import ValidationException
        from textx import TextXSyntaxError,TextXSemanticError
        from pynmodl.unparser import Unparser
        import re
        cwd = os.getcwd()
        mods_path = os.path.join(self.mechanism_dir,'modfiles')
        if not os.path.exists(mods_path):
            mods_path = os.path.join(self.mechanism_dir)
        os.chdir(mods_path)
        self.mechanism_files = glob.glob("*.mod")
        os.chdir(cwd)
        for mech in self.mechanism_files:
            # There is an issue with reading point process files
            # Or any file that uses builtin t or dt or v variables
            # Manually search for those words and make a manual list
            # of these possible point processes
            mech_suffix = ".".join(mech.split(".")[:-1]) # Regex it?
            if self.print_debug:
                print("Loading mech: " + mech)
            with open(os.path.join(mods_path,mech)) as f:
                mod_file = f.read()
                if "POINT_PROCESS" in mod_file:
                    try:
                        pp = re.search("POINT_PROCESS [A-Za-z0-9 ]*",mod_file).group().split(" ")[-1]
                        self.mechanism_point_processes.append(pp)
                    except Exception as e:
                        pass
                try:
                    #parse = mod2lems(mod_file)
                    parse = Unparser().mm.model_from_str(mod_file)
                    self.mechanism_parse[mech_suffix] = parse
                    suffix = mech_suffix
                    ranges = []
                    read_ions = []
                    for statement in parse.neuron.statements:
                        if statement.__class__.__name__ == "UseIon":
                            read_ions = read_ions + [r.name for r in statement.r[0].reads]
                        elif statement.__class__.__name__ == "Suffix":
                            suffix = statement.suffix
                        elif statement.__class__.__name__ == "Range":
                            ranges = ranges + [rng.name for rng in statement.ranges]
                    ranges = [r for r in ranges if r not in read_ions] #Remove any external ions read in
                    self.mechanism_dict[suffix] = {}
                    self.mechanism_dict[suffix]["filename"] = mech
                    self.mechanism_dict[suffix]["NEURON"] = {}
                    self.mechanism_dict[suffix]["NEURON"]["RANGE"] = ranges
                    self.mechanism_dict[suffix]["NEURON"]["USEION"] = {}
                    self.mechanism_dict[suffix]["NEURON"]["USEION"]["READ"] = read_ions

                    self.mechanism_dict[suffix]["STATE"] = {}
                    self.mechanism_dict[suffix]["STATE"]["variables"] = []
                    if hasattr(parse,'state'):
                        self.mechanism_dict[suffix]["STATE"]["variables"] = [var.name for var in parse.state.state_vars]
                    
                    self.mechanism_dict[suffix]["PARAMETER"] = [(p.name,p.unit,p.value) for p in parse.parameter.parameters]
                
                    self.mechanism_dict[suffix]["DERIVATIVE"] = []

                    parse_funcs = [func for func in parse.blocks if func.__class__.__name__ == "FuncDef"]

                    # Should extract vhalf and slope
                    boltzmann_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\+\-]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    boltzmann_activation_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\-]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    boltzmann_inactivation_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\+]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    
                    func_extract_reg = r"([A-Za-z0-9]*)\("

                    if hasattr(parse,'derivative'):
                        for statement in parse.derivative.b.stmts:
                            line = {}
                            line["unparsed"] = statement.unparsed
                            line["primed"] = False
                            line["procedure_call"] = False
                            line["variable_assignment"] = False
                            line["is_likely_activation"] = False
                            line["inf"] = ""
                            line["inf_in_range"] = False
                            line["inf_in_derivative_block"] = False
                            line["variable"] = ""
                            line["expression"] = ""
                            line["procedure"] = ""
                            try:
                                line["expression"] = statement.expression.unparsed
                            except:
                                pass
                            if statement.__class__.__name__ == "Assignment": # ex: method(v)
                                if not statement.variable:
                                    line["procedure_call"] = True
                                    try:
                                        line["procedure"] = re.search(func_extract_reg, statement.expression.unparsed).group(1)
                                        #process_procedure(line["procedure"])
                                    except AttributeError:
                                        # procedure not found in the string
                                        pass
                                else:    
                                    line["variable_assignment"] = True
                                    line["variable"] = statement.variable
                                    
                            if statement.__class__.__name__ == "Primed": # ex: m' = ... 
                                if statement.variable in self.mechanism_dict[suffix]["STATE"]["variables"]:
                                    var = statement.variable
                                    line["variable"] = var
                                    inf_reg = var + r"'\s*=\s*\(([A-Za-z0-9\*\/+\.\-\(\)]*)\s*-\s*"+var+"\)\s*\/\s[A-Za-z]*"
                                    line["primed"] = True
                                    try: 
                                        line["inf"] = re.search(inf_reg, statement.unparsed).group(1)
                                        line["is_likely_activation"] = True
                                    except AttributeError:
                                        # inf_reg not found in the original string
                                        pass
                                    if line["inf"]:
                                        if line["inf"] in ranges: # the inf expression is a defined variable in RANGES section
                                            line["inf_in_range"] = True
                                        elif line["inf"] in [l["inf"] for l in self.mechanism_dict[suffix]["DERIVATIVE"]]:
                                            line["inf_in_derivative_block"] = True

                            self.mechanism_dict[suffix]["DERIVATIVE"].append(line)

                    def is_number(s):
                        try:
                            float(s)
                            return True
                        except ValueError:
                            return False
                            
                    def get_vh_slope(inf_var,procedure):
                        func = [f for f in parse_funcs if f.name==procedure][0] # May be a crash point if not found, other issues
                        stmts = [s for s in func.b.stmts]
                        for statement in stmts:
                            if statement.__class__.__name__ == "Assignment":
                                if statement.variable.__class__.__name__ == "VarRef":
                                    if statement.variable.var.__class__.__name__ == "AssignedDef":
                                        var = statement.variable.var.name
                                        if inf_var == var:
                                            expression = statement.expression.unparsed
                                            try:
                                                vh = re.search(boltzmann_reg, expression).group(1)
                                                slope = re.search(boltzmann_reg, expression).group(2)
                                                return vh,slope,statement.unparsed
                                            except AttributeError:
                                                # inf_reg not found in the original string
                                                pass

                        return None,None,None
                    #print(mech)
                    

                    #activation to vhalf and slope matching
                    #state_activation_vars = []
                    self.mechanism_dict[suffix]["state_activation_vars"] = []
                    procedures_called = []
                    #Look at each state variable
                    for state_var in self.mechanism_dict[suffix]["STATE"]["variables"]:
                        #Look through each line of the derivative block
                        for derivative_line in self.mechanism_dict[suffix]["DERIVATIVE"]:
                            if derivative_line["procedure_call"]:
                                procedures_called.append(derivative_line["procedure"])
                            if derivative_line["variable_assignment"]:
                                # There may be a case where the inf var is set right before calculating the derivative
                                # TODO handle this case
                                pass
                            if state_var == derivative_line["variable"]: 
                                if derivative_line["is_likely_activation"]:
                                    inf_var = derivative_line["inf"]
                                    if not procedures_called:
                                        vardef = {}
                                        vardef['var'] = state_var
                                        vardef['var_inf'] = inf_var
                                        vardef['vh'] = None
                                        vardef['k'] = None
                                        vardef['procedure_set'] = None
                                        vardef['line_set'] = line
                                        self.mechanism_dict[suffix]["state_activation_vars"].append(vardef)

                                    for procedure in procedures_called:
                                        vh,slope,line = get_vh_slope(inf_var, procedure)
                                        if vh and slope:
                                            vardef = {}
                                            vardef['var'] = state_var
                                            vardef['var_inf'] = inf_var
                                            vardef['vh'] = vh
                                            vardef['k'] = slope
                                            vardef['procedure_set'] = procedure
                                            vardef['line_set'] = line
                                            self.mechanism_dict[suffix]["state_activation_vars"].append(vardef)
                                        else:
                                            vardef = {}
                                            vardef['var'] = state_var
                                            vardef['var_inf'] = inf_var
                                            vardef['vh'] = vh
                                            vardef['k'] = slope
                                            vardef['procedure_set'] = procedure
                                            vardef['line_set'] = line
                                            self.mechanism_dict[suffix]["state_activation_vars"].append(vardef)
                                            break
                                        #if not is_number(vh):
                                        #    vh_var = True
                                        #if not is_number(slope):
                                        #    slope_var = True

                except ValidationException as e:
                    if self.print_debug or True:
                        print("ValidationException: Unable to parse " + mech)
                        print(e)
                        #import pdb;pdb.set_trace()
                except TextXSyntaxError as e:
                    if self.print_debug or True:
                        print("TextXSyntaxError: Unable to parse " + mech)
                        print(e)
                        #import pdb;pdb.set_trace()
                except TextXSemanticError as e:
                    if self.print_debug or True:
                        print("TextXSemanticError: Unable to parse " + mech)
                        print(e)
                        #import pdb;pdb.set_trace()
                #except AttributeError as e:
                #    if self.print_debug or True:
                #        print("AttributeError: Unable to parse " + mech)
                #        print(e)
                #        import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        return

    def seg_mechs(self, folder=None):
        
        cwd = os.getcwd()

        if folder:
            os.chdir(os.path.abspath(folder))

        mechs = [mech.name() for mech in self.root_sec() if not mech.name().endswith("_ion")]
        
        valid_mechs = []
        for mech in mechs:
            if not self.mechanism_dict.get(mech):
                if self.print_debug:
                    click.echo("Skipping \"" + colored.green(mech) + "\" mechanism (mod file not originally parsed)")
            else:
                if self.print_debug:
                    print("Adding mechanism \"" + mech + "\" to queue")
                valid_mechs.append(mech)
        if self.print_debug:
            print("")
        mechs_processed = []
        for mech in valid_mechs:
            if self.print_debug:
                click.echo("Processing \"" + mech + "\"")

            parameters = [p[0] for p in self.mechanism_dict[mech]["PARAMETER"]]
            ranges = []
            if self.mechanism_dict[mech]["NEURON"].get("RANGE"):
                ranges = self.mechanism_dict[mech]["NEURON"]["RANGE"]

            act_vars = []
            if self.mechanism_dict[mech].get("state_activation_vars"):
                avars = self.mechanism_dict[mech]["state_activation_vars"]
                
                # If the values are already variables we don't need to process the mech further
                act_vars = [v for v in avars if v["vh"] not in parameters and v["k"] not in parameters and (v["k"]==None or float(v["k"]) < 0)]
                act_vars_names = [v["var"] for v in act_vars]
                inact_vars = [v for v in avars if v["vh"] not in parameters and v["k"] not in parameters and (v["k"]==None or float(v["k"]) > 0)]
                inact_vars_names = [v["var"] for v in inact_vars]

                if act_vars:
                    mechs_processed.append(mech)

                if len(act_vars):
                    if self.print_debug:
                        click.echo("Activation variables to process: " + colored.green(", ".join(act_vars_names)))
                if len(inact_vars):
                    if self.print_debug:
                        click.echo("Inactivation variables to process: " + colored.red(", ".join(inact_vars_names)))
            else:
                if self.print_debug:
                    print("No activation/inactivation variables")
                    print("")
                continue
            
            filename = mech + 'seg.mod' if not mech.endswith("seg") else mech + ".mod"
                           
            with open(filename, 'w+') as f:
                if self.print_debug:
                    click.echo("Writing " + colored.green(filename))
                f.write(': Ion channel activation segregation -- Generated by BMTool (https://github.com/tjbanks/bmtool)\n')
                f.write(': based on the paper "Distinct Current Modules Shape Cellular Dynamics in Model Neurons" (2016)\n\n')
                
                if mech in self.mechanism_parse:
                    code_blocks = self.mechanism_parse[mech].blocks
                else: # it may be possible that the filename is different from the actual mechanism suffix
                    mech_name = self.mechanism_dict[mech]["filename"].replace(".mod","")
                    code_blocks = self.mechanism_parse[mech_name].blocks

                for block in code_blocks:
                    if block.__class__.__name__ == "Neuron":
                        f.write("NEURON {\n")
                        for statement in block.statements:
                            st = statement.unparsed
                            if statement.__class__.__name__ == "Suffix":
                                if not statement.suffix.endswith("seg"):
                                    st = st + "seg"
                            if statement.__class__.__name__ == "UseIon":
                                st = st.replace("VALENCE +","VALENCE ") # This is a bit of a hack, as NEURON doesn't like `+2`, prefer `2`
                            f.write('\t' + st + '\n')
                        rngvars = []
                        for act_var in act_vars:
                            if act_var["vh"]:
                                rngvars.append(act_var['var']+"vhalf")
                            if act_var["k"]:
                                rngvars.append(act_var['var']+"k")
                            rngvars.append(act_var['var']+"seg")
                        if rngvars:
                            f.write("\t" + "RANGE " + ", ".join(rngvars) + "\n")
                        f.write("} \n")
                    elif block.__class__.__name__ == "Parameter":
                        f.write("PARAMETER {\n")
                        for parameter in block.parameters:
                            f.write('\t' + parameter.unparsed + '\n')
                        for act_var in act_vars:
                            act = [v for v in self.mechanism_dict[mech]["state_activation_vars"] if v['var'] == act_var['var']][0]
                            if act["vh"]:
                                f.write('\t' + act_var['var']+"vhalf = " + act["vh"] + "\n")
                            if act["k"]:
                                f.write('\t' + act_var['var']+"k = " + act["k"] + "\n")
                            f.write('\t' + act_var['var']+"seg = " + "-99" + "\n") #TODO CHANGE THIS TO USER DEFINED VARIABLE
                        f.write("}\n")

                    elif block.__class__.__name__ == "Derivative":
                        func_name = block.name
                        f.write("DERIVATIVE " + func_name + "{\n")
                        for statement in block.b.stmts:
                            statement_str = statement.unparsed
                            if statement.__class__.__name__ == "Assignment":
                                pass
                            if statement.__class__.__name__ == "Primed":
                                for act_var in act_vars:
                                    if statement.variable == act_var['var'] and act_var['var_inf'] in statement.unparsed:
                                        f.write("\t" + act_var['var'] + "segment(v)" + "\n")
                            f.write("\t" + statement_str + "\n")
                        f.write("}\n")

                    elif block.__class__.__name__ == "FuncDef":
                        func_name = block.name
                        #{'var': 'n', 'var_inf': 'inf', 'vh': '12.3', 'k': '-11.8', 'procedure_set': 'rate', 'line_set': 'inf = 1.0 / (1.0 + (exp((v + 12.3) / (-11.8))))'}
                        pars_arr = [p.unparsed for p in block.pars]
                        pars = ", ".join(pars_arr)
                        ftype = "PROCEDURE" if block.is_procedure else "FUNCTION"
                        f.write(ftype + " " + func_name + "(" + pars + "){\n")
                        for statement in block.b.stmts:
                            statement_str = statement.unparsed
                            if statement.__class__.__name__ == "Assignment":
                                for act_var in act_vars:
                                    if func_name == act_var["procedure_set"] and statement_str == act_var["line_set"]:
                                        if act["vh"]:
                                            statement_str = statement_str.replace(act_var["vh"],act_var["var"]+"vhalf",1)
                                        if act["k"]:
                                            statement_str = statement_str.replace(act_var["k"],act_var["var"]+"k",1)
                            f.write('\t' + statement_str + '\n')

                        f.write("}\n")

                    else:
                        f.write(block.unparsed)
                    f.write('\n')

                if act_vars:
                    f.write(": Segmentation functions\n\n")

                for act_var in act_vars:
                    #Create a xsegment(v) function for each variable
                    f.write("PROCEDURE " + act_var["var"] + "segment(v){\n")
                    f.write("\tif (v < " + act_var["var"] + "seg){\n")
                    f.write("\t\t" + act_var['var_inf'] + " = 0\n")
                    f.write("\t}\n")
                    f.write("}\n")
            if self.print_debug:
                print("")
            
        if folder:
            os.chdir(cwd)
            
        return mechs_processed

    def seg_template(self,outhoc, mechs_processed, hoc_template_file=None, outappend=False, returnnameonly=False):
        # open hoc template file 
        # scan through each line for template selected begintemplate
        # copy all lines until ^endtemplate TEMPL\s*$
        # go through all copied lines, replace "insert x" with "insert xseg" && "_x" with "_xseg"
        # write to outhoc and append if outappend set
        
        if mechs_processed:
            new_template_name = self.template_name+"Seg"
        else:
            new_template_name = self.template_name

        if returnnameonly:
            return new_template_name

        hoc_files = []
        if hoc_template_file:
            hoc_files.append(hoc_template_file)
        else:
            hoc_files = [os.path.join(self.template_dir, t) for t in self.hoc_templates]
        found = False

        template_text = []
        for hoc_file in hoc_files:
            if found:
                break
            with open(hoc_file, 'r') as f:
                lines = f.readlines()
                readon = False
                for line in lines:
                    if "begintemplate " + self.template_name in line:
                        readon = True
                        found = True
                    if readon:
                        template_text.append(line)
                    if "endtemplate " + self.template_name in line:
                        readon = False
        if found:
            mode = "a+" if outappend else "w+"
            #ins_mechs = ["insert " + mech for mech in mechs_processed]
            #ref_mechs = ["_ " + mech for mech in mechs_processed]
            with open(outhoc,mode) as f:
                click.echo("Writing segregated template to " + colored.green(outhoc) + " in " + mode + " mode.")
                for line in template_text:
                    line = line.replace (self.template_name, new_template_name)
                    for mech in mechs_processed:
                        line = line.replace("insert " + mech, "insert " + mech + "seg")
                        line = line.replace("_"+mech, "_"+mech+"seg")
                    f.write(line)
        else:
            print("Template "+ self.template_name +"not found in a hoc file, no file written.")

        return new_template_name

    def get_templates(self,hoc_template_file=None):
        if self.templates is None: # Can really only do this once
            ##import pdb;pdb.set_trace()
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

    def get_current_cell_values(self,cell=None,change_dict=None):
        if not cell:
            cell = self.template       
        
        #change_dict={'soma': {'g_kdrseg': 0.0, 'gbar_kdrseg': 0.3, 'ik_kdrseg': 0.0, 'inf_kdrseg': 0.0, 'n_kdrseg': 0.0, 'nk_kdrseg': -11.8, 'nseg_kdrseg': -999.0, 'nvhalf_kdrseg': 12.3, 'tau_kdrseg': 0.0, 'g_naseg': 0.0, 'gbar_naseg': 0.3, 'h_naseg': 0.0, 'hinf_naseg': 0.0, 'htau_naseg': 0.0, 'ina_naseg': 0.0, 'm_naseg': 0.0, 'minf_naseg': 0.0, 'mk_naseg': -5.29, 'mseg_naseg': -999.0, 'mtau_naseg': 0.0, 'mvhalf_naseg': 25.5, 'gbar_leak': 3e-05, 'ileak_leak': 0.0, 'dik_dv_': 0.0, 'ek': -80.0, 'ik': 0.0, 'ki': 54.4, 'ko': 2.5, 'dina_dv_': 0.0, 'ena': 50.0, 'ina': 0.0, 'nai': 10.0, 'nao': 140.0, 'dileak_dv_': 0.0, 'eleak': -60.0, 'ileak': 0.0, 'leaki': 1.0, 'leako': 1.0}}
        allsec = [a for a in h.allsec()]
        primary_cell_secs = []
        for s in allsec:
            if s.cell() == cell:
                primary_cell_secs.append(s)
        
        sections = {}

        for cellsec in primary_cell_secs:
            mechs = [mech.name() for mech in cellsec(0.5) if not mech.name().endswith("_ion")]
            ions = [mech.name() for mech in cellsec(0.5) if mech.name().endswith("_ion")]
            cellmechvars = ["cm","diam"]
    
            for mech in mechs:
                if hasattr(cellsec(0.5),mech):
                    mechobj = getattr(cellsec(0.5),mech)
                else:
                    print(mech + " not found on " + cellsec.name())
                    continue
                
                attribs = [at for at in dir(mechobj) if not at.startswith("__") and at !="name"]
                for attrib in attribs:
                    ref = attrib+"_"+mech
                    if hasattr(cellsec(0.5),ref):
                        cellmechvars.append(ref)

            for ion in ions:
                if hasattr(cellsec(0.5),ion):
                    ionobj = getattr(cellsec(0.5),ion)
                else:
                    print(ion + " not found on " + cellsec.name())
                    continue
                attribs = [at for at in dir(ionobj) if not at.startswith("__") and at !="name"]
                for attrib in attribs:
                    ref = attrib
                    if hasattr(cellsec(0.5),ref):
                        cellmechvars.append(ref)

            secname = cellsec.hname().split(".")[-1]
            sections[secname] = cellmechvars

        ret = {}
        for secname, values in sections.items():
            valpairs = {}
            compared = None
            if change_dict and change_dict.get(secname):
                compared = change_dict.get(secname)

            for v in values:
                if hasattr(eval("cell."+secname),v):
                    key = v
                    value = eval("cell."+secname+"."+v)
                    add = True
                    if compared and (compared.get(key) or (compared.get(key) == 0 and value == 0)):
                        if compared.get(key) == value:
                            add = False
                    if add:
                        valpairs[key]=value

            ret[secname] = valpairs
        #import pdb;pdb.set_trace()

        return ret