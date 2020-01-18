from neuron import h
import neuron
import os
import glob
import numpy as np
from datetime import datetime
import re

class Widget:
    def __init__(self):
        return

    def execute(self):
        raise NotImplementedError

    def hoc_declaration_str_list(self,**kwargs):
        return []

    def hoc_display_str_list(self,**kwargs):
        return []

class TextWidget(Widget):
    def __init__(self,label=""):
        super(TextWidget, self).__init__()
        self.label = label
        self.mystrs = []
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
    def __init__(self,template_name,i_increment=0.1,i_start=0,i_stop=1,tstart=50,
            tdur=1000,passive_amp=-0.1,passive_delay=200, record_sec="soma[0]", record_loc="0.5",
            inj_sec="soma[0]", inj_loc="0.5"):
        super(FICurveWidget, self).__init__()
        self.template_name = template_name
        self.i_increment = float(i_increment)/1000
        self.i_start = float(i_start)/1000
        self.i_stop = float(i_stop)/1000
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
        self.ampvec = h.Vector(self.amps)

        self.passive_amp = passive_amp
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
                return
            cvode.event(0 , start_event)

            def stop_event():
                nonlocal ctstop, cvectors, cvgraph, ctemplate_name, ampvec, lenvec,camps,plenvec
                nonlocal cfir_widget
                tplenvec = [len(cvec) for cvec in cvectors]
                hzlenvec = [i * (1000/cdur) for i in tplenvec]
                for vec in hzlenvec:
                    plenvec.append(vec)
                lenvec = h.Vector(plenvec)
                cvgraph.label(ctemplate_name + " FI Curve")
                plot = lenvec.plot(cvgraph,ampvec)
                cvgraph.size(0,max(camps),0,max(lenvec)+1)
                
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
                cfir_widget.tau = time_tau / 1000
                cfir_widget.r_in = (v_diff)/(0-cfir_widget.passive_amp) #MegaOhms
                
                return
            
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

    def add_window(self,title=None,width=1000,height=600):
        if title is None:
            title = self.template_name + " - BMTools Single Cell Tuner"

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

    def show(self,auto_run=False, on_complete=None,on_complete_fih=None):
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

        if auto_run:
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

            h.stdinit()
            
            h.run()   
            if on_complete:
                on_complete()
        print("Press enter to close the GUI window and continue...")
        input()
        return

    def get_mech_variables(self, sec):

        return

    def write_hoc(self, filename):
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
            f.write("//# GUI Built using BMTools (https://github.com/tjbanks/bmtools) #//\n")
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
            if self.mechanism_dir != './' and self.mechanism_dir != '.':
                f.write("//Loading mechanisms in other folder\n")
                f.write("nrn_load_dll(\""+self.mechanism_dir+"/x86_64/.libs/libnrnmech.so\")\n")#UNIX
                f.write("nrn_load_dll(\""+self.mechanism_dir+"/nrnmech.dll\")\n")#WINDOWS
            f.write("\n")

            #LOAD TEMPLATES
            cwd = os.getcwd()
            f.write("// Load Template(s) (some may not be needed if a folder was specified and may cause problems, remove as needed)\n")
            #os.chdir(self.template_dir)
            #hoc_templates = glob.glob("*.hoc")
            #os.chdir(cwd)
            for hoc_template in self.hoc_templates:
                f.write("{load_file(\"" + os.path.join(self.template_dir,hoc_template).replace('\\','/') + "\")}\n")

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
                f.write(window_method_prefix + str(window_index+1) + "()")

        return

    def load_template(self,template_name):
        self.template_name = template_name
        templates = self.get_templates() #also serves to load templates
        if template_name not in templates:
            raise Exception("NEURON template not found")
        
        self.template = eval('h.'+template_name+'()')
        self.sections = [sec for sec in h.allsec()]
        root_sec = [sec for sec in h.allsec() if sec.parentseg() is None]
        assert len(root_sec) is 1
        self.root_sec = root_sec[0]
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

                    self.mechanism_dict[suffix]["STATE"] = {}
                    self.mechanism_dict[suffix]["STATE"]["variables"] = [var.name for var in parse.state.state_vars]

                    def process_procedure(func):
                        return

                    self.mechanism_dict[suffix]["DERIVATIVE"] = []

                    parse_funcs = [func for func in parse.blocks if func.__class__.__name__ == "FuncDef"]

                    # Should extract vhalf and slope
                    boltzmann_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\+\-]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    boltzmann_activation_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\-]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    boltzmann_inactivation_reg = r"[A-Za-z0-9\*\/+\.\-\(\)]*\s*\/\s*\([A-Za-z0-9\*\/+\.\-\(\)]*\s*\+\s*\(\s*exp\s*\(\s*\(v\s*[\+]\s*([A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\/\s*\(\s*(\-*[A-Za-z0-9\*\/+\.\-\(\)]*)\s*\)\s*\)\s*\)\s*\)\s*"
                    
                    func_extract_reg = r"([A-Za-z0-9]*)\("

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

                        line["expression"] = statement.expression.unparsed

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

                    #activation to vhalf and slope matching

                    procedures_called = []
                    #Look at each state variable
                    for state_var in self.mechanism_dict[suffix]["STATE"]["variables"]:
                        #Look through each line of the derivative block
                        for derivative_line in self.mechanism_dict[suffix]["DERIVATIVE"]:
                            if derivative_line["procedure_call"]:
                                procedures_called.append(derivative_line["procedure"])
                            if derivative_line["variable_assignment"]:
                                pass
                            if state_var == derivative_line["variable"]: 
                                if derivative_line["is_likely_activation"]:
                                    inf_var = derivative_line["inf"]
                                    for procedure in procedures_called:
                                        vh_var = False
                                        slope_var = False
                                        vh,slope,line = get_vh_slope(inf_var, procedure)
                                        if vh and slope:
                                            print(suffix)
                                            print(inf_var)
                                            print(vh)
                                            print(slope)
                                            break
                                        #if not is_number(vh):
                                        #    vh_var = True
                                        #if not is_number(slope):
                                        #    slope_var = True


                            
                    #for statement in parse.derivative.statements:
                    if mech == "na.mod":
                        import pdb;pdb.set_trace()

                except ValidationException as e:
                    if self.print_debug:
                        print("Unable to load " + mech)
                        print(e)
                except TextXSyntaxError as e:
                    if self.print_debug:
                        print("Unable to load " + mech)
                        print(e)
                except TextXSemanticError as e:
                    if self.print_debug:
                        print("Unable to load " + mech)
                        print(e)
                except AttributeError as e:
                    if self.print_debug:
                        print("Unable to load " + mech)
                        print(e)
        return

    def get_templates(self,hoc_template_file=None):
        if self.templates is None: # Can really only do this once
            ##import pdb;pdb.set_trace()
            if self.mechanism_dir != './' and self.mechanism_dir != '.':
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
