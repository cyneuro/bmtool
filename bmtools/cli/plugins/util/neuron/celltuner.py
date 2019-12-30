from neuron import h
import neuron
import os
import glob
import numpy as np

class Widget:
    def __init__(self):
        return

    def execute(self):
        raise NotImplementedError

    def hoc_str(self):
        raise NotImplementedError

class TextWidget(Widget):
    def __init__(self,label=""):
        super()
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
        h.xpanel('xvarlabel demo')
        h.xlabel(self.label)
        for mystr in self.mystrs:
            h.xvarlabel(mystr)
        h.xpanel()
        return
        
class PointMenuWidget(Widget):
    def __init__(self,pointprocess):
        super()
        self.pointprocess = pointprocess
        return
    
    def execute(self):
        h.nrnpointmenu(self.pointprocess)
        return

    def hoc_str(self):
        return ""
    
class PlotWidget(Widget):

    def __init__(self, tstart=0,tstop=50,miny=-80,maxy=50):
        super()
        self.tstart = tstart
        self.tstop = tstop
        self.miny = miny
        self.maxy = maxy
        self.graph = None
        self.current_index = 0
        self.color = 1
        self.expressions = {}
        return

    def advance_color(self):
        #https://www.neuron.yale.edu/neuron/static/py_doc/visualization/graph.html#Graph.color
        self.color = self.color + 1
        if self.color == 10:
            self.color = 1
        self.graph.color(self.color)

    def add_expr(self,variable,text):
        self.expressions[text] = variable
        return
    
    def execute(self):
        self.graph = h.Graph()
        for text, variable in self.expressions.items():
            #self.graph.addvar('soma(0.5).v', my_cell.soma(0.5)._ref_v)
            self.graph.addvar(text,variable)
            self.advance_color()
        self.graph.size(self.tstart,self.tstop,self.miny,self.maxy)
        h.graphList[0].append(self.graph)
        return

    def hoc_str(self):
        return ""

class FICurveWidget(Widget):
    def __init__(self,template_name,i_increment=0.1,i_start=0,i_stop=1,tstart=50,tdur=1000,passive_amp=-0.1,passive_delay=200):
        super()
        self.template_name = template_name
        self.i_increment = float(i_increment)/1000
        self.i_start = float(i_start)/1000
        self.i_stop = float(i_stop)/1000
        self.tstart = tstart
        self.tdur = tdur
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

        self.passive_src = h.IClamp(self.passive_cell.soma[0](0.5))
        self.passive_src.delay = self.tstart
        self.passive_src.dur = self.tdur
        self.passive_src.amp = self.passive_amp
        
        self.passive_nc = h.NetCon(self.passive_cell.soma[0](0.5)._ref_v,None,sec=self.passive_cell.soma[0]) 
        self.passive_nc.threshold = 0
        self.passive_vec = h.Vector()
        self.passive_vec.record(self.passive_cell.soma[0](0.5)._ref_v)

        for i, amp in enumerate(self.amps):
            #Injection
            cell = self.cells[i]

            src = h.IClamp(cell.soma[0](0.5))
            src.delay = self.tstart
            src.dur = self.tdur
            src.amp = amp
            self.sources.append(src)

            #Recording
            nc = h.NetCon(cell.soma[0](0.5)._ref_v,None,sec=cell.soma[0])
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

    def hoc_str(self):
        return ""

class SecMenuWidget(Widget):
    def __init__(self, sec, x=0.5, vartype=1):
        """
        vartype=1,2,3 shows parameters, assigned, or states respectively.
        0 < x < 1 shows variables at segment containing x changing these variables changes only the values in that segment eg. equivalent to section.v(.2) = -65
        """

        self.x = x
        self.vartype = vartype
        self.sec = sec
        return

    def execute(self):
        h.nrnsecmenu(self.x,self.vartype,self.sec)
        return

    def hoc_str(self):
        return ""

class ControlMenuWidget(Widget):

    def __init__(self):
        super()
        return

    def add_expr(self):
        return
    
    def execute(self):
        h.nrncontrolmenu()
        return

    def hoc_str(self):
        return ""



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
    def __init__(self, template_dir, mechanism_dir,title='NEURON GUI', tstop=250, dt=.1):
        self.template_dir = template_dir
        self.mechanism_dir = mechanism_dir
        self.title = title
        self.templates = None

        self.display = [] # Don't feel like dealing with classes

        self.template = None #Template file used for GUI
        self.sections = []

        self.setup_hoc_text = []
        
        self.tstop = tstop
        h.dt = dt
        return 

    def set_title(self,title):
        self.title = title

    def add_window(self,title="BMTools NEURON Cell Tuner",width=1000,height=600):
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
        return 

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

    def show(self,auto_run=False, on_complete=None):
        from neuron import gui
        fih_commands = []
        h.tstop = self.tstop
        for window_index,window in enumerate(self.display):
            hBoxObj = h.HBox()
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

            hBoxObj.intercept(True)
            for col in window['_column_objs']:
                col.map()
            hBoxObj.intercept(False)
            hBoxObj.map(window['title'],0,0,window['width'],window['height'])

        if auto_run:
            #https://www.neuron.yale.edu/phpbb/viewtopic.php?f=2&t=2236
            fih = []
            for commands in fih_commands:
                fih.append(h.FInitializeHandler(0, commands))
            if on_complete:
                tstop = self.tstop
                cvode = h.CVode()
                def commands_complete():
                    nonlocal tstop
                    cvode.event(tstop,on_complete)
                    
                fih.append(h.FInitializeHandler(0, commands_complete))
                #on_complete()

            h.stdinit()
            h.run()
        
        print("Press enter to close the GUI window and continue...")
        input()
        return
        
    def write_hoc(self, filename):
        print("Writing hoc file to " + filename)
        for text in self.setup_hoc_text:
            pass
        return

    def load_template(self,template_name):
        templates = self.get_templates() #also serves to load templates
        if template_name not in templates:
            raise Exception("NEURON template not found")
        
        self.template = eval('h.'+template_name+'()')
        self.sections = [sec for sec in self.template.all]
        return

    def get_sections(self):
        return self.sections

    def get_section_names(self):
        return [sec.name() for sec in self.get_sections()]

    def get_templates(self):
        if self.templates is None: # Can really only do this once
            neuron.load_mechanisms(self.mechanism_dir)
            h_base = dir(h)

            cwd = os.getcwd()
            os.chdir(self.template_dir)

            hoc_templates = glob.glob("*.hoc")

            for hoc_template in hoc_templates:
                h.load_file(str(hoc_template))

            os.chdir(cwd)

            #h.load_file('biophys_components/hoc_templates/Template.hoc')
            h_loaded = dir(h)

            self.templates = [x for x in h_loaded if x not in h_base]

        return self.templates
