from neuron import h
import neuron
import os
import glob
import numpy as np
from datetime import datetime

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
        h.xpanel('xvarlabel')
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

    def hoc_display_str_list(self):
        return []
    
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

    def hoc_display_str_list(self):
        return []

class FICurveWidget(Widget):
    def __init__(self,template_name,i_increment=0.1,i_start=0,i_stop=1,tstart=50,
            tdur=1000,passive_amp=-0.1,passive_delay=200, record_sec="soma[0]", record_loc="0.5",
            inj_sec="soma[0]", inj_loc="0.5"):
        super()
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

    def hoc_display_str_list(self):
        return []

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
        h.nrnsecmenu(self.x,self.vartype,sec=self.sec)
        return

    def hoc_display_str_list(self):
        return []

class ControlMenuWidget(Widget):

    def __init__(self):
        super()
        return

    def add_expr(self):
        return
    
    def execute(self):
        h.nrncontrolmenu()
        return

    def hoc_display_str_list(self):
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
    def __init__(self, template_dir, mechanism_dir,title='NEURON GUI', tstop=250, dt=.1):
        self.template_dir = template_dir
        self.mechanism_dir = mechanism_dir
        self.title = title
        self.hoc_templates = []
        self.templates = None
        self.template_name = ""

        self.display = [] # Don't feel like dealing with classes

        self.template = None #Template file used for GUI
        self.root_sec = None
        self.sections = []

        self.setup_hoc_text = []
        
        self.tstop = tstop
        h.dt = dt
        return 

    def set_title(self,window_index,title):
        self.display[window_index]['title'] = title
    def get_title(self,window_index):
        return self.display[window_index]['title']

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
            f.write("objref Cell // declare the cell object\n")
            f.write("Cell = new " + self.template_name + "() // build the neuron from template\n")
            for text in self.setup_hoc_text:
                f.write(text + "\n")
            
            f.write("\n\n")
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
                        for widget_line in widget.hoc_display_str_list():
                            f.write("        " + widget_line +"\n")
                    f.write("    "+var_prefix+"SubVBoxObj["+str(column_index)+"].intercept(0)\n")
                    f.write("\n")
                f.write("    "+var_prefix+"HBoxObj.intercept(1)\n")
                f.write("        for i=0,"+var_prefix+"SubVBoxNum-1 "+var_prefix+"SubVBoxObj[i].map()\n")
                f.write("    "+var_prefix+"HBoxObj.intercept(0)\n")
                f.write("    "+var_prefix+"HBoxObj.map("+var_prefix+"BoxTitle,0,0,"+str(window["width"])+","+str(window["height"])+")\n")

            f.write("\n")
            f.write("}// end " + window_method_prefix + str(window_index) + "()\n")

            f.write("\n\n")
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

    def get_sections(self):
        return self.sections

    def get_section_names(self):
        return [sec.name() for sec in self.get_sections()]

    def get_templates(self,hoc_template_file=None):
        if self.templates is None: # Can really only do this once
            ##import pdb;pdb.set_trace()
            if self.mechanism_dir != './' and self.mechanism_dir != '.':
                neuron.load_mechanisms(self.mechanism_dir)
            h_base = dir(h)
            
            cwd = os.getcwd()
            os.chdir(self.template_dir)
            if not hoc_template_file:
                self.hoc_templates = self.glob.glob("*.hoc")
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
