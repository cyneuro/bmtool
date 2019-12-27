from neuron import h
import neuron
import os
import glob

class Widget:
    def __init__():
        return

    def execute():
        raise NotImplementedError

    def hoc_str():
        raise NotImplementedError

class PlotWidget(Widget):

    def __init__(self):
        super()
        return

    def add_expr(self):
        return
    
    def execute(self):
        h.Graph()
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
    def __init__(self, template_dir, mechanism_dir,title='NEURON GUI'):
        self.template_dir = template_dir
        self.mechanism_dir = mechanism_dir
        self.title = title
        self.templates = None

        self.display = [] # Don't feel like dealing with classes

        self.template = None #Template file used for GUI
        self.sections = []
        return 

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


    def show(self):
        """
        Thread blocking.
        """

        from neuron import gui
        for window_index,window in enumerate(self.display):
            hBoxObj = h.HBox()
            # Instance for each column
            window['_column_objs'] = [h.VBox() for _ in range(len(window['columns']))]

            for column_index, col_vbox_obj in enumerate(window['_column_objs']):
                col_vbox_obj.intercept(True)
                column = window['columns'][column_index]
                for widget in column['widgets']:
                    widget.execute()
                col_vbox_obj.intercept(False)

            hBoxObj.intercept(True)
            for col in window['_column_objs']:
                col.map()
            hBoxObj.intercept(False)
            hBoxObj.map(window['title'],0,0,window['width'],window['height'])
            print("Press enter to close the GUI window and continue...")
            input()
        return

    def write_hoc(self, filename):
        print("Writing hoc file to " + filename)
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
