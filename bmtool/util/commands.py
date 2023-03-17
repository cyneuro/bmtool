from os import system, name
import click
import logging
import os
import questionary
import json
import tempfile
import shutil
import glob

from clint.textui import puts, colored, indent

from .util import load_config

@click.group('util')
@click.option('-c', '--config', type=click.Path(), default='./simulation_config.json', help='Configuration file to use, default: "simulation_config.json"')
@click.pass_context
def cli(ctx, config):
    config_path = os.path.abspath(os.path.expanduser(config)).replace("\\","/")

    ctx.obj["config"] = config_path

    if not os.path.exists(config_path):
        #click.echo(colored.red("Config file not found: " + config))
        pass

def check_neuron_installed(confirm=True):
    try:
        import neuron
    except ModuleNotFoundError as e:
        print("Error: Python NEURON was not found.")
        if not confirm or not questionary.confirm("Do you want to continue anyway? ").ask():
            return False
    return True

@click.group('cell', help="Access various utilities for manipulating your cell")
@click.option('--hoc-folder', type=click.STRING, default=None, help="override the default cell picker from the simulation config hoc location")
@click.option('--mod-folder', type=click.STRING, default=None, help="override the default simulation config mod file location")
@click.option('--template', type=click.STRING, default=None, help="supply template name and skip interactive mode question")
@click.option('--hoc', type=click.STRING, default=None, help="loads a single hoc file, best for directories with multiple NON-Template hoc files, specify --hoc TEMPLATE_FILE.hoc")
@click.option('--prefab',type=click.BOOL,default=False,is_flag=True,help="Downloads a set of pre-defined cells to the current directory")
@click.option('--prefab-repo', type=click.STRING, default="https://github.com/tjbanks/bmtool-cell-prefab", help="Override the github repository URL to download pre-defined cells from (default: https://github.com/tjbanks/bmtool-cell-prefab)")
@click.option('--prefab-branch', type=click.STRING, default="master", help="Override the github repository branch (default: master)")
@click.option('--prefab-refresh',type=click.BOOL,default=False,is_flag=True,help="Delete cached cells directory and re-download from prefab repository (WARNING: deletes everything in the folder)")
@click.option('--prefab-no-compile',type=click.BOOL,default=False,is_flag=True,help="Don't attempt to (re-)compile prefab cell mod files")
@click.pass_context
def cell(ctx,hoc_folder,mod_folder,template,hoc,prefab,prefab_repo,prefab_branch,prefab_refresh,prefab_no_compile):
  
    if not check_neuron_installed():
        return   

    hoc_template_file = None
    prefab_dict = None
    prefab_zip_url = prefab_repo + "/archive/" + prefab_branch + ".zip"
    prefab_repo_name = prefab_repo.split("/")[-1]
    prefab_directory = "./" + prefab_repo_name + "-" + prefab_branch
    prefab_dict_file = "PREFAB.bmtool"
    prefab_template_file = "template.hoc"
    prefab_location = ""

    dl = False
    if prefab: # User has elected to use prefab cells
        if os.path.exists("./"+prefab_dict_file): #If the current directory contains the repository continue
            prefab_location = "./"

            #mod_folder = "./" + os.path.relpath("./").replace("\\","/")
            #hoc_folder = "./" + os.path.relpath("./").replace("\\","/")
            #hoc_template_file = prefab_template_file
            if prefab_refresh:
                click.echo(colored.red("Refresh selected -- Change directory to parent directory (cd ..) and re-run to refresh"))

        else: # Check if current directory contains the repository

            if os.path.exists("./" + prefab_directory + "/" + prefab_dict_file):
                if prefab_refresh:
                    shutil.rmtree("./"+prefab_directory)
                    dl = True
            else: # The folder/file we're looking for doesn't exist
                dl = True

            if dl: # Download the repo
                click.echo(colored.green("Downloading premade cells from " + prefab_zip_url))
                import requests, zipfile, io
                r = requests.get(prefab_zip_url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall()
                
            prefab_location = "./" + os.path.relpath(prefab_directory).replace("\\","/")
  
            
        with open(os.path.join(prefab_location,prefab_dict_file), 'r') as f:
            prefab_dict = json.load(f)

        
        prefab_cells = prefab_dict.get("cells") or []
        
        if not template:
            template = questionary.select(
            "Select a cell:",
            choices=prefab_cells).ask()

        mod_folder = prefab_location + "/" + prefab_cells.get(template).get("mod_folder").replace("\\","/")
        hoc_folder = prefab_location + "/" + prefab_cells.get(template).get("hoc_folder").replace("\\","/")
        hoc_template_file =  prefab_cells.get(template).get("hoc_template_file").replace("\\","/")

        if not prefab_no_compile:
            click.echo(colored.green("Compiling mod files..."))
            cwd = os.getcwd()
            os.chdir(mod_folder)
            ret = os.system("nrnivmodl")
            os.chdir(cwd)

            if not ret:
                click.echo(colored.green("COMPILATION COMPLETE"))
            else:
                click.echo(colored.red("nrnivmodl may not have been run, execute nrnivmodl or mknrndll manually in the `")+colored.green(os.path.abspath(mod_folder))+colored.red("` folder then press enter... SKIP THIS IF YOU HAVE ALREADY COMPILED"))
                input()
            
    else:
        if hoc:
            if not hoc_folder:
                hoc_folder = './'
            if not mod_folder:
                mod_folder = './'
            hoc_template_file = hoc
        elif not hoc_folder or not mod_folder:
            try:
                cfg = load_config(ctx.obj['config'])
                if not hoc_folder:
                    hoc_folder = cfg['components']['templates_dir']
                if not mod_folder:
                    mod_folder = cfg['components']['mechanisms_dir']  
            except Exception as e:
                #lazy way of passing cases where sim config is not found and template provided
                if not hoc_folder:
                    print("Setting hoc folder to ./")
                    hoc_folder = '.'
                if not mod_folder:
                    print("Setting mod folder to ./")
                    mod_folder = '.'

    ctx.obj["hoc_folder"] = hoc_folder
    ctx.obj["mod_folder"] = mod_folder
    ctx.obj["hoc_template_file"] = hoc_template_file
    ctx.obj["cell_template"] = template
    ctx.obj["prefab"] = prefab_dict
    
    return

cli.add_command(cell)

class Builder():
    def __init__(self):
        self.title = ""
        self._options = {}
        self._exit_option = {}
        return

    def run(self):
        def clear(): 
            # for windows 
            if name == 'nt': 
                _ = system('cls') 
            # for mac and linux(here, os.name is 'posix') 
            else: 
                _ = system('clear') 
                
        prompter = self
        while prompter:
            clear()
            prompter = prompter.prompt()
        clear()

        return

    def prompt(self):
        options = list(self._options.keys()) 
        options_ind = [str(i+1)+") "+op for i,op in enumerate(options)]
        if bool(self._exit_option):
            exit_option = list(self._exit_option.keys()) 
            exit_option_ind = [str(i)+") "+op for i,op in enumerate(exit_option)]#should only be one
            options = options + exit_option
            options_ind = options_ind + exit_option_ind
        
        selected = questionary.select(
        self.title,
        choices=options_ind).ask()

        selected = options[options_ind.index(selected)]
        
        if self._exit_option.get(selected):
            _prompt = self._exit_option[selected][0]()
        else:
            _prompt = self._options[selected][0]()
        return _prompt

    def register(self, text, handler,args=None,is_exit=False):
        if is_exit:
            self._exit_option.clear()
            self._exit_option[text] = (handler,args) 
        else:
            self._options[text] = (handler,args)
        return



class BaseBuilder(Builder):
    def __init__(self, ctg):
        super(BaseBuilder, self).__init__()
        self.title = "Main Menu"
        self.ctg = ctg
        self.register_all()
        return

    def register_all(self):

        def new_window():
            return WindowBuilder(self,self.ctg)
        
        def print_ctx():
            print("Not currently implemented")
            print("Press enter to continue...")
            input()
            return self
        
        def write_to_hoc():
            #print("Enter the name of the file you wish to write: (eg: cellgui.hoc)")
            filename = questionary.text("Enter the name of the file you wish to write: (eg: cellgui.hoc)",default="cellgui.hoc").ask()
            self.ctg.write_hoc(filename)
            print("Done. Press enter to continue...")
            input()
            return self

        def set_tstop():
            tstop = questionary.text("Set tstop (ms): )",default=str(self.ctg.tstop)).ask()
            self.ctg.tstop = tstop
            return self

        def set_v_init():
            v_init = questionary.text("Set v_init (mV): )",default=str(self.ctg.v_init)).ask()
            self.ctg.v_init = v_init
            return self

        def finished():
            return None

        self.register("New Window", new_window)
        self.register("Display Current Setup", print_ctx)
        self.register("Set tstop", set_tstop)
        self.register("Set v_init", set_v_init)
        self.register("Write to HOC executable", write_to_hoc)
        self.register("Finish and Display", finished ,is_exit=True)
 
class WindowBuilder(Builder):
    def __init__(self, parent, ctg):
        super(WindowBuilder, self).__init__()
        self.parent = parent
        self.ctg = ctg
        self.window_index = ctg.add_window()
        self.title = self.ctg.get_title(self.window_index)
        self.register_all()
        return

    def register_all(self):

        def set_title():
            print("Type the new title for this window")
            self.ctg.set_title(self.window_index,input())
            self.title = self.ctg.get_title(self.window_index)
            return self
        
        def new_column():
            return ColumnBuilder(self,self.ctg)
        
        def finished():
            return self.parent

        self.register("Add Column", new_column)
        self.register("Set Window Title", set_title)
        self.register("Finish Window", finished ,is_exit=True)

class ColumnBuilder(Builder):
    def __init__(self, parent, ctg):
        super(ColumnBuilder, self).__init__()
        self.parent = parent
        self.ctg = ctg
        self.column_index = ctg.add_column(self.parent.window_index)
        self.title = "Window " + str(self.parent.window_index+1) + " Column " + str(self.column_index+1)
        self.register_all()
        return

    def register_all(self):
        
        def new_widget():
            print("Not currently implemented")
            print("Press enter to continue...")
            input()
            return self
        
        def new_plot_widget():
            return PlotWidgetBuilder(self,self.ctg)
        
        def new_controlmenu_widget():
            from .neuron.celltuner import ControlMenuWidget
            widget = ControlMenuWidget()
            self.ctg.add_widget(self.parent.window_index, self.column_index,widget)
            print("Done. Press enter to continue...")
            input()
            return self

        def new_pointmenu_widget():
            return PointMenuWidgetBuilder(self,self.ctg)

        def new_secmenu_widget():
            return SecMenuWidgetBuilder(self,self.ctg)

        def finished():
            return self.parent

        #self.register("Add Widget", new_widget)
        self.register("Add Plot Widget", new_plot_widget)
        self.register("Add Control Menu Widget (Init & Run)", new_controlmenu_widget)
        self.register("Add SecMenu Widget (Section Variables)", new_secmenu_widget)
        self.register("Add Point Menu Widget (Current Clamp, Netstim)", new_pointmenu_widget)
        self.register("Finish Column", finished ,is_exit=True)

class SecMenuWidgetBuilder(Builder):
    def __init__(self, parent, ctg):
        super(SecMenuWidgetBuilder, self).__init__()

        from .neuron.celltuner import SecMenuWidget

        self.parent = parent
        self.ctg = ctg
        self.title =  "(Section Menu Widget)"
        self.register_all()
        return

    def register_all(self):
        
        def select():
            from .neuron.celltuner import SecMenuWidget
            cell_options = []
            cell_options_obj = []
            cell_options.append(self.ctg.template.hname())
            cell_options_obj.append(self.ctg.template)
            
            cell_selected = questionary.select(
            "Select the Cell",
            choices=cell_options).ask()

            section_options = []
            section_options_obj = []
            all_sections = self.ctg.all_sections()
            section_options_obj = [s for s in all_sections if s.hname().startswith(cell_selected)]
            section_options = [s.hname() for s in section_options_obj]

            section_selected = questionary.select(
            "Select the Section",
            choices=section_options).ask()

            section_selected_obj = section_options_obj[section_options.index(section_selected)]
            section_location = questionary.text("Enter recording location (default:0.5): ",default="0.5").ask()

            self.widget = SecMenuWidget(section_selected_obj,x=float(section_location))
            self.widget_index = self.ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget)
        
            return self.parent

        def finish():
            return self.parent

        self.register("Select Section", select)
        self.register("Return without adding widget", finish ,is_exit=True)
        return

class PointMenuWidgetBuilder(Builder):
    def __init__(self, parent, ctg):
        super(PointMenuWidgetBuilder, self).__init__()

        self.parent = parent
        self.ctg = ctg
        self.title =  "(Point Menu Widget)"
        self.register_all()
        return

    def register_all(self):

        def select_section_location():
            cell_options = []
            cell_options_obj = []
            cell_options.append(self.ctg.template.hname())
            cell_options_obj.append(self.ctg.template)
            
            cell_selected = questionary.select(
            "Select the Cell",
            choices=cell_options).ask()

            section_options = []
            section_options_obj = []
            all_sections = self.ctg.all_sections()
            section_options_obj = [s for s in all_sections if s.hname().startswith(cell_selected)]
            section_options = [s.hname() for s in section_options_obj]

            section_selected = questionary.select(
            "Select the Section",
            choices=section_options).ask()

            section_selected_obj = section_options_obj[section_options.index(section_selected)]
            section_location = questionary.text("Enter location (default:0.5): ",default="0.5").ask()

            return section_selected_obj, section_location

        def new_clamp():
            from .neuron.celltuner import PointMenuWidget
            
            section_selected_obj, section_location = select_section_location()

            delay = float(questionary.text("Enter default iclamp delay (default:0): ",default="0").ask())
            dur = float(questionary.text("Enter default iclamp duration (default:100): ",default="100").ask())
            amp = float(questionary.text("Enter default iclamp amp(mA) (default:1): ",default="1").ask())            

            self.widget = PointMenuWidget(None)
            iclamp = self.widget.iclamp(section_selected_obj(float(section_location)),dur,amp,delay)
            self.ctg.register_iclamp(iclamp)
            self.widget_index = self.ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget)

            return self.parent

        def new_netstim():
            from .neuron.celltuner import PointMenuWidget
            section_selected_obj, section_location = select_section_location()

            notlisted = "Synapse not listed"

            synapse_options = self.ctg.mechanism_point_processes[:]
            synapse_options = synapse_options + ["AlphaSynapse","Exp2Syn","ExpSyn"] #Builtins
            synapse_options.append(notlisted)

            synapse_selected = questionary.select(
            "Select the Synapse type from the most likely options",
            choices=synapse_options).ask()

            if synapse_selected == notlisted:
                synapse_options = [i for i in self.ctg.get_all_h_hocobjects()]
                synapse_selected = questionary.select(
                "Select the Synapse (ALL HOCOBJECTS)",
                choices=synapse_options).ask()

            interval = int(questionary.text("Enter default netstim interval (ms (mean) time between spikes): ",default="50").ask())
            number = int(questionary.text("Enter default netstim number of events ((average) number of spikes): ",default="10").ask())
            start = int(questionary.text("Enter default netstim start (ms (most likely) start time of first spike): ",default="0").ask())
            noise = float(questionary.text("Enter default netstim noise (range 0 to 1. Fractional randomness.): ",default="0").ask())
            weight = float(questionary.text("Enter default netcon weight (range 0 to 1. Default: 1): ",default="1").ask())
            
            self.widget = PointMenuWidget(None)
            self.widget_extra = PointMenuWidget(None)
            synapse = self.widget_extra.synapse(section_selected_obj,section_location,synapse_selected)
            netstim,netcon = self.widget.netstim(interval,number,start,noise,target=synapse,weight=weight)
            self.ctg.register_netstim(netstim)
            self.ctg.register_netcon(netcon)
            self.ctg.register_synapse(synapse)

            self.widget_index_extra = self.ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget_extra)
            self.widget_index = self.ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget)
            
            return self.parent

        def finish():
            return self.parent

        self.register("Add Netstim to Cell and Insert Widgets", new_netstim)
        self.register("Add Current Clamp to Cell and Insert Widget", new_clamp)
        self.register("Finished", finish ,is_exit=True)        
        return

class PlotWidgetBuilder(Builder):
    def __init__(self, parent, ctg):
        super(PlotWidgetBuilder, self).__init__()

        from .neuron.celltuner import PlotWidget

        self.parent = parent
        self.ctg = ctg
        self.widget = PlotWidget(tstop=self.ctg.tstop)
        self.widget_index = ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget)
        self.title = "Window " + str(self.parent.parent.window_index + 1) + " Column " + \
            str(self.parent.column_index + 1) + " Widget " + str(self.widget_index) + " (Plot Widget)"
        self.register_all()
        return

    def register_all(self):
            
        def new_expression():
            obj_options = ["Cell","Quick - Template Cell.soma Membrane Voltage (0.5)"]
            obj_selected = questionary.select(
            "Select the object type to plot",
            choices=obj_options).ask()
            if obj_selected == obj_options[0]:
                
                cell_options = []
                cell_options_obj = []
                cell_options.append(self.ctg.template.hname())
                cell_options_obj.append(self.ctg.template)
                
                cell_selected = questionary.select(
                "Select the Cell",
                choices=cell_options).ask()

                section_options = []
                section_options_obj = []
                all_sections = self.ctg.all_sections()
                section_options_obj = [s for s in all_sections if s.hname().startswith(cell_selected)]
                section_options = [s.hname() for s in section_options_obj]

                section_selected = questionary.select(
                "Select the Section",
                choices=section_options).ask()

                section_selected_obj = section_options_obj[section_options.index(section_selected)]
                section_location = questionary.text("Enter recording location (default:0.5): ",default="0.5").ask()
                
                mechs = [mech.name() for mech in section_selected_obj(float(section_location)) if not mech.name().endswith("_ion")]            

                variable_options = []
                variable_options.append("v") # builtin voltage variable
                for mech in mechs:
                    if self.ctg.mechanism_dict.get(mech):
                        ranges = self.ctg.mechanism_dict[mech]["NEURON"]["RANGE"]
                        variable_options = variable_options + [rng + "_" + mech for rng in ranges]

                variables_selected = questionary.checkbox(
                "Select the Variables",
                choices=variable_options).ask()

                #sec_text = self.ctg.root_sec.hname().split('.')[-1]+"(.5)"
                #self.widget.add_expr(self.ctg.root_sec(0.5)._ref_v,sec_text,hoc_text="%s.soma.v(0.5)",hoc_text_obj=self.ctg.template)
                
                for variable_selected in variables_selected:
                    #sec_var_ref = exec("section_selected_obj(float(section_location))."+variable_selected)
                    sec_var_ref = getattr(section_selected_obj(float(section_location)),"_ref_"+variable_selected)
                    sec_text = section_selected_obj.hname().split('.')[-1]+"("+section_location+")."+variable_selected
                    sec_hoc_text = section_selected.split('.')[-1]
                    hoc_text = "%s." + sec_hoc_text +"."+ variable_selected + "(" + section_location +")"
                    hoc_text_obj = cell_selected
                    
                    self.widget.add_expr(sec_var_ref,sec_text,hoc_text=hoc_text,hoc_text_obj=hoc_text_obj)

                #import pdb;pdb.set_trace()

            elif obj_selected == obj_options[1]:
                sec_text = self.ctg.root_sec.hname().split('.')[-1]+"(.5)"
                self.widget.add_expr(self.ctg.root_sec(0.5)._ref_v,sec_text,hoc_text="%s.soma.v(0.5)",hoc_text_obj=self.ctg.template)
                print("Captured. Press enter to continue...")
                input()
            return self
        
        def finished():
            return self.parent

        self.register("Add Expression", new_expression)
        self.register("Finish Widget", finished ,is_exit=True)

@cell.command('tune', help="Creates a NEURON GUI window with everything you need to tune a cell")
@click.option('--easy', type=click.BOOL, default=None, is_flag=True, help="Builds a simple GUI with no walkthrough")
@click.option('--builder', type=click.BOOL, default=None, is_flag=True, help="A commandline walkthrough for building your own GUI")
@click.option('--write-hoc', type=click.STRING, default=None, help="write a standalone hoc file for your GUI, supply filename")
@click.option('--hide', type=click.BOOL, default=False, is_flag=True, help="hide the interface that shows automatically after building the GUI")
@click.option('--title',type=click.STRING,default=None)
@click.option('--tstop',type=click.INT,default=250)
@click.option('--debug', type=click.BOOL, default=False, is_flag=True, help="Print debug messages and errors")
@click.pass_context
def cell_tune(ctx,easy,builder,write_hoc,hide,title,tstop,debug):#, title, populations, group_by, save_file):
    print("Loading...")
    from .neuron.celltuner import CellTunerGUI, PlotWidget, ControlMenuWidget, SecMenuWidget

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    ctg = CellTunerGUI(hoc_folder,mod_folder,title=title,print_debug=debug)
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    
    # Cell selector
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()

    ctg.load_template(template)
    if not title:
        title = template + " - Cell Configurator - Interface generated by BMTool (https://github.com/tjbanks/bmtool)"

    # Mode selector
    if easy is None and builder is None:
        easy = questionary.confirm("Use pre-built interface? (no for advanced mode) ").ask()
    
    if easy:
        #Window 1
        window_index = ctg.add_window(title=title)
        #Column 1
        column_index = ctg.add_column(window_index)
        plot_widget = PlotWidget(tstop=tstop)
        sec_text = ctg.root_sec.hname().split('.')[-1]+"(.5)"
        plot_widget.add_expr(ctg.root_sec(0.5)._ref_v,sec_text)
        ctg.add_widget(window_index,column_index,plot_widget)
        
        if len(ctg.sections) > 1:
            plot_widget = PlotWidget(tstop=tstop)
            for sec in ctg.sections:
                sec_text = sec.hname().split('.')[-1]+"(.5)"
                plot_widget.add_expr(sec(0.5)._ref_v,sec_text)
            ctg.add_widget(window_index,column_index,plot_widget)
        
        #Column 2
        column_index = ctg.add_column(window_index)
        for i in range(len(ctg.sections)):#regular iteration was acting funny
            #import pdb;pdb.set_trace()
            sec_menu_widget = SecMenuWidget(ctg.sections[i])
            ctg.add_widget(window_index,column_index,sec_menu_widget)

        #Column 3
        column_index = ctg.add_column(window_index)
        control_widget = ControlMenuWidget()
        ctg.add_widget(window_index,column_index,control_widget)
        iclamp_widget, iclamp = ctg.new_IClamp_Widget(ctg.sections[0](0.5),200,0.1,25)
        ctg.add_widget(window_index,column_index,iclamp_widget)
        
    else:        
        cmd_builder = BaseBuilder(ctg)
        cmd_builder.run()
        

    # Section selector
    #section_names = ctg.get_section_names()

    #sections_selected = questionary.checkbox(
    #'Select sections you want to configure (each will recieve a window):',
    #choices=section_names).ask()

    # Display selector
    #displays_available = ['Voltages', 'Currents', 'Conductances', 'FIR']
    #inputs_available = ['Current Clamp', 'Spike Input']
    #configuration_available = ['Parameter']

    #Do you want to select which currents to plot?
    #import pdb;pdb.set_trace()
    if write_hoc:
        ctg.write_hoc(write_hoc)

    if not hide:
        ctg.show()

@cell.command('fi', help="Creates a NEURON GUI window with FI curve and passive properties")
#@click.option('--easy', type=click.BOOL, default=None, is_flag=True, help="override the default simulation config mod file location")
#@click.option('--write-hoc', type=click.STRING, default=None, help="write a standalone hoc file for your GUI, supply filename")
#@click.option('--hide', type=click.BOOL, default=False, is_flag=True, help="hide the interface that shows automatically after building the GUI")
@click.option('--title',type=click.STRING,default=None)
@click.option('--min-pa',type=click.INT,default=0,help="Min pA for injection")
@click.option('--max-pa',type=click.INT,default=1000,help="Max pA for injection")
@click.option('--passive-delay',type=click.INT,default=650,help="Wait n ms before determining steadystate value (default: 650)")
@click.option('--increment',type=click.FLOAT,default=100,help="Increment the injection by [i] pA")
@click.option('--tstart',type=click.INT,default=150, help="Injection start time")
@click.option('--tdur',type=click.INT,default=1000,help="Duration of injection default:1000ms")
@click.option('--advanced',type=click.BOOL,default=False,is_flag=True,help="Interactive dialog to select injection and recording points")
@click.pass_context
def cell_fir(ctx,title,min_pa,max_pa,passive_delay,increment,tstart,tdur,advanced):#, title, populations, group_by, save_file):
    
    from .neuron.celltuner import CellTunerGUI, TextWidget, PlotWidget, ControlMenuWidget, SecMenuWidget, FICurveWidget

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    tstop = tstart+tdur

    ctg = CellTunerGUI(hoc_folder,mod_folder,tstop=tstop,skip_load_mod=True)
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    
    # Cell selector
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()

    ctg.load_template(template)
    if not title:
        title = template + " - Cell FI Curve - Interface generated by BMTool (https://github.com/tjbanks/bmtool)"
        #ctg.set_title(title)
    
    inj_sec = ctg.root_sec.hname()
    rec_sec = ctg.root_sec.hname()

    inj_loc = "0.5"    
    rec_loc = "0.5"
    
    if advanced:
        inj_sec = questionary.select(
            "Select the current injection segment: ",
            choices=ctg.get_section_names()
        ).ask()
        inj_loc = questionary.text("Enter current injection segment location (eg:0.5): ").ask()
        rec_sec = questionary.select(
            "Select the recording segment: ",
            choices=ctg.get_section_names()
        ).ask()
        rec_loc = questionary.text("Enter recording segment location (eg:0.5): ").ask()
        
    
    rec_sec_split = rec_sec.split('.')[-1]
    inj_sec_split = inj_sec.split('.')[-1]

    click.echo("Using section " + colored.green(inj_sec_split + "("+inj_loc+")") + " for injection")
    click.echo("Using section " + colored.green(rec_sec_split + "("+rec_loc+")") + " for recording")
    
    #Window 1
    window_index = ctg.add_window(title=title,width=800,height=650)
    #Column 1
    column_index = ctg.add_column(window_index)
    fir_widget = FICurveWidget(template,i_increment=increment,i_start=min_pa,i_stop=max_pa,tstart=tstart,tdur=tdur,passive_delay=passive_delay,
        record_sec=rec_sec_split, record_loc=rec_loc, inj_sec=inj_sec_split, inj_loc=inj_loc)

    
    plot_widget = PlotWidget(tstop=ctg.tstop)
    plot_widget.add_expr(eval("fir_widget.passive_cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),str(round(float(fir_widget.passive_amp),2)))
    for cell,amp in zip(fir_widget.cells, fir_widget.amps):
        plot_widget.add_expr(eval("cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),str(round(float(amp),2)))

    ctg.add_widget(window_index,column_index,fir_widget)
    ctg.add_widget(window_index,column_index,plot_widget)

    #Column 2
    #column_index = ctg.add_column(window_index)
    #control_widget = ControlMenuWidget()
    #ctg.add_widget(window_index,column_index,control_widget)
    
    text_widget = TextWidget()
    text_widget.set_to_fir_passive(fir_widget)
    ctg.add_widget(window_index, column_index, text_widget)

    ctg.show(auto_run=True,on_complete=text_widget.update_fir_passive)

#https://www.youtube.com/watch?v=MkzeOmkOUHM

@cell.command('vhsegbuild', help="Alturki et al. (2016) V1/2 Automated Segregation Interface, simplify tuning by separating channel activation AUTO BUILD EXPERIMENT")
@click.option('--title',type=click.STRING,default=None)
@click.option('--tstop',type=click.INT,default=1150)
@click.option('--outhoc',type=click.STRING,default="segmented_template.hoc",help="Specify the file you want the modified cell template written to")
@click.option('--outfolder',type=click.STRING,default="./",help="Specify the directory you want the modified cell template and mod files written to (default: _seg)")
@click.option('--outappend',type=click.BOOL,default=False,is_flag=True,help="Append out instead of overwriting (default: False)")
#@click.option('--skipmod',type=click.BOOL,default=False,is_flag=True,help="Skip new mod file generation")
@click.option('--debug',type=click.BOOL,default=False,is_flag=True,help="Print all debug statements")
@click.option('--build',type=click.BOOL,default=False,is_flag=True,help="Build must be run before viewing GUI")
@click.option('--fminpa',type=click.INT,default=0,help="Starting FIR Curve amps (default: 0)")
@click.option('--fmaxpa',type=click.INT,default=1000,help="Ending FIR Curve amps (default: 1000)")
@click.option('--fincrement',type=click.INT,default=100,help="Increment the FIR Curve amps by supplied pA (default: 100)")

@click.pass_context
def cell_vhsegbuild(ctx,title,tstop,outhoc,outfolder,outappend,debug,build,fminpa,fmaxpa,fincrement):
    click.echo(colored.red("EXPERIMENTAL - UNLIKELY TO WORK"))
    if not build:
        click.echo(colored.red("BE SURE TO RUN `vhseg --build` FOR YOUR CELL FIRST!"))
    from .neuron.celltuner import CellTunerGUI, TextWidget, PlotWidget, ControlMenuWidget, SecMenuWidget, FICurveWidget,PointMenuWidget, MultiSecMenuWidget
    from .neuron.celltuner import VoltagePlotWidget, SegregationSelectorWidget, SegregationPassiveWidget, SegregationFIRFitWidget, AutoVInitWidget
    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    if build:
        ctg = CellTunerGUI(hoc_folder,mod_folder,tstop=tstop,print_debug=debug)
    else:
        #template = template + "Seg" #Likely not the best way to do it
        #slm = True if os.path.abspath("./"+outfolder) == os.path.abspath(mod_folder) else False
        ctg = CellTunerGUI("./"+outfolder,"./"+outfolder,tstop=tstop,print_debug=debug)#,skip_load_mod=slm)
        pass
        #ctg.load_template(template,hoc_template_file=outhoc)

    
    if build:  
        # Cell selector
        hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
        if not template:
            template = questionary.select(
            "Select a cell:",
            choices=hoc_templates).ask()

        ctg.load_template(template)
    else:
        # Cell selector
        hoc_templates = ctg.get_templates(hoc_template_file=outhoc)
        if not template:
            template = questionary.select(
            "Select a cell:",
            choices=hoc_templates).ask()
        
        ctg.load_template(template,hoc_template_file=outhoc)
   
    if not title:
        title = template + " - V1/2 Segregation - Interface generated by BMTool (https://github.com/tjbanks/bmtool)"
    #import pdb;pdb.set_trace()
    sec = ctg.root_sec.hname()
    sec_split = sec.split('.')[-1]
    click.echo("Using section " + colored.green(sec_split))
    
    if build:
        # Carry out the segregation method
        fpath = os.path.abspath('./'+outfolder)
        if not os.path.exists(fpath):
            try:
                os.mkdir(fpath)
            except OSError:
                print("Creation of the directory %s failed" % fpath)

        mechs_processed = ctg.seg_mechs(folder=fpath)
        template = ctg.seg_template(os.path.join(fpath,outhoc),mechs_processed)
        click.echo(colored.green("COMPILING MOD FILES"))

        cwd = os.getcwd()

        if os.path.abspath(mod_folder) != cwd: #The mod files will get loaded already
            import glob, shutil

            files = glob.iglob(os.path.join(mod_folder, "*.mod"))
            for file in files:
                if os.path.isfile(file):
                    shutil.copy2(file, fpath)
                
        os.chdir(fpath)
        ret = os.system("nrnivmodl")
        os.chdir(cwd)

        if not ret:
            click.echo(colored.green("COMPILATION COMPLETE"))
        else:
            click.echo(colored.red("nrnivmodl may not have been run, execute nrnivmodl or mknrndll manually in the `")+colored.green(fpath)+colored.red("` folder then press enter..."))
            input()
            click.echo(colored.green("Done... remove the `--build` flag and re-run."))
        return
    else:
        #template = template + "Seg" #Likely not the best way to do it
        #slm = True if os.path.abspath("./"+outfolder) == os.path.abspath(mod_folder) else False
        #ctg = CellTunerGUI("./"+outfolder,"./"+outfolder,tstop=tstop,print_debug=debug,skip_load_mod=slm)
        pass
        #ctg.load_template(template,hoc_template_file=outhoc)

    do_others = False
    if ctg.other_sec:
        do_others = questionary.confirm("Show other sections? (default: No)",default=False).ask()
    selected_segments = []
    if do_others:
        choices = [s.name().split('.')[-1] for s in ctg.other_sec]
        selected_segments = questionary.checkbox(
            'Select other sections (space bar to select):',
            choices=choices).ask()
        
    

    section_selected = "soma"
    #FIR Properties
    min_pa = fminpa
    max_pa = fmaxpa
    increment = fincrement
    tstart = 150
    tdur = 1000
    inj_sec = ctg.root_sec.hname()
    rec_sec = ctg.root_sec.hname()
    inj_loc = "0.5"    
    rec_loc = "0.5"
    rec_sec_split = rec_sec.split('.')[-1]
    inj_sec_split = inj_sec.split('.')[-1]

    if tstop < tstart+tdur:
        print("tstop must be greater than " + str(tstart+tdur) + " due to FIR injection properties")
        print("Exiting")
        return
    # Current Clamp properties
    delay = 150
    dur = 1000
    amp = 0.2

    fir_widget = FICurveWidget(template,i_increment=increment,i_start=min_pa,i_stop=max_pa,tstart=tstart,tdur=tdur,
        record_sec=rec_sec_split, record_loc=rec_loc, inj_sec=inj_sec_split, inj_loc=inj_loc)
    other_cells = fir_widget.cells + [fir_widget.passive_cell]

    for segment in selected_segments:
        window_index = ctg.add_window(title="(" + segment + ")" + title, width=650)
        # Column 1
        column_index = ctg.add_column(window_index)
        plot_widget = PlotWidget(tstop=tstop)
        sec_text = segment+"(.5)"
        cellsec = eval('ctg.template.'+section_selected)
        plot_widget.add_expr(cellsec(0.5)._ref_v,sec_text)

        ctg.add_widget(window_index,column_index,plot_widget)

        widget = ControlMenuWidget()
        ctg.add_widget(window_index,column_index,widget)

        # Column 2
        column_index = ctg.add_column(window_index)
        widget = MultiSecMenuWidget(ctg.root_sec.cell(), other_cells,segment,ctg.mechanism_dict)
        ctg.add_widget(window_index,column_index,widget)
    

    #Window 1
    window_index = ctg.add_window(title=title)

    #Column 1
    column_index = ctg.add_column(window_index)

    plot_widget = PlotWidget(tstop=tstop)
    sec_text = ctg.root_sec.hname().split('.')[-1]+"(.5)"

    plot_widget.add_expr(ctg.root_sec(0.5)._ref_v,sec_text)
    plot_widget.add_expr(eval("fir_widget.passive_cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),"Passive @"+str(int(fir_widget.passive_amp*1e3))+"pA")
    ctg.add_widget(window_index,column_index,plot_widget)

    ctg.add_widget(window_index,column_index,fir_widget)

    

    plot_widget = VoltagePlotWidget(ctg.root_sec.cell(),section="soma")
    plot_widget.add_act_inf(ctg.mechanism_dict)
    ctg.add_widget(window_index,column_index,plot_widget)
    

    #Column 2
    column_index = ctg.add_column(window_index)
    
    #import pdb;pdb.set_trace()
    widget = MultiSecMenuWidget(ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict)

    #for cell,amp in zip(fir_widget.cells, fir_widget.amps):
        #plot_widget.add_expr(eval("cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),str(round(float(amp),2)))
    widget_index = ctg.add_widget(window_index, column_index,widget) 
    #widget = SecMenuWidget(ctg.root_sec,x=float(inj_loc))
    #widget_index = ctg.add_widget(window_index, column_index,widget) 

    #Column 3
    column_index = ctg.add_column(window_index)
    widget = ControlMenuWidget()
    ctg.add_widget(window_index,column_index,widget)
              

    widget = PointMenuWidget(None)
    iclamp = widget.iclamp(ctg.root_sec(float(inj_loc)),dur,amp,delay)
    ctg.register_iclamp(iclamp)
    widget_index = ctg.add_widget(window_index, column_index,widget)

    text_widget = TextWidget()
    text_widget.set_to_fir_passive(fir_widget,print_calc=False,print_fi=False)
    widget_index = ctg.add_widget(window_index, column_index,text_widget)

    vinit_widget = AutoVInitWidget(fir_widget)
    widget_index = ctg.add_widget(window_index, column_index,vinit_widget)

    #Column 4
    column_index = ctg.add_column(window_index)
    
    widget = SegregationSelectorWidget(ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict,all_sec=True)
    ctg.add_widget(window_index,column_index,widget)

    widget = SegregationPassiveWidget(fir_widget,ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict)
    ctg.add_widget(window_index,column_index,widget)

    widget = SegregationFIRFitWidget(fir_widget)
    ctg.add_widget(window_index,column_index,widget)

    ctg.show(auto_run=True,on_complete_fih=text_widget.update_fir_passive,run_count=2)

    return

@cell.command('vhseg', help="Alturki et al. (2016) V1/2 Automated Segregation Interface, simplify tuning by separating channel activation")
@click.option('--title',type=click.STRING,default=None)
@click.option('--tstop',type=click.INT,default=1150)
#@click.option('--outhoc',type=click.STRING,default="segmented_template.hoc",help="Specify the file you want the modified cell template written to")
#@click.option('--outfolder',type=click.STRING,default="./",help="Specify the directory you want the modified cell template and mod files written to (default: _seg)")
#@click.option('--outappend',type=click.BOOL,default=False,is_flag=True,help="Append out instead of overwriting (default: False)")
#@click.option('--skipmod',type=click.BOOL,default=False,is_flag=True,help="Skip new mod file generation")
@click.option('--debug',type=click.BOOL,default=False,is_flag=True,help="Print all debug statements")
@click.option('--fminpa',type=click.INT,default=0,help="Starting FI Curve amps (default: 0)")
@click.option('--fmaxpa',type=click.INT,default=1000,help="Ending FI Curve amps (default: 1000)")
@click.option('--passive-delay',type=click.INT,default=650,help="Wait n ms before determining steadystate value (default: 650)")
@click.option('--fincrement',type=click.INT,default=100,help="Increment the FIR Curve amps by supplied pA (default: 100)")
@click.option('--infvars',type=click.STRING,default=None,help="Specify the inf variables to plot, skips the wizard. (Comma separated, eg: inf_mech,minf_mech2,ninf_mech2)")
@click.option('--segvars',type=click.STRING,default=None,help="Specify the segregation variables to globally set, skips the wizard. (Comma separated, eg: mseg_mech,nseg_mech2)")
@click.option('--eleak',type=click.STRING,default=None,help="Specify the eleak var manually")
@click.option('--gleak',type=click.STRING,default=None,help="Specify the gleak var manually")
@click.option('--othersec',type=click.STRING,default=None,help="Specify other sections that a window should be generated for (Comma separated, eg: dend[0],dend[1])")
@click.option('--clampsec',type=click.STRING,default=None,help="Specify sections that a current clamp should be attached. Root section will always have a clamp. (Comma separated, eg: dend[0],dend[1])")
@click.option('--synsec',type=click.STRING,default=None,help="Specify sections that a synapse should be attached. Exp2Syn default, unless --syntype specified. (Comma separated, eg: dend[0],dend[1])")
@click.option('--syntype',type=click.STRING,default="Exp2Syn",help="Specify the synapse mechanism that will be attached to the cell (Single type)")
@click.option('--synloc',type=click.STRING,default="0.5",help="Specify the synapse location (Default: 0.5)")
@click.pass_context
def cell_vhseg(ctx,title,tstop,debug,fminpa,fmaxpa,passive_delay,fincrement,infvars,segvars,eleak,gleak,othersec,clampsec,synsec,syntype,synloc):
    
    from .neuron.celltuner import CellTunerGUI, TextWidget, PlotWidget, ControlMenuWidget, SecMenuWidget, FICurveWidget,PointMenuWidget, MultiSecMenuWidget
    from .neuron.celltuner import VoltagePlotWidget, SegregationSelectorWidget, SegregationPassiveWidget, SegregationFIRFitWidget, AutoVInitWidget, SingleButtonWidget
    from .util import tk_email_input, send_mail, popupmsg

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]
    prefab_dict = ctx.obj["prefab"]
    prefab_dictvh = None
    
    ctg = CellTunerGUI(hoc_folder,mod_folder,tstop=tstop,print_debug=debug)
    
    
    # Cell selector
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()
    
    ctg.load_template(template)

    original_cell_values = ctg.get_current_cell_values()

    if prefab_dict and prefab_dict.get("cells") and prefab_dict["cells"].get(template) and prefab_dict["cells"][template].get("vhseg"):
        prefab_dictvh = prefab_dict["cells"][template]["vhseg"]

    if not title:
        title = template + " - V1/2 - Interface generated by BMTool (https://github.com/tjbanks/bmtool)"
    #import pdb;pdb.set_trace()
    sec = ctg.root_sec.hname()
    sec_split = sec.split('.')[-1]
    click.echo("Using section " + colored.green(sec_split))
    
    selected_segments = []

    if prefab_dictvh and prefab_dictvh.get("othersec"):
        selected_segments = prefab_dictvh["othersec"]

    else:
        do_others = False
        if ctg.other_sec:

            if othersec:
                do_others = True
            else:
                do_others = questionary.confirm("Show other sections? (default: No)",default=False).ask()

        
        if do_others:
            if othersec:
                selected_segments = othersec.split(",")
            else:
                choices = [s.name().split('.')[-1] for s in ctg.other_sec]
                selected_segments = questionary.checkbox(
                    'Select other sections (space bar to select):',
                    choices=choices).ask()
        
    section_selected = sec_split
    
    #cellsec = getattr(ctg.template,section_selected)
    cellsec = eval('ctg.template.'+section_selected)

    mechs = [mech.name() for mech in cellsec(0.5) if not mech.name().endswith("_ion")]
    ions = [mech.name() for mech in cellsec(0.5) if mech.name().endswith("_ion")]
    cellmechvars = []
    
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

    #Puts the best matches at the top
    inf_choices = [s for s in cellmechvars if "inf" in s.lower()] + [s for s in cellmechvars if not "inf" in s.lower()]
    seg_choices = [s for s in cellmechvars if "seg" in s.lower()] + [s for s in cellmechvars if not "seg" in s.lower()]
    from neuron import h
    globalvars = dir(h)
    list_global_text = "** More ** (Shows unselected global [MANY]) (current selection retained)"
    
    if prefab_dictvh and prefab_dictvh.get("infvars"):
        infvars = prefab_dictvh["infvars"]
        
    else:
        if infvars:
            infvars = infvars.split(",")
        else:
            question_text = 'Select inf variables to plot (with respect to membrane potential) (space bar to select): '
            inf_choices = inf_choices + [list_global_text]
            infvars = questionary.checkbox(
                question_text,
                choices=inf_choices).ask()
            if list_global_text in infvars:
                infvars.remove(list_global_text)
                global_choices =  [c for c in globalvars if c not in infvars]
                globalinfvars = questionary.checkbox(
                question_text,
                choices=global_choices).ask()
                infvars = infvars + globalinfvars

    if prefab_dictvh and prefab_dictvh.get("segvars"):
        segvars = prefab_dictvh["segvars"]
        
    else:
        if segvars:
            segvars = segvars.split(",")
        else:
            question_text = 'Select segregation variables [OR VARIABLES YOU WANT TO CHANGE ON ALL SEGMENTS at the same time] (space bar to select):'
            seg_choices = seg_choices + [list_global_text]
            segvars = questionary.checkbox(
                question_text,
                choices=seg_choices).ask()
            if list_global_text in segvars:
                segvars.remove(list_global_text)
                global_choices =  [c for c in globalvars if c not in segvars]
                globalsegvars = questionary.checkbox(
                question_text,
                choices=global_choices).ask()
                segvars = segvars + globalsegvars

    #clampsec,synsec,syntype
    clampme = []
    synme = []
    if selected_segments:
        if prefab_dictvh and prefab_dictvh.get("clampsec"):
            clampme = prefab_dictvh["clampsec"]
        else:
            if not clampsec:
                do_clamps = questionary.confirm("Attach a current clamp to other section? (root section ["+section_selected+"] automatically attached) (default: No)",default=False).ask()        

                if do_clamps:
                    clampme = questionary.checkbox(
                    'Select other sections to attach a current clamp to (space bar to select):',
                    choices=selected_segments).ask()
            else:
                clampme = clampsec.split(",")

        if prefab_dictvh and prefab_dictvh.get("synsec"):
            clampme = prefab_dictvh["synsec"]
        else:
            if not synsec:
                do_syns = questionary.confirm("Attach a synapse to other section? (root section ["+section_selected+"] automatically attached) (default: No)",default=False).ask()        

                if do_syns:
                    synme = questionary.checkbox(
                    'Select sections to attach an artifical synapse (space bar to select):',
                    choices=selected_segments).ask()
            else:
                synme = synsec.split(",")
        
    if prefab_dictvh and prefab_dictvh.get("eleak"):
        eleak = prefab_dictvh["eleak"]

    if prefab_dictvh and prefab_dictvh.get("gleak"):
        gleak = prefab_dictvh["gleak"]

    if prefab_dictvh and prefab_dictvh.get("syntype"):
        syntype = prefab_dictvh["syntype"]

    if prefab_dictvh and prefab_dictvh.get("synloc"):
        synloc = prefab_dictvh["synloc"]
    
    #FIR Properties
    min_pa = fminpa
    max_pa = fmaxpa
    increment = fincrement
    tstart = 150
    tdur = 1000
    inj_sec = ctg.root_sec.hname()
    rec_sec = ctg.root_sec.hname()
    inj_loc = "0.5"    
    rec_loc = "0.5"
    rec_sec_split = rec_sec.split('.')[-1]
    inj_sec_split = inj_sec.split('.')[-1]

    if tstop < tstart+tdur:
        print("tstop must be greater than " + str(tstart+tdur) + " due to FIR injection properties")
        print("Exiting")
        return
    # Current Clamp properties
    delay = 150
    dur = 1000
    amp = 0.2

    fir_widget = FICurveWidget(template,i_increment=increment,i_start=min_pa,i_stop=max_pa,tstart=tstart,tdur=tdur,passive_delay=passive_delay,
        record_sec=rec_sec_split, record_loc=rec_loc, inj_sec=inj_sec_split, inj_loc=inj_loc)
    other_cells = fir_widget.cells + [fir_widget.passive_cell]

    for segment in selected_segments:
        width = 650
        if segment in synme:
            width = 850
        window_index = ctg.add_window(title="(" + segment + ")" + title, width=width)
        # Column 1
        column_index = ctg.add_column(window_index)
        plot_widget = PlotWidget(tstop=tstop)
        sec_text = segment+"(.5)"
        cellsec = eval('ctg.template.'+segment)
        plot_widget.add_expr(cellsec(0.5)._ref_v,sec_text,hoc_text="%s."+segment+".v(0.5)",hoc_text_obj=ctg.template)

        ctg.add_widget(window_index,column_index,plot_widget)

        if segment in clampme:
            widget = PointMenuWidget(None)
            iclamp = widget.iclamp(cellsec(float(inj_loc)),0,0,0)
            ctg.register_iclamp(iclamp)
            widget_index = ctg.add_widget(window_index, column_index,widget)
            
        
        widget = ControlMenuWidget()
        ctg.add_widget(window_index,column_index,widget)

        # Column 2
        column_index = ctg.add_column(window_index)
        widget = MultiSecMenuWidget(ctg.root_sec.cell(), other_cells,segment,ctg.mechanism_dict)
        ctg.add_widget(window_index,column_index,widget)

        if segment in synme:
            column_index = ctg.add_column(window_index)

            interval = 50 #int(questionary.text("Enter default netstim interval (ms (mean) time between spikes): ",default="50").ask())
            number = 0 #int(questionary.text("Enter default netstim number of events ((average) number of spikes): ",default="10").ask())
            start = 0 #int(questionary.text("Enter default netstim start (ms (most likely) start time of first spike): ",default="0").ask())
            noise = 0 #float(questionary.text("Enter default netstim noise (range 0 to 1. Fractional randomness.): ",default="0").ask())
            weight = 1 #float(questionary.text("Enter default netcon weight (range 0 to 1. Default: 1): ",default="1").ask())
            
            widget = PointMenuWidget(None)
            widget_extra = PointMenuWidget(None)
            synapse = widget_extra.synapse(eval("ctg.template."+segment),synloc,syntype)
            netstim,netcon = widget.netstim(interval,number,start,noise,target=synapse,weight=weight)
            ctg.register_netstim(netstim)
            ctg.register_netcon(netcon)
            ctg.register_synapse(synapse)

            widget_index_extra = ctg.add_widget(window_index, column_index, widget_extra)
            widget_index = ctg.add_widget(window_index, column_index, widget)
    

    #Window 1
    window_index = ctg.add_window(title=title)

    #Column 1
    column_index = ctg.add_column(window_index)

    plot_widget = PlotWidget(tstop=tstop)
    sec_text = ctg.root_sec.hname().split('.')[-1]+"(.5)"

    plot_widget.add_expr(ctg.root_sec(0.5)._ref_v,sec_text,hoc_text="%s."+section_selected+".v(0.5)",hoc_text_obj=ctg.template)
    plot_widget.add_expr(eval("fir_widget.passive_cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),"Passive @"+str(int(fir_widget.passive_amp*1e3))+"pA")
    ctg.add_widget(window_index,column_index,plot_widget)

    ctg.add_widget(window_index,column_index,fir_widget)

    plot_widget = VoltagePlotWidget(ctg.root_sec.cell(),section=section_selected)
    plot_widget.add_act_inf(variables=infvars)#ctg.mechanism_dict)
    ctg.add_widget(window_index,column_index,plot_widget)
    

    #Column 2
    column_index = ctg.add_column(window_index)
    
    #import pdb;pdb.set_trace()
    widget = MultiSecMenuWidget(ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict)

    #for cell,amp in zip(fir_widget.cells, fir_widget.amps):
        #plot_widget.add_expr(eval("cell." + rec_sec_split + "("+ rec_loc+")._ref_v"),str(round(float(amp),2)))
    widget_index = ctg.add_widget(window_index, column_index,widget) 
    #widget = SecMenuWidget(ctg.root_sec,x=float(inj_loc))
    #widget_index = ctg.add_widget(window_index, column_index,widget) 

    interval = 50 #int(questionary.text("Enter default netstim interval (ms (mean) time between spikes): ",default="50").ask())
    number = 0 #int(questionary.text("Enter default netstim number of events ((average) number of spikes): ",default="10").ask())
    start = 0 #int(questionary.text("Enter default netstim start (ms (most likely) start time of first spike): ",default="0").ask())
    noise = 0 #float(questionary.text("Enter default netstim noise (range 0 to 1. Fractional randomness.): ",default="0").ask())
    weight = 1 #float(questionary.text("Enter default netcon weight (range 0 to 1. Default: 1): ",default="1").ask())
    
    widget = PointMenuWidget(None)
    widget_extra = PointMenuWidget(None)
    synapse = widget_extra.synapse(eval("ctg.template."+section_selected),synloc,syntype)
    netstim,netcon = widget.netstim(interval,number,start,noise,target=synapse,weight=weight)
    ctg.register_netstim(netstim)
    ctg.register_netcon(netcon)
    ctg.register_synapse(synapse)

    widget_index_extra = ctg.add_widget(window_index, column_index, widget_extra)
    widget_index = ctg.add_widget(window_index, column_index, widget)

    #Column 3
    column_index = ctg.add_column(window_index)
    widget = ControlMenuWidget()
    ctg.add_widget(window_index,column_index,widget)
              

    widget = PointMenuWidget(None)
    iclamp = widget.iclamp(ctg.root_sec(float(inj_loc)),dur,amp,delay)
    ctg.register_iclamp(iclamp)
    widget_index = ctg.add_widget(window_index, column_index,widget)

    text_widget = TextWidget()
    text_widget.set_to_fir_passive(fir_widget,print_calc=False,print_fi=False)
    widget_index = ctg.add_widget(window_index, column_index,text_widget)

    vinit_widget = AutoVInitWidget(fir_widget)
    widget_index = ctg.add_widget(window_index, column_index,vinit_widget)

    

    def email_func():
        addr = tk_email_input()
        if addr is not None:
            usernotes = tk_email_input(title="Usernotes",prompt="Enter any notes you want to include with the email. Click cancel for no notes, and send.")
            experiment_hoc = "run_experiment.hoc"
            changed_cell_values = ctg.get_current_cell_values(change_dict=original_cell_values)
            ctg.write_hoc(experiment_hoc,mechanism_dir="./",template_dir="./", val_set={ctg.template:changed_cell_values})
            report_file = write_report(exp_hoc=experiment_hoc)
            template_zip = template+".zip"

            dirpath = tempfile.mkdtemp()

            for file in glob.glob(os.path.join(hoc_folder,"*.hoc")):
                shutil.copy2(file,dirpath)
            for file in glob.glob(os.path.join(mod_folder,"*.mod")):
                shutil.copy2(file,dirpath)
            
            mod_mod_folder = os.path.join(mod_folder,"modfiles")
            if os.path.exists(mod_mod_folder):
                for file in glob.glob(os.path.join(mod_mod_folder,"*.mod")):
                    shutil.copy2(file,dirpath)
            
            shutil.copy2(experiment_hoc,dirpath)
            shutil.copy2(report_file,dirpath)
            shutil.make_archive(template,"zip",dirpath)
            shutil.rmtree(dirpath)

            if usernotes is not None:
                usernotes = "User notes: " + usernotes
            else:
                usernotes = ""
            
            message_subject = "Your \"" + template + "\" Model from Cyneuro.org"
            message_text = """
Dear BMTool User,

Thank you for using bmtool. Your "{}" model cell is enclosed in the attached zip file. To view your cell:

1. You'll need to have NEURON installed (https://neuron.yale.edu/)
2. Unzip the {} file.
3. Compile the .mod files using `mknrndll` (Windows) or `nrnivmodl` (Mac/Linux), included with NEURON. 
4. Finally, double click the `{}` file to view the user interface.

{}

TB
University of Missouri

Cyneuro.org
BMTool
https://github.com/tjbanks/bmtool

            """.format(template,template_zip, experiment_hoc,usernotes)
        
            send_mail("BMTool@cyneuro.org",[addr],message_subject,message_text,files=[os.path.abspath(template_zip)])

            os.remove(experiment_hoc)
            os.remove(report_file)
            os.remove(template_zip)


    def save_func():
        experiment_hoc = "run_experiment.hoc"
        write_report(experiment_hoc)
        changed_cell_values = ctg.get_current_cell_values(change_dict=original_cell_values)
        ctg.write_hoc(experiment_hoc,val_set={ctg.template:changed_cell_values})
        popupmsg("Saved to " + os.path.abspath("./"+experiment_hoc))

    emailer_widget = SingleButtonWidget("Email this model",email_func)
    widget_index = ctg.add_widget(window_index, column_index,emailer_widget)

    save_widget = SingleButtonWidget("Save Hoc GUI with parameters",save_func)
    widget_index = ctg.add_widget(window_index, column_index,save_widget)

    #Column 4
    column_index = ctg.add_column(window_index)
    
    widget = SegregationSelectorWidget(ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict,all_sec=True,variables=segvars)
    ctg.add_widget(window_index,column_index,widget)

    segpassivewidget = SegregationPassiveWidget(fir_widget,ctg.root_sec.cell(), other_cells,section_selected,ctg.mechanism_dict,gleak_var=gleak,eleak_var=eleak)
    ctg.add_widget(window_index,column_index,segpassivewidget)

    widget = SegregationFIRFitWidget(fir_widget)
    ctg.add_widget(window_index,column_index,widget)


    def write_report(exp_hoc=""):
        report_file = "Report and Instructions.txt"

        uvrest = str(round(fir_widget.v_rest,2))
        if segpassivewidget.v_rest:
            uvrest = str(round(segpassivewidget.v_rest.val,2))
        
        urin = str(round(fir_widget.r_in/1e6,2))
        if segpassivewidget.r_in:
            urin = str(round(segpassivewidget.r_in.val,2))

        utau = str(round(fir_widget.tau,2))
        if segpassivewidget.tau:
            utau = str(round(segpassivewidget.tau.val,2))

        vrest = str(round(fir_widget.v_rest,2))
        rin = str(round(fir_widget.r_in/1e6,2))
        tau = str(round(fir_widget.tau,2))

        spikes = [str(round(i,0)) for i in fir_widget.plenvec]
        amps = fir_widget.amps
        ficurve = " | ".join("["+str(round(a*1e3,0))+" pA]:"+n for a,n in zip(amps,spikes))

        report_text = """
Report generated by BMTool (https://github.com/tjbanks/bmtool)

Thank you for using bmtool to model your "{}" cell. To view your cell:

1. You'll need to have NEURON installed (https://neuron.yale.edu/)
2. Compile the .mod files using `mknrndll` (Windows) or `nrnivmodl` (Mac/Linux), included with NEURON. 
3. Finally, double click the `{}` file to view the user interface.

Cell template: {}

=== User supplied info (can be prefilled) ===

Passive Properties:
V_rest = {} (mV)
R_in = {} (MOhms)
tau = {} (ms)

=== Cell info ===

Passive Properties:
V_rest = {} (mV)
R_in = {} (MOhms)
tau = {} (ms)

FI Curve: 
{}

Cell values:

""".format(template,exp_hoc,template,uvrest,urin,utau,vrest,rin,tau,ficurve)
        changed_cell_values = ctg.get_current_cell_values()

        with open(report_file,"w+") as f:
            f.write(report_text)
            for sec,vals in changed_cell_values.items():
                f.write(sec+"\n")
                for key,val in vals.items():
                    f.write("\t" + key + " = " + str(round(val,6)) + "\n")

        return report_file


    ctg.show(auto_run=True,on_complete_fih=text_widget.update_fir_passive,run_count=2)

    

    return

if __name__ == "__main__":
    cli()