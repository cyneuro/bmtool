from os import system, name
import click
import logging
import os
import questionary

from clint.textui import puts, colored, indent

from .util import load_config

@click.group('util')
@click.option('-c', '--config', type=click.Path(), default='./simulation_config.json', help='Configuration file to use, default: "simulation_config.json"')
@click.pass_context
def cli(ctx, config):
    config_path = os.path.abspath(os.path.expanduser(config)).replace("\\","/")

    ctx.obj["config"] = config_path

    if not os.path.exists(config_path):
        click.echo(colored.red("Config file not found: " + config))

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
@click.pass_context
def cell(ctx,hoc_folder,mod_folder,template,hoc):
  
    if not check_neuron_installed():
        return   

    hoc_template_file = None
    
    if hoc:
        if not hoc_folder:
            hoc_folder = './'
        if not mod_folder:
            mod_folder = './'
        hoc_template_file = hoc
    elif not hoc_folder or not mod_folder:
        cfg = load_config(ctx.obj['config'])
        if not hoc_folder:
            hoc_folder = cfg['components']['templates_dir']
        if not mod_folder:
            mod_folder = cfg['components']['mechanisms_dir']  

    ctx.obj["hoc_folder"] = hoc_folder
    ctx.obj["mod_folder"] = mod_folder
    ctx.obj["hoc_template_file"] = hoc_template_file
    ctx.obj["cell_template"] = template
    
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
            print("Enter the name of the file you wish to write: (eg: cellgui.hoc)")
            filename = input()
            self.ctg.write_hoc(filename)
            print("Done. Press enter to continue...")
            input()
            return self

        def finished():
            return None

        self.register("New Window", new_window)
        self.register("Display Current Setup", print_ctx)
        self.register("Write to HOC executable", write_to_hoc)
        self.register("Done", finished ,is_exit=True)
 
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
        self.title = "Window " + str(self.parent.window_index) + " Column " + str(self.column_index)
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

        def finished():
            return self.parent

        self.register("Add Widget", new_widget)
        self.register("Add Plot Widget", new_plot_widget)
        self.register("Add NEURON Control Menu Widget", new_controlmenu_widget)
        self.register("Finish Column", finished ,is_exit=True)

class PlotWidgetBuilder(Builder):
    def __init__(self, parent, ctg):
        super(PlotWidgetBuilder, self).__init__()

        from .neuron.celltuner import PlotWidget

        self.parent = parent
        self.ctg = ctg
        self.widget = PlotWidget()
        self.widget_index = ctg.add_widget(self.parent.parent.window_index, self.parent.column_index,self.widget)
        self.title = "Window " + str(self.parent.parent.window_index + 1) + " Column " + \
            str(self.parent.column_index + 1) + " Widget " + str(self.widget_index) + " (Plot Widget)"
        self.register_all()
        return

    def register_all(self):
            
        def new_expression():
            obj_options = ["Cell","Quick - Template Cell Membrane Voltage (0.5)"]
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
                
                import pdb;pdb.set_trace()
                variable_options = []
                variable_selected = questionary.select(
                "Select the Variable",
                choices=variable_options).ask()

                section_location = questionary.text("Enter recording location (default:0.5): ").ask(default="0.5")

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
@click.pass_context
def cell_tune(ctx,easy,builder,write_hoc,hide,title,tstop):#, title, populations, group_by, save_file):
    
    from .neuron.celltuner import CellTunerGUI, PlotWidget, ControlMenuWidget, SecMenuWidget

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    ctg = CellTunerGUI(hoc_folder,mod_folder,title=title)
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    
    # Cell selector
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()

    ctg.load_template(template)
    if not title:
        title = template + " - Cell Configurator - Interface generated by BMTools (https://github.com/tjbanks/bmtools)"

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

@cell.command('fir', help="Creates a NEURON GUI window with FI curve and passive properties")
#@click.option('--easy', type=click.BOOL, default=None, is_flag=True, help="override the default simulation config mod file location")
#@click.option('--write-hoc', type=click.STRING, default=None, help="write a standalone hoc file for your GUI, supply filename")
#@click.option('--hide', type=click.BOOL, default=False, is_flag=True, help="hide the interface that shows automatically after building the GUI")
@click.option('--title',type=click.STRING,default=None)
@click.option('--min-pa',type=click.INT,default=0,help="Min pA for injection")
@click.option('--max-pa',type=click.INT,default=1000,help="Max pA for injection")
@click.option('--increment',type=click.FLOAT,default=100,help="Increment the injection by [i] pA")
@click.option('--tstart',type=click.INT,default=50, help="Injection start time")
@click.option('--tdur',type=click.INT,default=1000,help="Duration of injection default:1000ms")
@click.option('--advanced',type=click.BOOL,default=False,is_flag=True,help="Interactive dialog to select injection and recording points")
@click.pass_context
def cell_fir(ctx,title,min_pa,max_pa,increment,tstart,tdur,advanced):#, title, populations, group_by, save_file):
    
    from .neuron.celltuner import CellTunerGUI, TextWidget, PlotWidget, ControlMenuWidget, SecMenuWidget, FICurveWidget

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    tstop = tstart+tdur

    ctg = CellTunerGUI(hoc_folder,mod_folder,tstop=tstop)
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    
    # Cell selector
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()

    ctg.load_template(template)
    if not title:
        title = template + " - Cell FI Curve - Interface generated by BMTools (https://github.com/tjbanks/bmtools)"
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
    fir_widget = FICurveWidget(template,i_increment=increment,i_start=min_pa,i_stop=max_pa,tstart=tstart,tdur=tdur,
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

    text_widget = TextWidget(label='PASSIVE PROPERTIES:\n')
    text_widget.add_text("V_rest: ")
    text_widget.add_text("R_in: ")
    text_widget.add_text("Tau: ")
    text_widget.add_text("")
    text_widget.add_text("V_rest Calculation: ")
    text_widget.add_text("R_in Calculation: ")
    text_widget.add_text("")
    text_widget.add_text("Tau Calculation: ")
    text_widget.add_text("")
    text_widget.add_text("")
    text_widget.add_text("FICurve ([nA]:Hz): ")
    text_widget.add_text("")
    ctg.add_widget(window_index, column_index, text_widget)

    def set_text():
        nonlocal fir_widget
        v_rest_calc = "V_rest Calculation: Taken at time " + str(fir_widget.v_rest_time) + "(ms) on negative injection cell"
        rin_calc = "R_in Calculation: [(dV/dI)] = (v_start-v_final)/(i_start-i_final) = " + str(round(fir_widget.v_rest,2)) + \
                "-(" + str(round(fir_widget.passive_v_final,2))+"))/(0-(" + str(fir_widget.passive_amp) + "))" + \
                " = (" + str(round(fir_widget.v_rest-fir_widget.passive_v_final,2))+ " (mV) /" + str(0-fir_widget.passive_amp) + " (nA))" + \
                " = ("+ str(round(((fir_widget.v_rest-fir_widget.passive_v_final)/(0-fir_widget.passive_amp)),2)) + " (MOhms))"
        tau_calc = "Tau Calculation: [(s) until 63.2% change in mV] = " + \
                "(mV at inj_start_time (" + str(fir_widget.tstart) + ")) - ((mV at inj_time  - mV at inj_final (" + str(fir_widget.tstart+fir_widget.passive_delay) + ")) * 0.632) = " + \
                "(" + str(round(fir_widget.v_rest,2)) + ") - (" + str(round(fir_widget.v_rest,2)) +"-" +str(round(fir_widget.passive_v_final,2))+")*0.632 = " + str(round(fir_widget.v_rest - ((fir_widget.v_rest - fir_widget.passive_v_final)*0.632),2))
        tau_calc2 = "Time where mV == " + str(round(fir_widget.v_t_const,2)) + " = " + str(fir_widget.tstart+fir_widget.tau*1000) + "(ms) | (" + str(fir_widget.tstart+fir_widget.tau*1000) + " - v_start_time (" + str(fir_widget.tstart) +"))/1000 = " + str(round(fir_widget.tau,4))
        text_widget.set_text(0,"V_rest: " + str(round(fir_widget.v_rest,2)) + " (mV) ")
        text_widget.set_text(1,"R_in: " + str(round(fir_widget.r_in,2)) + " (MOhms) ")
        text_widget.set_text(2,"Tau: " + str(round(fir_widget.tau,4)) + " (s) ")

        text_widget.set_text(4,v_rest_calc)
        text_widget.set_text(5,rin_calc)
        text_widget.set_text(6,"v_start time: " + str(fir_widget.v_rest_time) + "(ms) | v_final time: " + str(fir_widget.v_final_time) + "(ms)")
        text_widget.set_text(7,tau_calc)
        text_widget.set_text(8,tau_calc2)

        spikes = [str(round(i,0)) for i in fir_widget.plenvec]
        amps = fir_widget.amps
        text_widget.set_text(11," | ".join("["+str(round(a,2))+"]:"+n for a,n in zip(amps,spikes)))
        
        return

    ctg.show(auto_run=True,on_complete=set_text)

#https://www.youtube.com/watch?v=MkzeOmkOUHM

@cell.command('vhseg', help="Alturki et al. (2016) V1/2 Automated Segregation Interface, simplify tuning by separating channel activation")
@click.option('--title',type=click.STRING,default=None)
@click.option('--tstop',type=click.INT,default=250)
@click.pass_context
def cell_vhseg(ctx,title,tstop):
    click.echo(colored.red("EXPERIMENTAL!"))
    from .neuron.celltuner import CellTunerGUI, TextWidget, PlotWidget, ControlMenuWidget, SecMenuWidget

    hoc_folder = ctx.obj["hoc_folder"]
    mod_folder = ctx.obj["mod_folder"]
    hoc_template_file = ctx.obj["hoc_template_file"]
    template = ctx.obj["cell_template"]

    ctg = CellTunerGUI(hoc_folder,mod_folder,tstop=tstop)
    hoc_templates = ctg.get_templates(hoc_template_file=hoc_template_file)
    
    # Cell selector
    if not template:
        template = questionary.select(
        "Select a cell:",
        choices=hoc_templates).ask()

    ctg.load_template(template)
    if not title:
        title = template + " - V1/2 Segregation - Interface generated by BMTools (https://github.com/tjbanks/bmtools)"
        #ctg.set_title(title)

    sec = ctg.root_sec.hname()
    sec_split = sec.split('.')[-1]

    click.echo("Using section " + colored.green(sec_split))
    mechs = [s for s in ctg.root_sec() if not s.is_ion()]
    #must install pip install textX==1.6.1
    from xml.dom.minidom import parseString
    from pynmodl.unparser import Unparser
    from pynmodl.lems import mod2lems
    mod_files = []

    mods_path = os.path.join(mod_folder,'modfiles')
    if not os.path.exists(mods_path):
        mods_path = os.path.join(mod_folder)
    import pdb;pdb.set_trace()

    with open(os.path.join(mod_folder,'modfiles/borgkaCA3.mod')) as f:
        mod_file = f.read()
        print(parseString(mod2lems(mod_file)).toprettyxml())
        mod_files.append(Unparser().compile(mod_file))
        print(mod_files[-1])
    import pdb;pdb.set_trace()

    return
    
if __name__ == "__main__":
    cli()