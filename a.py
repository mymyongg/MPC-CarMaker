import matlab.engine
import time

# Run 'cmenv.m'
# Open 'generic.mdl'
# Open CarMaker GUI
# Load TestRun
eng.set_param('MyModel', 'SimulationCommand', 'start', nargout=0)
eng.set_param('MyModel', 'SimulationCommand', 'pause', nargout=0)
eng.set_param('MyModel', 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)

# Add 'To Workspace' block in Simulink
eng.eval('tout')
eng.eval('gas')

eng.workspace['output']
eng.eval("out.output")
eng.set_param('{}/u'.format(self.modelName),'value',str(u),nargout=0)
eng.set_param('MyModel/CarMaker/VehicleControl/CreateBus VhclCtrl/gas_input', 'value', '0.50', nargout=0)

eng = matlab.engine.start_matlab('-desktop')
eng.eval("cmenv", nargout=0)
eng.eval("load_system(MyModel)", nargout=0)
eng.set_param('MyModel', 'SimulationCommand', 'start', nargout=0)
time.sleep(20)