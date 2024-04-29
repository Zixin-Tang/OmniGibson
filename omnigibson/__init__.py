import builtins
import logging
import os
import shutil
import signal
import tempfile

from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.envs import Environment

# TODO: Need to fix somehow -- omnigibson gets imported first BEFORE we can actually modify the macros
from omnigibson.macros import gm
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.sensors import ALL_SENSOR_MODALITIES
from omnigibson.simulator import launch_simulator as launch
from omnigibson.tasks import REGISTERED_TASKS

# Create logger
logging.basicConfig(format='[%(levelname)s] [%(name)s] %(message)s')
log = logging.getLogger(__name__)

builtins.ISAAC_LAUNCHED_FROM_JUPYTER = (
    os.getenv("ISAAC_JUPYTER_KERNEL") is not None
)  # We set this in the kernel.json file

# Always enable nest_asyncio because MaterialPrim calls asyncio.run()
import nest_asyncio

nest_asyncio.apply()

__version__ = "1.0.0"

log.setLevel(logging.DEBUG if gm.DEBUG else logging.INFO)

root_path = os.path.dirname(os.path.realpath(__file__))

# Store paths to example configs
example_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# Initialize global variables
app = None  # (this is a singleton so it's okay that it's global)
sim = None  # (this is a singleton so it's okay that it's global)


# Create and expose a temporary directory for any use cases. It will get destroyed upon omni
# shutdown by the shutdown function.
tempdir = tempfile.mkdtemp()

def cleanup(*args, **kwargs):
    # TODO: Currently tempfile removal will fail due to CopyPrim command (for example, GranularSystem in dicing_apple example.)
    try:
        shutil.rmtree(tempdir)
    except PermissionError:
        log.info("Permission error when removing temp files. Ignoring")
    from omnigibson.simulator import logo_small
    log.info(f"{'-' * 10} Shutting Down {logo_small()} {'-' * 10}")

def shutdown(due_to_signal=False):
    if app is not None:
        # If Isaac is running, we do the cleanup in its shutdown callback to avoid open handles.
        # TODO: Automated cleanup in callback doesn't work for some reason. Need to investigate.
        # Manually call cleanup for now.
        cleanup()
        app.close()
    else:
        # Otherwise, we do the cleanup here.
        cleanup()
        
        # If we're not shutting down due to a signal, we need to manually exit
        if not due_to_signal:
            exit(0)
        
def shutdown_handler(*args, **kwargs):
    shutdown(due_to_signal=True)
    return signal.default_int_handler(*args, **kwargs)
    
# Something somewhere disables the default SIGINT handler, so we need to re-enable it
signal.signal(signal.SIGINT, shutdown_handler)
