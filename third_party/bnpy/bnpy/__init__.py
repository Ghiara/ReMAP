''' bnpy module __init__ file
'''
import os
import sys
import psutil

# Allow legacy imports like `import bnpy` to resolve to this vendored package.
sys.modules.setdefault('bnpy', sys.modules[__name__])

# Configure PYTHONPATH before importing any bnpy modules
ROOT_PATH = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-2])

DATASET_PATH = os.path.join(ROOT_PATH, 'bnpy/datasets/')

from third_party.CARE.bnpy import data
from third_party.CARE.bnpy import suffstats
from third_party.CARE.bnpy import util

from third_party.CARE.bnpy import allocmodel
from third_party.CARE.bnpy import obsmodel
from third_party.CARE.bnpy.HModel import HModel

from third_party.CARE.bnpy import ioutil
from third_party.CARE.bnpy import init
from third_party.CARE.bnpy import learnalg
from third_party.CARE.bnpy import birthmove
from third_party.CARE.bnpy import mergemove
from third_party.CARE.bnpy import deletemove

from third_party.CARE.bnpy import callbacks

from third_party.CARE.bnpy import Runner

# Convenient aliases to existing functions
run = Runner.run
load_model_at_lap = ioutil.ModelReader.load_model_at_lap
save_model = ioutil.ModelWriter.save_model
make_initialized_model = Runner.make_initialized_model


__all__ = ['run', 'learnalg', 'allocmodel', 'obsmodel', 'suffstats',
           'HModel', 'init', 'util', 'ioutil']

# Optional viz package for plotting
try:
    from matplotlib import pylab
    from third_party.CARE.bnpy import viz
    __all__.append('viz')
except ImportError:
    print("Error importing matplotlib. Plotting disabled.")
    print("Fix by making sure this produces a figure window on your system")
    print(" >>> from matplotlib import pylab; pylab.figure(); pylab.show();")
