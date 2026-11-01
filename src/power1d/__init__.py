'''
power1d: numerical power estimates for one-dimensional continua

Pataky TC (2017) Power1D: A Python toolbox for numerical
power estimates in experiments involving one-dimensional
continua. PeerJ Computer Science 3: e125.
DOI: 10.7717/peerj-cs.125
https://peerj.com/articles/cs-125/


Copyright (C) 2024  Todd Pataky
'''


__version__ = '0.1.16'  # 2026-11-01


__all__ = ['geom', 'models', 'noise', 'roi', 'stats', 'get_config']



from . import data
from . import geom
from . import io
from . import models
from . import noise
from . import prob
from . import random
from . import roi
from . import stats
from . _support import get_config


DataSample          = models.DataSample
Experiment          = models.Experiment
ExperimentSimulator = models.ExperimentSimulator
RegionOfInterest    = roi.RegionOfInterest





