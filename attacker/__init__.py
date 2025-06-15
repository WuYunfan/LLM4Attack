import sys
from attacker.heuristic import RandomAttacker
from attacker.heuristic import BandwagonAttacker
from attacker.basic_attacker import BasicAttacker
from attacker.dpa2dl_attacker import DPA2DLAttacker
from attacker.revadv_attacker import RevAdvAttacker
from attacker.pga_attacker import PGAAttacker
from attacker.feo_attacker import FEOAttacker
from attacker.rapu_r_attack import RAPURAttacker
from attacker.legup_attacker import LegUPAttacker


def get_attacker(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    config['device'] = dataset.device
    attacker = getattr(sys.modules['attacker'], config['name'])
    attacker = attacker(config)
    return attacker


__all__ = ['get_attacker']
