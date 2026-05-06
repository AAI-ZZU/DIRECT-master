REGISTRY = {}

from .basic_controller import BasicMAC
from .intent_controller import TMAC
from .n_controller import NMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["t_mac"] = TMAC
REGISTRY["n_mac"] = NMAC
