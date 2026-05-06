from .q_learner import QLearner
from .coma_learner import COMALearner
from .ppo_learner import PPOLearner
from .t_learner import TLearner
from .nq_learner import NQLearner
REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["t_learner"] = TLearner
REGISTRY["nq_learner"] = NQLearner
