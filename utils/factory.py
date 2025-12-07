from models.finetune import Finetune
from models.PRL import PRL
from models.PRL2 import PRL2
from models.PRL_MU import PRL_MU
from models.BASELINE import BASELINE
from models.BASELINE_MU import BASELINE_MU
from models.BASELINE_replay import BASELINE_replay
from models.BASELINE_replay2 import BASELINE_replay2
from models.BASELINE_replay3 import BASELINE_replay3
from models.BASELINE_replay4 import BASELINE_replay4
from models.BASELINE_replay5 import BASELINE_replay5
from models.BASELINE_replay6 import BASELINE_replay6
from models.BASELINE_DI import BASELINE_DI


def get_model(model_name, args):

    # model(手法) の名前
    name = model_name.lower()
    
    # model名毎に異なる learner を返す
    if name == "finetune":
        return Finetune(args)
    elif name == "prl":
        return PRL(args)
    elif name == "prl-mu":
        return PRL_MU(args)
    elif name == "baseline":
        return BASELINE(args)
    elif name == "baseline_mu":
        return BASELINE_MU(args)
    elif name == "prl2":
        return PRL2(args)
    elif name == "baseline-replay":
        return BASELINE_replay(args)
    elif name == "baseline-replay2":
        return BASELINE_replay2(args)
    elif name == "baseline-replay3":
        return BASELINE_replay3(args)
    elif name == "baseline-replay4":
        return BASELINE_replay4(args)
    elif name == "baseline-replay5":
        return BASELINE_replay5(args)
    elif name == "baseline-replay6":
        return BASELINE_replay6(args)
    elif name == "baseline-di":
        return BASELINE_DI(args)

    else:
        assert 0
