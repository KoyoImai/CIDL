from models.finetune import Finetune
from models.PRL import PRL
from models.PRL_MU import PRL_MU
from models.BASELINE import BASELINE
from models.BASELINE_MU import BASELINE_MU


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
    else:
        assert 0
