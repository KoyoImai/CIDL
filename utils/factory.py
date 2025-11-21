from models.finetune import Finetune
from models.PRL import PRL

def get_model(model_name, args):

    # model(手法) の名前
    name = model_name.lower()
    
    # model名毎に異なる learner を返す
    if name == "finetune":
        return Finetune(args)
    elif name == "prl":
        return PRL(args)
    elif name == "prl-mu":
        assert False
    else:
        assert 0
