import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        use_gpu = getattr(self.args, "use_gpu", False) and torch.cuda.is_available()
        if use_gpu:
            device = torch.device("cuda:0")
            print("Use GPU: cuda:0")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
