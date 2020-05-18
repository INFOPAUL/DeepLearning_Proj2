
class Module(object):
    def forward(self , input):
        raise  NotImplementedError

    def backward(self , grad):
        raise  NotImplementedError

    def param(self):
        return []

    def update_params(self, func):
        pass

    def zero_grad(self):
        pass