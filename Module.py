
class Module(object):
    """Module interface class
    
    Initializes the methods that needs to be implemented in children classes
    """
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