from settings import USE_CUDA
if USE_CUDA:
    import cupy as np
else:
    import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, decay=0.0 ):
        self.learning_rate= learning_rate
        self.decay = decay


    def update(self, grad):
        update = self.learning_rate * grad
        return update
    
    def copy(self):
        return SGD(self.learning_rate, self.decay)

    def update_lr(self, learning_rate):
        self.learning_rate= learning_rate

class SGD2:

    def __init__(self, learning_rate=0.1, decay=0.0, momentum=0.1 ):
        self.learning_rate= learning_rate
        self.initial_lr = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0
        self.velocity = 0


    def update(self, grad):
        self.iteration += 1
        self.velocity =  (self.learning_rate * grad) + self.velocity*self.momentum
        update = self.velocity
        self.learning_rate *= (1 - self.decay)
        return update
    
    def copy(self):
        return SGD(self.learning_rate, self.decay, self.momentum)

    def update_lr(self, learning_rate):
        self.learning_rate = learning_rate

class RMSProp:

    def __init__(self, learning_rate=0.001, rho=0.9, decay_rate=0.9 ):
        self.learning_rate= learning_rate
        self.decay_rate = decay_rate
        self.rho = rho
        self.cache = 0
        self.epsilon= 1e-07

    def update(self, grad):
        self.cache =  self.rho * self.cache + (1 - self.rho) * (grad**2)
        update = self.learning_rate * grad / (np.sqrt(self.cache) + self.epsilon)
        return update
    
    def copy(self):
        return RMSProp(self.learning_rate, self.rho, self.decay_rate)
    
    def update_lr(self, learning_rate):
        self.learning_rate= learning_rate
