from settings import USE_CUDA
if USE_CUDA:
    import cupy as np
else:
    import numpy as np

def rmse(y_pred, y):
    return np.sqrt(np.mean((y_pred.flatten() - y.flatten())**2))

def loss_function(y_pred, y):
    return (2*(y_pred - y)).clip(max=0.8, min=0.0001)

class SSE:
    '''
        Sum of square error
    '''
    def forward(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return np.sum((self.y_pred - self.y)**2)
    
    def backward(self):
        return 2*(self.y_pred - self.y).reshape(-1, 1)

class CrossEntropy:
    epsilon = 1e-5
    def forward(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return (np.where(y==1,-np.log(self.y_pred+self.epsilon), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.y == 1, -1/(self.y_pred+self.epsilon), 0)

def accuracy(y_pred, y):
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y, axis=-1))



if __name__ == "__main__":
    CrossEntropy()