from settings import USE_CUDA
if USE_CUDA:
    import cupy as np
else:
    import numpy as np

class Model:

    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss_func = loss
        self.optimizer = optimizer
        for layer in self.layers:
            if layer.has_weight():
                layer.set_optimizer(self.optimizer)
    
    def set_learning_rate(self, lr):
        self.optimizer.update_lr(lr)
        for idx, layer in enumerate(self.layers):
            if layer.has_weight():
                self.layers[idx].update_lr(lr)

    def forward(self, X, training=True):
        for layer in self.layers:
            if type(layer).__name__  == 'Dropout':
                X = layer.forward(X, training)
            else:
                X = layer.forward(X)
        return X
    
    def predict(self, X):
        return self.forward(X, training=False)
    
    def get_layers_output(self, X, stop_at=4, training=False):
        for layer in self.layers[:stop_at]:
            if type(layer).__name__  == 'Dropout':
                X = layer.forward(X, training)
            else:
                X = layer.forward(X)
        return X
    
    def evaluate(self, X, y):
        y_pred = self.forward(X, training=False)
        loss = self.loss_func.forward(y_pred, y)
        return loss
    
    def train_on_batch(self, X, y):
        y_pred = self.forward(X, training=True)
        # print(y_pred)
        self.loss = self.loss_func.forward(y_pred, y)
        return self.loss 

    def update_weight(self):
        gradient = self.loss_func.backward()
        # print(gradient.shape)

        for idx in range( len(self.layers)-1, -1, -1):
            # print(idx)
            # print(type(self.layers[idx]))
            gradient = self.layers[idx].backward(gradient)

        for layer in self.layers:
            layer.update()
    
    def fit(self, X, y, n_epochs=100, batch_size=16):
        training_loss = []
        for epoch in range(n_epochs):
            for start_idx in range(0, len(X), batch_size):
                end_idx = start_idx+batch_size
                x_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                self.train_on_batch(x_batch, y_batch)
                self.update_weight()
            
            training_loss.append(np.mean(self.evaluate(X, y)/len(X)))
        return training_loss