from models import Model
import keras
from keras.datasets import cifar10
from layers import *
from utils import *
from optimizers import SGD, RMSProp
from sklearn.model_selection import train_test_split
from metrics import *

num_classes = 10

def load_model():
    cifar_model = Model([
        Conv(8, kernel_size=(3,3)), 
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(16, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(32, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(64, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Flatten(),
        Dense(input_dim=256, output_dim=num_classes),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.005) )
    return cifar_model


def test_run():
    cifar_model = load_model()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(len(x_train), 3, 32, 32)

    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    epoch_num = 50
    batch_size = 64
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    for epoch in range(epoch_num):
        train_loss = 0
        for start_idx in range(0, len(x_train)-batch_size, batch_size):
            end_idx = start_idx+batch_size
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            loss = np.mean(cifar_model.train_on_batch(x_batch, y_batch))
            train_loss += loss
            cifar_model.update_weight()
            if start_idx%50 == 0:
                print("Iter : %d, loss : %f" % (start_idx, loss))
        training_acc = accuracy( cifar_model.predict(x_train), y_train)
        testing_acc = accuracy( cifar_model.predict(x_test), y_test)
        val_loss = np.mean(cifar_model.evaluate(x_test[:100], y_train[:100]))
        print("Epoch: %d, Loss: %f, Acc: %f, Val Acc: %f, Val loss %f" % ( epoch, train_loss/len(X_train), training_acc, testing_acc, val_loss))

if __name__ == "__main__":
    test_run()