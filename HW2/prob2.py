from models import Model
import keras
from keras.datasets import cifar10
from layers import *
from utils import *
import pickle
from optimizers import SGD, RMSProp
from sklearn.model_selection import train_test_split
from metrics import *
import joblib
from helper import MacOSFile
from sklearn.model_selection import train_test_split

num_classes = 10

def load_model():
    cifar_model = Model([
        Conv(16, kernel_size=(3,3)), 
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(32, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(32, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Conv(64, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Dropout(0.1),
        Flatten(),
        Dense(input_dim=256, output_dim=num_classes),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.0005) )
    return cifar_model


def test_run():
    cifar_model = load_model()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(len(x_train), 3, 32, 32).astype(dtype=np.float64)
    x_test = x_test.reshape(len(x_test), 3, 32, 32).astype(dtype=np.float64)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    epoch_num = 50
    batch_size = 64
    # Convert class vectors to binary class matrices.
    training_history = {
        'training_loss': [],
        'val_loss': [],
        'testing_acc': [],
        'training_acc': [],
        'validation_acc': [],
    }
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size= 0.3, random_state=42)

    for epoch in range(epoch_num):
        train_loss = 0
        for idx, start_idx in enumerate(range(0, len(x_train)-batch_size, batch_size)):
            end_idx = start_idx+batch_size
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            loss = np.mean(cifar_model.train_on_batch(x_batch, y_batch))
            train_loss += loss
            cifar_model.update_weight()
            if idx % 100 == 0:
                print("Iter : %d, loss : %f" % (idx, loss))
        training_acc = accuracy( cifar_model.predict(x_train[:100]), y_train[:100])
        testing_acc = accuracy( cifar_model.predict(x_val), y_val)
        validation_acc = accuracy( cifar_model.predict(x_test), y_test)
        val_loss = np.mean(cifar_model.evaluate(x_test[:100], y_train[:100]))

        training_history['training_loss'].append(train_loss/len(x_train))
        training_history['val_loss'].append(val_loss)
        training_history['testing_acc'].append(testing_acc)
        training_history['training_acc'].append(training_acc)
        training_history['validation_acc'].append(validation_acc)

        print("Epoch: %d, Loss: %f, Acc: %f, Val Acc: %f, Val loss %f" % ( epoch, train_loss/len(x_train), training_acc, testing_acc, val_loss))

        if epoch % 100 == 0:
            joblib.dump(training_history, open("cifar_training_history.pkl", "wb"))
            with open('cifar_finished_model.pkl', 'wb') as f:
                joblib.dump(cifar_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    test_run()