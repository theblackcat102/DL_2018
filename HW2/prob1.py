from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
from utils import *
from optimizers import RMSProp, SGD
from models import Model
from layers import *
from helper import MacOSFile
from metrics import *
import pickle
from sklearn.model_selection import train_test_split
import datetime
from sklearn.externals import joblib
num_classes = 10

def build_model():
    model = Model([
        Conv(16, kernel_size=(3,3)), 
        ReLU(),
        Conv(16, kernel_size=(3,3)), 
        ReLU(),
        MaxPooling(pool_size=2, strides=2), 
        Conv(32, kernel_size=(3,3)),
        ReLU(),
        MaxPooling(pool_size=2, strides=2), 
        # Conv(64, kernel_size=(3,3)),
        # MaxPooling(pool_size=2, strides=1), 
        # ReLU(),
        Flatten(),
        Dense(input_dim=1568, output_dim=256),
        ReLU(),
        Dense(input_dim=256, output_dim=num_classes),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.0002) )
    return model

def benchmark():
    
    epoch_num = 120
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_train = x_train.reshape(len(x_train), 1, 28, 28)

    X_test = x_test.reshape(len(x_test), 1, 28, 28)

    clf = build_model()
    x_batch = X_train[:batch_size]
    y_batch = y_train[:batch_size]
    avg_sum = 0
    for i in range(10):
        start = datetime.datetime.now()
        loss = np.mean(clf.train_on_batch(x_batch, y_batch))
        end = datetime.datetime.now()
        delta = end - start
        avg_sum += delta.total_seconds() * 1000
    print("Batch size %d, time : %f ms" % (batch_size, avg_sum/10))

def test_run():
    epoch_num = 50
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_train = (x_train.reshape(len(x_train), 1, 28, 28).astype('float32') - 125 ) / 255.0

    X_test = (x_test.reshape(len(x_test), 1, 28, 28).astype('float32') - 125 ) / 255.0

    clf = build_model()


    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state=42)
    train_idx = [ x for x in range(0,len(X_train))]
    # y_train = y_train[:20000]
    # X_train = X_train[:20000]
    training_history = {
        'training_loss': [],
        'val_loss': [],
        'testing_acc': [],
        'training_acc': [],
        'validation_acc': [],
    }
    # pickle.dump(clf, open("mnist_test_model.pkl", "wb"))
    for epoch in range(epoch_num):
        train_loss = 0
        start = datetime.datetime.now()
        for idx, start_idx in enumerate(range(0, len(X_train)-batch_size, batch_size)):
            end_idx = start_idx+batch_size
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            loss = np.mean(clf.train_on_batch(x_batch, y_batch))
            train_loss += loss
            # training_history['training_loss'].append(loss)
            clf.update_weight()
            if idx % 100 == 0:
                print("Iter : %d, loss : %f" % (idx, loss))
        training_acc = accuracy( clf.predict(X_train[:100]), y_train[:100])
        validation_acc = accuracy( clf.predict(X_test), y_test)
        testing_acc = accuracy(clf.predict(x_val), y_val)

        val_loss = np.mean(clf.evaluate(X_test, y_test))
        training_history['training_loss'].append(train_loss/len(X_train))
        training_history['val_loss'].append(val_loss)
        training_history['testing_acc'].append(testing_acc)
        training_history['training_acc'].append(training_acc)
        training_history['validation_acc'].append(validation_acc)
        end = datetime.datetime.now()
        delta = end - start
        print("Epoch: %d, Time : %f" % (epoch, delta.total_seconds() * 1000))
        print("Loss: %f, Acc: %f, Val Acc: %f, Val loss %f" % ( train_loss/len(X_train), training_acc, testing_acc, val_loss))

        np.random.shuffle(train_idx)

        X_train = X_train[train_idx]
        y_train = y_train[train_idx]

        if epoch % 10 == 0:
            with open('mnist_finished_model.pkl', 'wb') as f:
                joblib.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
            joblib.dump(training_history, open("mnist_training_history.pkl", "wb"))

if __name__ == "__main__":
    test_run()