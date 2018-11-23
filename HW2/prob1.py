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

num_classes = 10

def build_model():
    model = Model([
        Conv(8, kernel_size=(3,3)), 
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        # Dropout(0.2),
        Conv(16, kernel_size=(3,3)),
        MaxPooling(pool_size=2, strides=2), 
        ReLU(),
        Flatten(),
        # Dropout(0.2),
        Dense(input_dim=784, output_dim=256),
        ReLU(),
        Dense(input_dim=256, output_dim=num_classes),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.001) )
    return model


def test_run():
    epoch_num = 120
    batch_size = 512

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    X_train = x_train.reshape(len(x_train), 1, 28, 28)

    X_test = x_test.reshape(len(x_test), 1, 28, 28)

    clf = build_model()


    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state=42)

    # y_train = y_train[:20000]
    # X_train = X_train[:20000]
    training_history = {
        'training_loss': [],
        'val_loss': [],
        'testing_acc': [],
        'training_acc': [],
        'validation_acc': [],
    }
    pickle.dump(clf, open("mnist_test_model.pkl", "wb"))
    for epoch in range(epoch_num):
        train_loss = 0
        for idx, start_idx in enumerate(range(0, len(X_train)-batch_size, batch_size)):
            end_idx = start_idx+batch_size
            x_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            loss = np.mean(clf.train_on_batch(x_batch, y_batch))
            train_loss += loss
            # training_history['training_loss'].append(loss)
            clf.update_weight()
            if idx % 50 == 0:
                print("Iter : %d, loss : %f" % (idx, loss))
        training_acc = accuracy( clf.predict(X_train), y_train)
        validation_acc = accuracy( clf.predict(X_test), y_test)
        testing_acc = accuracy(clf.predict(x_val), y_val)

        val_loss = np.mean(clf.evaluate(X_test, y_test))
        training_history['training_loss'].append(train_loss/len(X_train))
        training_history['val_loss'].append(val_loss)
        training_history['testing_acc'].append(testing_acc)
        training_history['training_acc'].append(training_acc)
        training_history['validation_acc'].append(validation_acc)

        print("Epoch: %d, Loss: %f, Acc: %f, Val Acc: %f, Val loss %f" % ( epoch, train_loss/len(X_train), training_acc, testing_acc, val_loss))

    with open('mnist_finished_model.pkl', 'wb') as f:
        pickle.dump(clf, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(training_history, open("mnist_training_history.pkl", "wb"))

if __name__ == "__main__":
    test_run()