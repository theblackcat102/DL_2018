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


l2_regularization = None
basename = 'mnist_small'

def build_model(l2_regularization=None):
    model = Model([
        Conv(4, kernel_size=(3,3), l2_regularization=l2_regularization),
        ReLU(),
        MaxPooling(pool_size=2, strides=2),
        Conv(4, kernel_size=(3,3), l2_regularization=l2_regularization),
        ReLU(),
        MaxPooling(pool_size=2, strides=2),
        Flatten(),
        Dense(input_dim=196, output_dim=32),
        ReLU(),
        Dense(input_dim=32, output_dim=num_classes, l2_regularization=l2_regularization),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.001) )
    return model

def benchmark():
    
    epoch_num = 50
    batch_size = 32

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

def test_run(regularizer=None):
    epoch_num = 50
    batch_size = 256

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    X_train = (x_train.reshape(len(x_train), 1, 28, 28).astype('float32') ) / 255.0
    X_test = (x_test.reshape(len(x_test), 1, 28, 28).astype('float32') ) / 255.0
    clf = build_model(regularizer)

    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state=42)
    train_idx = [ x for x in range(0,len(X_train))]
    print(regularizer)
    training_history = {
        'training_loss': [],
        'val_loss': [],
        'testing_acc': [],
        'training_acc': [],
        'validation_acc': [],
    }

    if regularizer is None:
        filename = basename + '_finished_model.pkl'
        history_name = basename + '_training_history.pkl'
    else:
        filename = basename + '_finished_model_l2_%f.pkl' % regularizer
        history_name = basename + '_training_history_l2_%f.pkl' % regularizer

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
            # print(training_acc)
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

        if epoch % 5 == 0:
            # with open(filename, 'wb') as f:
            #     pickle.dump(clf, f)
            pickle.dump({'history': training_history,
                'model': clf} , open(history_name, "wb"))


def plot_one(history_name='mnist_training_history.pkl'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    category = ['Training Accuracy','Validation Accuracy','Test Accuracy']
    iterations = [0,0,0]
    value = [0,0,0]

    stats = joblib.load(open(history_name, 'rb'))
    stats_df = pd.DataFrame(data=stats)
    for idx, row in stats_df.iterrows():
        iterations.append(idx+1)
        value.append(row['training_acc'])
        category.append('Training Accuracy')

        iterations.append(idx+1)
        value.append(row['validation_acc'])
        category.append('Validation Accuracy')

        iterations.append(idx+1)
        value.append(row['testing_acc'])
        category.append('Test Accuracy')

    plot_df = pd.DataFrame(data={
        'Accuracy': value,
        'type': category,
        'iterations': iterations,
    })
    sns.lineplot(x='iterations', y='Accuracy', hue='type', data=plot_df)
    plt.show()


if __name__ == "__main__":
    test_run(None)
