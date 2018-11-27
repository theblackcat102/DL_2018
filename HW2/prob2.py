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
from tqdm import tqdm

num_classes = 10

def load_model(l2_regularization=None):
    cifar_model = Model([
        Conv(16, kernel_size=(3,3), l2_regularization=l2_regularization),
        ReLU(),
        Conv(16, kernel_size=(3,3), l2_regularization=l2_regularization),
        ReLU(),
        # Dropout(0.1),
        MaxPooling(pool_size=2, strides=2),
        Conv(16, kernel_size=(3,3), l2_regularization=l2_regularization),
        ReLU(),
        MaxPooling(pool_size=2, strides=2),
        Flatten(),
        # Dropout(0.2),
        Dense(input_dim=1024, output_dim=128),
        # Dropout(0.5),
        ReLU(),
        Dense(input_dim=128, output_dim=num_classes, l2_regularization=l2_regularization),
        Softmax(),
        ],loss=CrossEntropy(), optimizer=RMSProp(learning_rate=0.001) )
    return cifar_model

def test_predict():
    cifar_model = load_model()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(len(x_train),  32, 32, 3).astype(dtype=np.float64)
    x_test = x_test.reshape(len(x_test), 32, 32, 3).astype(dtype=np.float64)

    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    cifar_model.train_on_batch(x_train[:10], y_train[:10])
    y_preds = cifar_model.predict(x_train)


def test_evaluate():
    cifar_model = load_model()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(len(x_train), 3, 32, 32).astype(dtype=np.float64)
    x_test = x_test.reshape(len(x_test), 3, 32, 32).astype(dtype=np.float64)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    val_loss = np.mean(cifar_model.evaluate(x_test, y_test))

def test_run(regularizer=None):
    if regularizer is not None:
        print(regularizer)
    cifar_model = load_model(regularizer)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(len(x_train), 3, 32, 32).astype(dtype=np.float64)
    x_test = x_test.reshape(len(x_test), 3, 32, 32).astype(dtype=np.float64)

    x_train = (x_train -  0) / 255.0
    x_test = (x_test - 0) / 255.0

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    epoch_num = 50
    batch_size = 128
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

    train_idx = [ x for x in range(0,len(x_train))]
    with tqdm(total=epoch_num*batch_size, ncols=80) as pbar:
        for epoch in range(epoch_num):
            train_loss = 0
            for idx, start_idx in enumerate(range(0, len(x_train)-batch_size, batch_size)):
                end_idx = start_idx+batch_size
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                loss = np.mean(cifar_model.train_on_batch(x_batch, y_batch))
                train_loss += loss
                cifar_model.update_weight()
                if idx % 10 == 0:
                    testing_acc = accuracy( cifar_model.predict(x_val), y_val)
                    pbar.write('Epoch %3d/%3d, train-loss: %.4f,'
                                'val-acc: %.3f' % (epoch + 1, epoch_num, loss, testing_acc))
                    pbar.update(1)
                    # print("Iter : %d, loss : %f, accuracy: %f" % (idx, loss, testing_acc))
            # training data accuracy
            training_acc = accuracy( cifar_model.predict(x_train[:100]), y_train[:100])
            # training data validation split
            testing_acc = accuracy( cifar_model.predict(x_val), y_val)
            # testing dataset accuracy
            validation_acc = accuracy( cifar_model.predict(x_test), y_test)
            val_loss = np.mean(cifar_model.evaluate(x_test, y_test))

            training_history['training_loss'].append(train_loss/len(x_train))
            training_history['val_loss'].append(val_loss)
            training_history['testing_acc'].append(testing_acc)
            training_history['training_acc'].append(training_acc)
            training_history['validation_acc'].append(validation_acc)
            
            np.random.shuffle(train_idx)
            x_train = x_train[train_idx]
            y_train = y_train[train_idx]
            if regularizer is None:
                filename = "cifar_%d_training_history.pkl" % epoch
            else:
                filename = "cifar_%d_training_history_%f.pkl" % (epoch, regularizer)

            joblib.dump({'history': training_history,
                'model': cifar_model }, open(filename, "wb"))
    with open('model_'+filename, 'wb') as f:
        pickle.dump(cifar_model, f)
    
if __name__ == "__main__":
    test_run(regularizer=0.001)