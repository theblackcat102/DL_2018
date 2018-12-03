import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
import seaborn as sns
import argparse
import pandas as pd
import keras
from metrics import accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m',type=str, help='model checkpoint')
args = parser.parse_args()

predict_batch = 100

num_classes = 10

def viz_mnist(model_name):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    X_train = x_train.reshape(len(x_train), 1, 28, 28) / 255.0
    X_test = x_test.reshape(len(x_test), 1, 28, 28) / 255.0

    stats = joblib.load(open(model_name, "rb"))

    mnist = stats['model']
    training_stats = stats['history']
    accuracy_rate = accuracy(mnist.predict(X_test), y_test)
    print("Test data accuracy : %f" % accuracy_rate)
    # plt.figure()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
    plot_axis = [ax1, ax2, ax3, ax4, ax5, ax6]

    iterations = [ x for x in range(len(training_stats['training_loss'])) ]

    accurate_dict = {
        'Accuracy Rate': [0,0,0],
        'Iteration': [0,0,0],
        'Type': ['Training Accuracy','Validation Accuracy','Test Accuracy'],
    }
    for idx in range(len(training_stats['training_loss'])):
        for key in ['testing_acc', 'validation_acc', 'training_acc']:        
            accurate_dict['Accuracy Rate'].append(training_stats[key][idx])
            accurate_dict['Iteration'].append(idx+1)
            if key == 'training_acc':
                accurate_dict['Type'].append('Training Accuracy')
            elif key == 'testing_acc':
                accurate_dict['Type'].append('Testing Accuracy')
            elif key == 'validation_acc':
                accurate_dict['Type'].append('Validation Accuracy')

    accuracy_rate = pd.DataFrame(data=accurate_dict)
    plot_axis[0].set_title("Training Accuracy")
    sns.lineplot(x='Iteration', y='Accuracy Rate',hue='Type', data=accuracy_rate, ax=plot_axis[0])


    learning_curve = pd.DataFrame(data={
        'Iteration': iterations,
        'Loss': training_stats['training_loss']
    })
    plot_axis[1].set_title("Learning Curve")
    sns.lineplot(x='Iteration', y='Loss', data=learning_curve, ax=plot_axis[1])

    # conv1
    weights_dist = mnist.layers[0].weights.flatten()
    plot_axis[2].set_title("History of conv1")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[2])

    # conv2
    weights_dist = mnist.layers[3].weights.flatten()
    plot_axis[3].set_title("History of conv2")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[3])

    # dense 1
    weights_dist = mnist.layers[-4].W.flatten()
    plot_axis[4].set_title("History of dense1")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[4])

    weights_dist = mnist.layers[-2].W.flatten()
    plot_axis[5].set_title("History of dense2")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[5])


    images = []
    labels = []
    falsed_labels = []
    true_labels = []
    falsed_images = []


    max_matched = 10

    y_preds = mnist.predict(X_train[:predict_batch])
    for idx, y_pred in enumerate(y_preds):
        pred_num = np.argmax(y_pred.flatten())
        true_num = np.argmax(y_train[idx])

        if pred_num == true_num:
            if len(labels) > 0 and pred_num in labels:
                continue
            else:
                images.append(X_train[idx][0])
                labels.append(true_num)

        else:
            if len(labels) > 0 and pred_num not in labels:
                continue
            else:
                falsed_labels.append(pred_num)
                true_labels.append(true_num)
                falsed_images.append(X_train[idx][0])
                max_matched -= 1
        if max_matched == 0:
            print("Found two matched!")
            break

    fig, (upper_row,lower_row) = plt.subplots(nrows=2, ncols=3)

    for i in range(3):
        upper_row[i].imshow(images[i])
        upper_row[i].set_title("True Label %d, Predict Label %d" % (labels[i], labels[i]))
        lower_row[i].imshow(falsed_images[i])
        lower_row[i].set_title("True Label %d, Predict Label %d" % (true_labels[i], falsed_labels[i]))

    for image_id in range(10):
        fig, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(nrows=1, ncols=6)

        ax1.imshow(images[image_id])
        ax1.set_title('Original Image')
        conv1_output = mnist.layers[0].forward(np.asarray([[images[image_id]]]))
        ax2.imshow(conv1_output[0][0])
        ax2.set_title('Conv1 output')

        activated_output = mnist.layers[1].forward(conv1_output)
        ax3.imshow(activated_output[0][0])
        ax3.set_title('ReLU output')

        maxpool_output = mnist.layers[2].forward(activated_output)
        ax4.imshow(maxpool_output[0][0])
        ax4.set_title('Maxpool output')

        conv2_output = mnist.layers[3].forward(maxpool_output)
        ax5.imshow(conv2_output[0][0])
        ax5.set_title('Conv 2 output')

        relu2_output = mnist.layers[4].forward(conv2_output)
        ax6.imshow(relu2_output[0][0])
        ax6.set_title('ReLU 2 output')

        pool2_output = mnist.layers[5].forward(relu2_output)
        ax6.imshow(pool2_output[0][0])
        ax6.set_title('Maxpool 2 output')

    plt.show()

if __name__ == "__main__":
    viz_mnist(args.model)