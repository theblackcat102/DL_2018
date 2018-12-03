import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
import seaborn as sns
import argparse
import pandas as pd
from metrics import accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m',type=str, help='model checkpoint')
args = parser.parse_args()

def viz_mnist(model_name):
    stats = joblib.load(open(model_name, "rb"))

    mnist = stats['model']
    training_stats = stats['history']
    


    # plt.figure()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
    plot_axis = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

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
    weights_dist = mnist.layers[2].weights.flatten()
    plot_axis[3].set_title("History of conv2")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[3])

    weights_dist = mnist.layers[5].weights.flatten()
    plot_axis[4].set_title("History of conv3")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[4])


    # dense 1
    weights_dist = mnist.layers[-4].W.flatten()
    plot_axis[5].set_title("History of dense1")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[5])

    weights_dist = mnist.layers[-2].W.flatten()
    plot_axis[6].set_title("History of dense2")
    sns.distplot(weights_dist, bins=50, ax=plot_axis[6])


    import keras
    predict_batch = 500
    max_matched = 10
    
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.reshape(len(x_train),  32, 32, 3).astype(dtype=np.float64)
    x_test = x_test.reshape(len(x_test), 32, 32, 3).astype(dtype=np.float64)

    # X_train = x_train.transpose(0, 3, 1, 2)
    X_train = np.rollaxis(x_train, 3, 1)  
    X_test = np.rollaxis(x_test, 3, 1) 
    print(accuracy(mnist.predict(X_test)/ 255.0, y_test))

    images = []
    labels = []
    falsed_labels = []
    true_labels = []
    falsed_images = []

    y_preds = mnist.predict(X_train[:predict_batch]/ 255.0)
    for idx, y_pred in enumerate(y_preds):
        pred_num = np.argmax(y_pred.flatten())
        true_num = np.argmax(y_train[idx])

        if pred_num == true_num:
            if len(labels) > 0 and pred_num in labels:
                continue
            else:
                images.append(X_train[idx])
                labels.append(true_num)
                max_matched -= 1
        else:
            if len(labels) > 0 and pred_num not in labels:
                continue
            else:
                falsed_labels.append(pred_num)
                true_labels.append(true_num)
                falsed_images.append(X_train[idx])
        if max_matched == 0:
            print("Found two matched!")
            break
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig, (upper_row,lower_row) = plt.subplots(nrows=2, ncols=3)

    for i in range(3):
        upper_row[i].imshow(np.moveaxis(images[i], 0, -1).astype(dtype=np.uint8))
        upper_row[i].set_title("True Label %s, Predict Label %s" % (class_name[labels[i]], class_name[labels[i]]))
        lower_row[i].imshow(np.moveaxis(falsed_images[i], 0, -1).astype(dtype=np.uint8))
        lower_row[i].set_title("True Label %s, Predict Label %s" % (class_name[true_labels[i]], class_name[falsed_labels[i]]))

    for image_id in range(len(labels)):
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(nrows=1, ncols=4)
        fig.suptitle("Predict class %s, True class %s" % (class_name[labels[image_id]], class_name[labels[image_id]]))
        ax1.imshow(np.moveaxis(images[image_id], 0,-1).astype(dtype=np.uint8))
        ax1.set_title('Original Image')
        conv1_output = mnist.layers[0].forward(np.asarray([images[image_id]]))
        ax2.imshow(conv1_output[0][0])
        ax2.set_title('Conv1 output')

        activated_output = mnist.layers[1].forward(conv1_output)
        conv2_output = mnist.layers[2].forward(activated_output)
        activated_output = mnist.layers[3].forward(conv2_output)
        pool1_output = mnist.layers[4].forward(activated_output)

        conv3_output = mnist.layers[5].forward(pool1_output)
        ax3.set_title('Conv2 output')
        ax3.imshow(conv2_output[0][0])

        ax4.set_title('Conv 3 output')
        ax4.imshow(conv3_output[0][0])

    plt.show()

if __name__ == "__main__":
    viz_mnist(args.model)