import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_cifar(ignore_train_classes):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if len(ignore_train_classes) > 0:
        filter_labels = y_train == np.array(ignore_train_classes).reshape(1, -1)
        filter_labels = np.any(filter_labels, axis=1)
    else:
        filter_labels = np.zeros(y_train.shape[0], np.bool)
    train_img = X_train[~filter_labels][:10000].astype(np.float32) / 255.0
    test_img = X_test.astype(np.float32) / 255.0
    return tf.constant(train_img), tf.constant(test_img)

# SSIM : for distribution mismatch
def cifar_remove_class_test():
    np.set_printoptions(precision=2)
    result = {}
    for i in range(9):
        train_img, test_img = load_cifar([j for j in range(i)])  # remove classes
        dist = tf.reduce_mean(tf.image.ssim(train_img, test_img, max_val=1.0)).numpy()
        result[i] = dist
        print("remove classes", i, dist)
    with open("ssim_cifar_remove_class.pkl", "wb") as fp:
        pickle.dump(result, fp)

def plot_results(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    plt.plot(data.keys(), data.values())
    plt.show()
