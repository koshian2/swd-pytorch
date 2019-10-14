import torch
from swd import swd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# random image : change proj_repeat
def random_value_test():
    torch.manual_seed(123)
    image1 = torch.randn(16384, 3, 128, 128)
    image2 = torch.randn(16384, 3, 128, 128)
    np.set_printoptions(precision=2)
    result = {}
    for n_proj in [128, 64, 32, 16, 8, 4, 2, 1]:
        dists = []
        for i in range(10):
            dists.append(swd(image1, image2, proj_per_repeat=n_proj,
                         n_repeat_projection=512 // n_proj,
                         device="cuda" if n_proj <= 32 else "cpu",
                         return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[n_proj] = dists
        print("proj=", n_proj, dists)
    with open("swd_random_test.pkl", "wb") as fp:
        pickle.dump(result, fp)

# random image : change image size
def random_value_test2():
    np.set_printoptions(precision=2)
    result = {}
    for image_size in [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32]:
        torch.manual_seed(123)
        image1 = torch.randn(image_size, 3, 128, 128)
        image2 = torch.randn(image_size, 3, 128, 128)
        dists = []
        for i in range(10):
            dists.append(swd(image1, image2, proj_per_repeat=32,
                         n_repeat_projection=16,
                         device="cuda", return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[image_size] = dists
        print("image_size", image_size, dists)
    with open("swd_random_test2.pkl", "wb") as fp:
        pickle.dump(result, fp)

## Test on cifar
def load_cifar(ignore_train_classes):
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    labels = np.array(dataset.targets)
    if len(ignore_train_classes) > 0:
        filter_labels = labels.reshape(-1, 1) == np.array(ignore_train_classes).reshape(1, -1)
        filter_labels = np.any(filter_labels, axis=1)
    else:
        filter_labels = np.zeros(labels.shape[0], np.bool)
    train_img = torch.as_tensor(dataset.data[~filter_labels][:10000].transpose([0, 3, 1, 2]).astype(np.float32) / 255.0)        
    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    test_img = torch.as_tensor(dataset.data.transpose([0, 3, 1, 2]).astype(np.float32) / 255.0)
    return train_img, test_img

# cifar10 : change proj_repeat
def cifar_test():
    train_img, test_img = load_cifar([])

    np.set_printoptions(precision=2)
    result = {}
    for n_proj in [128, 64, 32, 16, 8, 4, 2, 1]:
        dists = []
        for i in range(10):
            dists.append(swd(train_img, test_img, proj_per_repeat=n_proj,
                         n_repeat_projection=512 // n_proj,
                         device="cuda" if n_proj <= 64 else "cpu",
                         return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[n_proj] = dists
        print("proj=", n_proj, dists)
    with open("swd_cifar_test.pkl", "wb") as fp:
        pickle.dump(result, fp)

# cifar : change image size
def cifar_test2():
    train_img, test_img = load_cifar([])

    np.set_printoptions(precision=2)
    result = {}
    for image_size in [10000, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32]:
        dists = []
        for i in range(10):
            dists.append(swd(train_img[:image_size], test_img[:image_size], proj_per_repeat=64,
                         n_repeat_projection=8,
                         device="cuda", return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[image_size] = dists
        print("image_size", image_size, dists)
    with open("swd_cifar_test2.pkl", "wb") as fp:
        pickle.dump(result, fp)

# cifar : for distribution mismatch
def cifar_remove_class_test():
    np.set_printoptions(precision=2)
    result = {}
    for i in range(9):
        train_img, test_img = load_cifar([j for j in range(i)])  # remove classes
        dists = []
        for j in range(10):
            dists.append(swd(train_img, test_img, proj_per_repeat=64, n_repeat_projection=8,
                             device="cuda", return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[i] = dists
        print("remove classes", i, dists)
    with open("swd_cifar_remove_class.pkl", "wb") as fp:
        pickle.dump(result, fp)

# cifar : for inbalance data mismatch
def cifar_inbalance_class_test():
    np.set_printoptions(precision=2)
    result = {}
    for i in range(11):
        dists = []
        for remove_class in range(10):
            # inbalance for one class, others are normal
            balanced_img, test_img = load_cifar([])
            inbalanced_img, _ = load_cifar([remove_class])
            train_img = torch.cat([inbalanced_img[:i * 1000], balanced_img[i * 1000:]], dim=0)            
            dists.append(swd(train_img, test_img, n_repeat_projection=8, proj_per_repeat=64,
                             device="cuda", return_by_resolution=True).numpy())
        dists = np.array(dists)
        result[i * 1000] = dists
        print("n_inbalance", i * 1000, dists)
    with open("swd_cifar_inbalance.pkl", "wb") as fp:
        pickle.dump(result, fp)


def plot_results(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    n = len(next(iter(data.values()))[0])
    for i in range(n):
        points = []
        for key in data.keys():
            points.append(data[key][:, i])
        rn = int(np.ceil(np.sqrt(n)))
        ax = plt.subplot(rn, rn, i + 1)
        ax.boxplot(points)
        ax.set_xticklabels(data.keys())
    plt.show()

def plot_inbalance(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.2, wspace=0.1, left=0.05, right=0.95)
    for i in range(10):
        points = []
        for key in data.keys():
            points.append(np.mean(data[key][i, :]))
        ax = plt.subplot(5, 2, i + 1)
        ax.plot(data.keys(), points, label="class = " + str(i))
        ax.legend()
    plt.show()
