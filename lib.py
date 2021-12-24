
import numpy as np
import matplotlib.pyplot as plt
import pylab
import os


def list_directory(path: str) -> list:
    return os.listdir(path)

def plot_singular_values_and_energy(sv: np.ndarray, k: int):
    en_cum = sv.cumsum()

    fig = pylab.figure(figsize=(15, 8))

    fig.add_subplot(1, 2, 2)
    plt.plot(sv)
    plt.vlines(k, 0.0, max(sv), colors='r', linestyles='solid')
    plt.xlim(0, len(sv))
    plt.ylim(0.0, max(sv))
    plt.xlabel('Index of singular value')
    plt.ylabel('Magnitude singular value')

    fig.add_subplot(1, 2, 1)
    plt.plot(en_cum)
    plt.vlines(k, 0.0, max(en_cum), colors='r', linestyles='solid')
    plt.xlim(0, len(en_cum))
    plt.ylim(0.0, max(en_cum))
    plt.ylabel('Accumulated singular values')
    plt.xlabel('Number of first singular value in accumulation.')

    plt.show()


def visualize_eigenfaces(n: int, pcs: np.ndarray, sv: np.ndarray, dim_x: int, dim_y: int):
    fig = pylab.figure(figsize=(15, 8))
    m = int(np.ceil(n / 2))
    n = 2 * m

    for i in range(n):
        fig.add_subplot(2, m, i + 1)
        eface = pcs[i, :].reshape((dim_y, dim_x))
        plt.imshow(eface, cmap="Greys_r")
        plt.title('sigma = %.f' % sv[i])

    plt.show()


def plot_identified_faces(scores: np.ndarray, training_images: list, test_images: list, pcs: np.ndarray, coeffs_test: np.ndarray, mean_data: np.ndarray):
    for i in range(scores.shape[1]):
        j = np.argmin(scores[:, i])

        fig = pylab.figure()

        fig.add_subplot(1, 3, 2)
        plt.imshow(training_images[j], cmap="Greys_r")
        plt.xlabel('Identified person')

        fig.add_subplot(1, 3, 1)
        plt.imshow(test_images[i], cmap="Greys_r")
        plt.xlabel('Query image')

        img_reconst = pcs.transpose().dot(coeffs_test[i, :]) + mean_data
        img_reconst = img_reconst.reshape(test_images[i].shape)

        fig.add_subplot(1, 3, 3)
        plt.imshow(img_reconst, cmap="Greys_r")
        plt.xlabel('Reconstructed image')

        plt.show()
