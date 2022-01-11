import matplotlib.pyplot as plt
import numpy as np

def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis("off")

def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows), dpi=1200)
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap='gray')
        plt.axis("off")

def plot_multiple_images_with_scores(images, scores, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        ax = plt.subplot(n_rows, n_cols, index + 1)
        ax.text(5, 0, "{:.8f}".format(scores[index]), fontsize=6)
        ax.imshow(image, cmap='gray')
        ax.axis("off")

def plot_interpolation(images):
    n_cols = 10
    n_rows = int(np.ceil(len(images)/10))
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    fig, axs = plt.subplots(n_cols, n_rows, figsize = (10, 10+n_rows*0.1))
    fig.subplots_adjust(wspace=0, hspace=0.1)
    for index, image in enumerate(images):
        axs[index//10, index%10].imshow(image, cmap='gray', aspect="auto")
        axs[index//10, index%10].axis("off")