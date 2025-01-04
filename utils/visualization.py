import numpy as np
from matplotlib import pyplot as plt


def display_images_side_by_side(image1, image2, title1="Source", title2="Compressed"):
    """
    Displays two images side by side using matplotlib.

    Parameters:
        image1 (numpy.ndarray): The first image to display.
        image2 (numpy.ndarray): The second image to display.
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    # Ensure both images are NumPy arrays
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    axes[0].imshow(image1, cmap="gray" if len(image1.shape) == 2 else None)
    axes[0].axis('off')  # Turn off axes
    axes[0].set_title(title1)

    # Display the second image
    axes[1].imshow(image2, cmap="gray" if len(image2.shape) == 2 else None)
    axes[1].axis('off')  # Turn off axes
    axes[1].set_title(title2)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()