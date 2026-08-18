import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


def integrate_velocity_field(v, timesteps=10):
    """
    Integrates a given velocity field v over time to compute the diffeomorphic transform.
    Args:
        v (np.ndarray): The velocity field of shape (2, H, W).
        timesteps (int): Number of time steps for the integration.
    Returns:
        np.ndarray: The final transformation field.
    """
    h, w = v.shape[1], v.shape[2]
    dt = 1.0 / timesteps

    # Initialize the transformation field
    phi = np.indices((h, w)).astype(np.float32)

    for _ in range(timesteps):
        # Compute the intermediate transformation field
        phi_v = np.zeros_like(phi)
        for i in range(2):
            phi_v[i] = map_coordinates(v[i], phi, order=1)
        phi += dt * phi_v

    return phi


def apply_transform(image, transform):
    """
    Applies a given transformation field to an image.
    Args:
        image (np.ndarray): The input image.
        transform (np.ndarray): The transformation field.
    Returns:
        np.ndarray: The transformed image.
    """
    coords = np.indices(image.shape).astype(np.float32)
    for i in range(2):
        coords[i] = map_coordinates(transform[i], coords, order=1)
    transformed_image = map_coordinates(image, coords, order=1, mode='nearest')

    return transformed_image


# Example image and velocity field
image = np.zeros((100, 100))
image[30:70, 30:70] = 1  # Create a square in the middle of the image

v = np.zeros((2, 100, 100))
v[0, 50:, :] = 1  # Velocity field in x direction (for the lower half of the image)
v[1, :, 50:] = 1  # Velocity field in y direction (for the right half of the image)

# Compute the transformation field
transform = integrate_velocity_field(v)

# Apply the transformation to the image
transformed_image = apply_transform(image, transform)

# Display the original and transformed images
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(transformed_image, cmap='gray')

plt.show()
