import numpy as np
import matplotlib.pyplot as plt
import random

X = np.load('../../data/pub_input.npy')
y = np.load('../../data/pub_out.npy')



def plot_image(ax, image, index, add=None, S=None):
    ax.imshow(image, cmap='gray')
    ax.set_title(f"$ CS_{{{index + 1}}} $ {add or ''}\n {S} ")  # oh god
    ax.set_xlabel('$ x \ [m] $')
    ax.set_ylabel('$ x \ [m] $')


# Load the dataset

# Choose one image from the dataset
image = X[22]

# Generate permutations
permutations = np.array([np.random.permutation(image.flatten()).reshape(40, 40) for _ in range(10)])

# Select a random permutation
random_permuted_image = permutations[random.randint(0, permutations.shape[0] - 1)]

# Plot the original and permuted images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_image(axes[0], image, index=22, add='[Original Image]', S=f"$ S' = {y[22]:.2f} \quad [m^2] $")
plot_image(axes[1], random_permuted_image, index=22, add='[Random Permuted Image]', S="$ S' = ? $")
plt.savefig('../../fig/og_vs_perm.png', bbox_inches='tight', pad_inches=0.5)
plt.show()
