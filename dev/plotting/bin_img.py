import matplotlib.pyplot as plt
import numpy as np

# Sample 40x40 binary image
binary_image = np.load('../../data/pub_input.npy')[22]
binary_image = binary_image.astype(int)

# Plot the binary image
plt.figure(figsize=(10, 10))
plt.imshow(binary_image, cmap='gray', interpolation='none')

# Step 3: Overlay binary digits on each block with conditional color
for i in range(40):
    for j in range(40):
        color = 'white' if binary_image[i, j] == 0 else 'black'
        plt.text(j, i, str(binary_image[i, j]), color=color, ha='center', va='center', fontsize=10)

# Display the plot
# plt.axis('off')
# plt.savefig('../../fig/binary_rep.png', bbox_inches='tight', pad_inches=0)
plt.show()
