import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Load the binary image
binary_image = np.load('../../data/pub_input.npy')[22]
binary_image = binary_image.astype(int)

# Extract the subset of the image
subset = binary_image[20:26, 10:16]

# Convert the subset to a tensor and add batch and channel dimensions
input_tensor = torch.tensor(subset, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Define a convolutional layer with a 5x5 kernel size
conv_layer = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0, bias=False)

# Use a fixed kernel for visualization purposes (simple averaging kernel for demonstration)
conv_layer.weight.data = torch.tensor([[[[1, 1, 1, 1],
                                         [1, 1, 1, 1],
                                         [1, 1, 1, 1],
                                         [1, 1, 1, 1]]]], dtype=torch.float32) / 16.0

# Perform the convolution
output_tensor = conv_layer(input_tensor)

# Convert the output tensor to a numpy array
output_image = output_tensor.squeeze().detach().numpy()

# Plot the subset and the convolved output
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Subset image with binary digits overlaid
# axes[0].imshow(subset, cmap='gray', interpolation='none', vmin=0, vmax=1)
# for i in range(subset.shape[0]):
#     for j in range(subset.shape[1]):
#         color = 'white' if subset[i, j] == 0 else 'black'
#         axes[0].text(j, i, str(subset[i, j]), color=color, ha='center', va='center', fontsize=10)
# axes[0].set_title('Subset of Binary Image with Digits')

# Convolved output image with values overlaid
plt.imshow(output_image, cmap='gray', interpolation='none', vmin=0, vmax=1)
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        plt.text(j, i, f'{output_image[i, j]:.2f}', color='white', ha='center', va='center', fontsize=10)
# plt.set_title('Convolved Output Image with Values')
plt.axis('off')
plt.savefig('../../fig/convolved.png', bbox_inches='tight', pad_inches=0)
plt.show()



subset_start_y, subset_end_y = 20, 26
subset_start_x, subset_end_x = 10, 16

# Plot the original binary image with a red box highlighting the subset
plt.figure(figsize=(10, 10))
plt.imshow(binary_image, cmap='gray', interpolation='none')

# Overlay binary digits on each block with conditional color
for i in range(40):
    for j in range(40):
        color = 'white' if binary_image[i, j] == 0 else 'black'
        plt.text(j, i, str(binary_image[i, j]), color=color, ha='center', va='center', fontsize=10)

# Draw a red box around the subset
rect = plt.Rectangle((subset_start_x - 0.5, subset_start_y - 0.5), subset_end_x - subset_start_x, subset_end_y - subset_start_y, edgecolor='red', facecolor='none', linewidth=2)
plt.gca().add_patch(rect)
plt.axis('off')
plt.savefig('../../fig/binary_rep_w_box.png', bbox_inches='tight', pad_inches=0)
plt.show()

