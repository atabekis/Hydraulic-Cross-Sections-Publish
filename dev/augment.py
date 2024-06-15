import numpy as np

X = np.load('../data/pub_input.npy')
output_data = np.load('../data/pub_out.npy')

# Apply square root transformation to the output data
y = np.sqrt(output_data)


def count_pixels(original_img, transformed_img):
    c_original, c_transformed = np.sum(original_img), np.sum(transformed_img)
    assert c_original == c_transformed, f'White pixel count not preserved! {c_original} != {c_transformed}'


def augment(image):
    transformations = [
        lambda x: x,  # original image
        lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3),  # r90, r180, r270
        np.fliplr, np.flipud,  # flip x, y
        lambda x: np.flipud(np.rot90(x, 1)), lambda x: np.flipud(np.rot90(x, 3))  # # flip X+r90, flip X+r270
    ]
    X_aug = [transform(image) for transform in transformations]

    # Validate pixel count preservation
    for transformed_img in X_aug:
        count_pixels(image, transformed_img)

    return X_aug


X_aug = np.array([aug_img for img in X for aug_img in augment(img)])
y_aug = np.array([target for target in y for _ in range(len(augment(X[0])))])
