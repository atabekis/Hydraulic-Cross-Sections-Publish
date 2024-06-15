import numpy as np
import pandas as pd
import torch

from functions import (to_torch, train, RMSPE, plot_losses, to_kaggle,
                       X, y, device)

from dev.augment import X_aug, augmented_targets

from models.simple_models import SimpleCNN, CombinedCNN

from models.discrete_sym import BaseCNN
from models.gcnn import GCNN2, GEquivariantNetwork

from models.vgg import SimpleVGG, ExpandedVGG


train_simple, test_simple = to_torch(X, y)
print(len(train_simple), len(test_simple))
train_aug, test_aug = to_torch(X_aug, augmented_targets)

results = dict()
results.update(train(test_aug, test_aug,
                     model=GEquivariantNetwork(),
                     loss_fn=RMSPE(),
                     num_epochs=1000,
                     batch_size=32,
                     learning_rate=1e-3,
                     early_stop=300,
                     verbosity=-1))

plot_losses(results['train_hist'], results['val_hist'])
# to_kaggle(results['model'])
# GCNN2 No bias: 0.0287

from send_help import predict_with_augmentation
# preds = predict_with_augmentation(results['model'], X)



def to_kaggle2(model):
    X_kaggle = np.load('../data/pri_in.npy')
    # X_kaggle_tensor = torch.tensor(X_kaggle, dtype=torch.float32).to(device)

    # Predict kaggle data
    y_pred_kaggle = predict_with_augmentation(model, X_kaggle)

    # Create a submission DataFrame and save to CSV
    df_submission = pd.DataFrame({'Id': np.arange(len(y_pred_kaggle)), 'Solution': y_pred_kaggle})
    print(f'saving submission [{df_submission.shape}]')
    df_submission.to_csv('submission_abekis.csv', index=False)

to_kaggle2(results['model'])



# CombinedCNN:  0.0240
# DeepSet regular: 0.0229




