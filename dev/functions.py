import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split

import numpy as np

np.random.seed(5)
torch.manual_seed(5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running PyTorch on: {device}")


class SpongeData(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(
            device)  # pass the data/tensors onto CPU or CUDA and unsqueeze
        self.targets = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def to_torch(X, y, split=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    train = SpongeData(X_train, y_train)
    test = SpongeData(X_test, y_test)

    return train, test


class RMSPE(nn.Module):
    def __init__(self):
        super(RMSPE, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2))


class EarlyStopping:
    def __init__(self, tolerance=30):
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False

    def __call__(self, best_loss, val_loss):
        if (self.tolerance > 0) and (best_loss < val_loss):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0


def plot_losses_and_predictions(train, test, y_test=None, y_pred=None, lab1='Train', lab2='Test'):
    if isinstance(train, torch.Tensor):
        train = train.cpu().numpy()
    if isinstance(test, torch.Tensor):
        test = test.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    # plot_losses from the lab notebooks
    axs[0].semilogy(train, label=lab1, marker='.')
    axs[0].semilogy(test, label=lab2, marker='.')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss (MAPE)')
    axs[0].legend()
    axs[0].set_title('Training and Testing Losses')

    if y_test is not None:
        # actual vs predicted scatter plot
        axs[1].scatter(y_test, y_pred, label='Predictions')
        axs[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='ideal fit')
        axs[1].set_xlabel('Actual')
        axs[1].set_ylabel('Predicted')
        axs[1].legend()
        axs[1].set_title('Actual vs Predicted $A_{max}$')
    else:
        axs[1].set_visible(False)

    plt.tight_layout()
    plt.show()


from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # for the beautiful progress bar


def train(train_dataset, val_dataset, model, loss_fn=RMSPE(), num_epochs=50, batch_size=128, learning_rate=1e-3,
          early_stop=50, verbosity=-1, use_pbar=True):
    # prepare the data - setting num_workers>0 in Jupyter breaks everything...
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Initialize variables of interest and history ----- #
    train_hist, val_hist = [], []

    best_loss = np.inf
    best_weights = None

    # ----- Initialize the methods ------- #
    early_stopper = EarlyStopping(tolerance=early_stop)  # initialize the early stopping logic
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # ADAM optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=10)  # actively updating the leatning rate

    # ----- Get the model ready ----- #
    model.to(device)  # pass it onto CUDA or CPU
    model.train()
    # scaler = GradScaler()  # CUDA automatic gradient scaling

    # ----- Training ----- #
    tqdm_pbar = tqdm(range(num_epochs), desc='Initializing') if use_pbar else range(num_epochs)  # progress bar

    for epoch in range(num_epochs):
        train_loss, val_loss = 0.0, 0.0

        for i, (X_batch, y_batch) in enumerate(train_loader, 0):  # cite to the pytorch documentation
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # pass the tensors to cuda

            optimizer.zero_grad()  # main forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # loss = loss_fn(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ----- Validation ----- #
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # loss = loss_fn(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item()
        model.train()

        # Get the losses & store
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        scheduler.step(val_loss)  # update learning rate

        # ------- EarlyStopping -------#
        if val_loss < best_loss:  # getting the best model alongside early stopping
            best_loss = val_loss
            best_weights = model.state_dict()

        early_stopper(best_loss, val_loss)
        if early_stopper.early_stop:
            print(f'Early stopping at epoch {epoch}, validation loss: {val_loss:.8f}')
            break

        # -------- Printing --------- #
        if (verbosity > 0) and (epoch % verbosity == 0):
            print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")

        if use_pbar:
            tqdm_pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.8f},"
                f" Val Loss: {val_loss:.8f}, Early Stop: {early_stopper.counter}")
            tqdm_pbar.update(1)

    # ------- Returning best model --------
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f'Returning the best model with validation loss: {best_loss:.4f}')

    # return (best) model parameters alongside train & val history
    return {
        'model': model,
        'train_hist': train_hist,
        'val_hist': val_hist,
    }


def predict(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        inputs = torch.tensor(input_data.clone().detach(), dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension and move to GPU
        outputs = model(inputs)
        predictions = outputs.squeeze().cpu().numpy()  # Move to CPU and convert to numpy
    return predictions



def evaluate_model(model, X, y):
    model.eval()

    # Evaluation step to calculate MAPE and MSE
    criterion = nn.MSELoss()
    rmspe = RMSPE()

    with torch.no_grad():
        outputs = model(X)
        mape = rmspe(outputs, y).item()

    print(f'Final RMSPE: {mape:6}%')


def plot_losses(train, test):
    plt.semilogy(train, label='Train', marker='.')
    plt.semilogy(test, label='Test', marker='.')
    plt.legend()
    plt.show()


def to_kaggle(model):
    X_kaggle = np.load('../data/pri_in.npy')
    X_kaggle_tensor = torch.tensor(X_kaggle, dtype=torch.float32).to(device)

    # Predict kaggle data
    y_pred_kaggle = predict(model, X_kaggle_tensor).flatten()

    # Create a submission DataFrame and save to CSV
    df_submission = pd.DataFrame({'Id': np.arange(len(y_pred_kaggle)), 'Solution': y_pred_kaggle})
    print(f'saving submission [{df_submission.shape}]')
    df_submission.to_csv('submission_abekis.csv', index=False)


X = np.load("../data/pub_input.npy")
y = np.sqrt(np.load("../data/pub_out.npy"))

