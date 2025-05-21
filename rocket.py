"""Code for RandOm Convolutional KErnel Transformation implemented with PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


def generate_kernels(n_kernels, n_timestamps, kernel_sizes, seed):
    """Generate the kernels.

    Parameters
    ----------
    n_kernels : int
        Number of kernels

    n_timestamps : int
        Number of timestamps

    kernel_sizes : array
        Possible sizes for the kernels.

    seed : int
        Seed for the random number generator.

    Returns
    -------
    weights : torch.Tensor, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    lengths : torch.Tensor, shape = (n_kernels,)
        Length of each kernel.

    biases : torch.Tensor, shape = (n_kernels,)
        Bias of each kernel.

    dilations : torch.Tensor, shape = (n_kernels,)
        Dilation of each kernel.

    paddings : torch.Tensor, shape = (n_kernels,)
        Padding of each kernel.
    """
    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)  # Used for numpy random choice

    # Select random kernel lengths
    lengths = torch.tensor(np.random.choice(kernel_sizes, size=n_kernels), dtype=torch.int64)
    
    # Generate random weights for all kernels
    max_kernel_size = int(torch.max(lengths).item())
    weights = torch.zeros((n_kernels, max_kernel_size))
    
    # Generate random weights and center them (subtract mean)
    for i in range(n_kernels):
        length = lengths[i].item()
        kernel_weights = torch.randn(length)
        weights[i, :length] = kernel_weights - kernel_weights.mean()
    
    # Generate random biases
    biases = torch.rand(n_kernels) * 2 - 1  # Uniform between -1 and 1
    
    # Calculate dilations
    upper_bounds = torch.log2(torch.floor_divide(
        torch.tensor(n_timestamps - 1, dtype=torch.float32), 
        (lengths.float() - 1)
    ))
    
    powers = torch.zeros(n_kernels)
    for i in range(n_kernels):
        powers[i] = torch.rand(1).item() * upper_bounds[i].item()
    
    dilations = torch.floor(torch.pow(2, powers)).to(torch.int64)
    
    # Calculate paddings
    paddings = torch.zeros(n_kernels, dtype=torch.int64)
    padding_cond = torch.randint(0, 2, (n_kernels,)).bool()
    paddings[padding_cond] = torch.floor_divide(
        (lengths[padding_cond] - 1) * dilations[padding_cond], 2
    )
    
    return weights, lengths, biases, dilations, paddings


def apply_kernels_batch(X, weights, lengths, biases, dilations, paddings, device=None):
    """Apply all kernels to a batch of time series using PyTorch operations.
    
    Parameters
    ----------
    X : torch.Tensor, shape = (n_samples, n_timestamps)
        Input data.
        
    weights : torch.Tensor, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels.
        
    lengths : torch.Tensor, shape = (n_kernels,)
        Length of each kernel.
        
    biases : torch.Tensor, shape = (n_kernels,)
        Bias of each kernel.
        
    dilations : torch.Tensor, shape = (n_kernels,)
        Dilation of each kernel.
        
    paddings : torch.Tensor, shape = (n_kernels,)
        Padding of each kernel.
        
    device : str or torch.device, optional
        Device to run the computations on.
        
    Returns
    -------
    X_new : torch.Tensor, shape = (n_samples, 2 * n_kernels)
        Extracted features using all the kernels.
    """
    if device is not None:
        X = X.to(device)
        weights = weights.to(device)
        lengths = lengths.to(device)
        biases = biases.to(device)
        dilations = dilations.to(device)
        paddings = paddings.to(device)
    
    n_samples, n_timestamps = X.shape
    n_kernels = lengths.size(0)
    
    # Prepare output tensor
    X_new = torch.empty((n_samples, 2 * n_kernels), device=X.device)
    
    for k in range(n_kernels):
        length = lengths[k].item()
        weight = weights[k, :length]
        bias = biases[k].item()
        dilation = dilations[k].item()
        padding = paddings[k].item()
    
        # Apply convolution using PyTorch's conv1d
        # X shape: (n_samples, 1, n_timestamps)
        # Add padding if needed
        x_pad = torch.nn.functional.pad(X.unsqueeze(1), (padding, padding))
        
        # Reshape weight for conv1d: (out_channels, in_channels, kernel_size)
        weight_reshaped = weight.view(1, 1, -1)
        
        # Apply convolution with dilation
        x_conv = torch.nn.functional.conv1d(
            x_pad, 
            weight_reshaped, 
            bias=None,  # We'll add bias manually after convolution
            stride=1, 
            padding=0, 
            dilation=dilation
        ).squeeze(1)
        
        # Add bias
        x_conv = x_conv + bias
        
        # Extract features: maximum and proportion of positive values
        X_new[:, 2 * k] = torch.max(x_conv, dim=1)[0]
        # X_new[:, 2 * k + 1] = torch.mean((x_conv > 0).float(), dim=1)
        X_new[:, 2 * k + 1] = torch.mean(torch.sigmoid(50 * x_conv), dim=1)
    return X_new


class PyTorchROCKET(BaseEstimator):
    """PyTorch implementation of RandOm Convolutional KErnel Transformation.

    This algorithm randomly generates a great variety of convolutional kernels
    and extracts two features for each convolution: the maximum and the
    proportion of positive values. This implementation uses PyTorch to support
    GPU acceleration.

    Parameters
    ----------
    n_kernels : int (default = 10000)
        Number of kernels.

    kernel_sizes : array-like (default = (7, 9, 11))
        The possible sizes of the kernels.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data.
        
    device : str or torch.device or None (default = None)
        Device to use for computation. If None, uses CUDA if available, otherwise CPU.

    Attributes
    ----------
    weights_ : torch.Tensor, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    length_ : torch.Tensor, shape = (n_kernels,)
        Length of each kernel.

    bias_ : torch.Tensor, shape = (n_kernels,)
        Bias of each kernel.

    dilation_ : torch.Tensor, shape = (n_kernels,)
        Dilation of each kernel.

    padding_ : torch.Tensor, shape = (n_kernels,)
        Padding of each kernel.
        
    device_ : torch.device
        The device used for computation.

    References
    ----------
    .. [1] A. Dempster, F. Petitjean and G. I. Webb, "ROCKET: Exceptionally
           fast and accurate time series classification using random
           convolutional kernels". https://arxiv.org/abs/1910.13051.
    """
    
    def __init__(self, n_kernels=10000, kernel_sizes=(7, 9, 11), random_state=None, device=None):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state
        self.device = device

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : None or array-like, shape = (n_samples,)
            Class labels for each data sample. Ignored.

        Returns
        -------
        self : object
        """
        # Check input data
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        
        # Set device
        if self.device is None:
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = torch.device(self.device)
        
        # Check parameters
        kernel_sizes, seed = self._check_params(n_timestamps)
        
        # Generate the kernels
        weights, lengths, biases, dilations, paddings = generate_kernels(
            self.n_kernels, n_timestamps, kernel_sizes, seed
        )
        
        # Store parameters
        self.weights_ = weights
        self.length_ = lengths
        self.bias_ = biases
        self.dilation_ = dilations
        self.padding_ = paddings
        
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X_new : numpy.ndarray, shape = (n_samples, 2 * n_kernels)
            Extracted features from the kernels.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, ['weights_', 'length_', 'bias_', 'dilation_', 'padding_'])
        
        # Check input data
        X = check_array(X, dtype='float64')
        
        # Convert to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Apply kernels to extract features
        X_new = apply_kernels_batch(
            X_tensor, 
            self.weights_, 
            self.length_, 
            self.bias_, 
            self.dilation_, 
            self.padding_,
            device=self.device_
        )
        
        # Convert back to numpy array
        return X_new

    def _check_params(self, n_timestamps):
        """Check parameters and return validated kernel_sizes and random seed."""
        if not isinstance(self.n_kernels, (int, np.integer)):
            raise TypeError("'n_kernels' must be an integer (got {})."
                            .format(self.n_kernels))

        if not isinstance(self.kernel_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'kernel_sizes' must be a list, a tuple or "
                            "an array (got {}).".format(self.kernel_sizes))
                            
        kernel_sizes = check_array(self.kernel_sizes, ensure_2d=False,
                                  dtype='int64', accept_large_sparse=False)
                                  
        if not np.all(1 <= kernel_sizes):
            raise ValueError("All the values in 'kernel_sizes' must be "
                            "greater than or equal to 1 ({} < 1)."
                            .format(kernel_sizes.min()))
                            
        if not np.all(kernel_sizes <= n_timestamps):
            raise ValueError("All the values in 'kernel_sizes' must be lower "
                            "than or equal to 'n_timestamps' ({} > {})."
                            .format(kernel_sizes.max(), n_timestamps))

        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return kernel_sizes, seed


class PytorchRidgeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PytorchRidgeClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)  # Output shape: (batch_size, num_classes)


def train(model, X_train, y_train, l2_reg=1.0, epochs=100, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # # One-hot encode labels: shape (N, C)
    # y_onehot = torch.zeros(X_train.size(0), model.linear.out_features, device=X_train.device)
    # y_onehot.scatter_(1, y_train.view(-1, 1), 1)
    # # Map {0,1} to {-1, 1}
    # y_target = 2 * y_onehot - 1

    # for epoch in range(epochs):
    #     model.train()
    #     optimizer.zero_grad()

    #     outputs = model(X_train)  # Shape: (N, C)
    #     loss = torch.mean((outputs - y_target)**2)  # Squared loss per class
    #     # L2 regularization (weight decay)
    #     l2_penalty = sum(torch.norm(param)**2 for param in model.parameters())
    #     loss += l2_reg * l2_penalty
    #     loss.backward()
    #     optimizer.step()
    # return model
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)  # Shape: (N, C)
        loss = criterion(outputs, y_train)
        l2_penalty = sum(torch.norm(param)**2 for param in model.parameters())
        loss += l2_reg * l2_penalty
        loss.backward()
        optimizer.step()
    return model


def predict(model, X):
    with torch.no_grad():
        logits = model(X)
        return torch.argmax(logits, dim=1)  # Pick class with highest score