import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Rocket(nn.Module):
    
    def __init__(self, 
                 sequence_length: int, 
                 kernel_size: int, 
                 kernel_num: int, 
                 in_channels: int,
                 device: torch.device, 
                 random_state: int=42,
                 init_type: str='sparse',
                 sparsity: float=0.8,
                 sigmoid_coeff: float = 50,
                 pretrain: bool = False):
        """
        Args:
            sequence_length (int): Length of the input time series
            kernel_size (int): Size of the 1D kernel
            kernel_num (int): Number of kernels
            in_channels (int): Number of input channels
            device (str): Device to run the model
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            init_type (str, optional): Weight initialization (random or sparse). Default is 'sparse'
            sparsity (float, optional): Sparsity of the kernel. Default is 0.8
            sigmoid_coeff (float, optional): Coefficient for sigmoid activation. Default is 50
            pretrain (bool, optional): Whether to use a pre-trained model. Defaults to False.
        """
        super(Rocket, self).__init__()
        
        self.sequence_length = sequence_length 
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.in_channels = in_channels
        self.device = device
        self.random_state = random_state
        self.pretrain = pretrain
        self.init_type = init_type
        self.sparsity = sparsity
        self.sigmoid_coeff = sigmoid_coeff
        self.dilations = self.generate_dilations()
        self.adp_pooling = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.Sigmoid()
        
        if not self.pretrain:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            self.init_weights(self.init_type)
    
    def init_weights(self, init_type: str):
        """
        Initialize weights.
        
        Args:
            init_type (str): Initialization type ('sparse' or 'random').
        """
        if self.random_state is None:  # Set random seed if random_state is provided
            self.random_state = 42
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        if init_type == 'sparse':
            self.kernel_tensor = nn.Parameter(
                self.generate_sparse_kernels().to(self.device), requires_grad=False
            )
        elif init_type == 'random':
            self.kernel_tensor = nn.Parameter(
                self.generate_random_kernels().to(self.device), requires_grad=False
            )
        else:
            raise ValueError(f'Invalid init_type: {init_type}')
        
    def load_pretrained_params(self, kernel_tensor: torch.Tensor):
        """
        Load a pre-trained model.

        Args:
            kernel_tensor (torch.Tensor): Tensor of kernels.
        """
        self.kernel_tensor = kernel_tensor
        
    def get_params(self) -> list:
        """
        Get parameters of the model.

        Returns:
            List of params.
        """
        return [self.sequence_length, self.kernel_num, self.kernel_size, self.sparsity, self.sigmoid_coeff]
    
    def generate_dilations(self) -> torch.Tensor:
        """
        Generate dilations for 1D convolution

        Args:
            None

        Returns:
            torch.Tensor: Tensor of dilations 
        """
        dilation_max = np.log2((self.sequence_length - 1) / (self.kernel_size - 1))
        dilation_exp = np.arange(0, np.ceil(dilation_max))
        dilation_exp = np.append(dilation_exp, dilation_max)
        dilations = np.unique(np.floor(2 ** dilation_exp))
        dilations = dilations.astype(int)
        return dilations
    
    def generate_sparse_kernels(self):
        """
        Generate a batch of 1D tensors with sparsity
        
        Args:
            None
            
        Returns:
            torch.Tensor: Batch of 1D tensors with the desired properties.
        """
        assert 0 <= self.sparsity <= 1, "Sparsity must be between 0 and 1"
        
        torch.manual_seed(self.random_state)
        
        batch = []
        for _ in range(self.kernel_num):
            # Create a kernel for each input channel
            channel_kernels = []
            for c in range(self.in_channels):
                # Generate random kernel for this channel
                kernel = torch.randn(self.kernel_size, dtype=torch.float32)
                
                # Apply sparsity
                num_zeros = int(self.sparsity * self.kernel_size)
                if num_zeros > 0:
                    zero_indices = torch.randperm(self.kernel_size)[:num_zeros]
                    kernel[zero_indices] = 0
                
                channel_kernels.append(kernel)
            
            # Stack the channel kernels
            kernel_tensor = torch.stack(channel_kernels)
            batch.append(kernel_tensor)
        
        # Final shape: (kernel_num, in_channels, kernel_size)
        return torch.stack(batch)
        
    def generate_random_kernels(self) -> torch.Tensor:
        """
        Generate multiple random 1D kernels

        Args:
            None

        Returns:
            torch.Tensor: Tensor of random kernels in the shape of (num_kernels, in_channels, kernel_size)
        """
        kernels = []
        for _ in range(self.kernel_num):
            kernel = torch.randn((self.in_channels, self.kernel_size), dtype=torch.float32)
            kernels.append(kernel)
        
        kernels_tensor = torch.stack(kernels)
        
        return kernels_tensor
    
    def normalize(self, x, eps=1e-8):
        # x shape: (B, C, L) - batch, channels, length
        mean = x.mean(dim=(0, 2), keepdim=True)  # one μ per channel
        std = x.std(dim=(0, 2), keepdim=True)
        return (x - mean) / (std + eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor in the shape of (batch_size, in_channels, sequence_length)

        Returns:
            ppvs (torch.Tensor): the portion of positive values in the shape of (batch_size, num_kernels * num_dilations)
        """
        ppvs = []
        for dilation in self.dilations:
            # Calculate padding to maintain the output length
            padding = (self.kernel_size // 2) * dilation
            
            # 1D convolution instead of 2D
            f_x = F.conv1d(x, self.kernel_tensor, dilation=dilation, padding=padding)
            
            acti = self.activation(self.sigmoid_coeff * f_x)
            
            # 1D adaptive pooling instead of 2D
            pooling = self.adp_pooling(acti)
            
            ppvs.append(pooling.view(x.shape[0], -1))

        return torch.cat(ppvs, dim=1)


if __name__ == '__main__':
    # Example usage
    # Create a model instance
    sequence_length = 256
    kernel_size = 11
    kernel_num = 512
    in_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Rocket(
        sequence_length=sequence_length,
        kernel_size=kernel_size,
        kernel_num=kernel_num,
        in_channels=in_channels,
        device=device,
        # init_type='random'
    )
    
    # Generate a sample input (batch_size, in_channels, sequence_length)
    sample_input = torch.randn(16, in_channels, sequence_length).to(device)
    
    # Forward pass
    output = model(sample_input)
    print(f"Output shape: {output.shape}")