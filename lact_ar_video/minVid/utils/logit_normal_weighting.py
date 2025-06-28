import torch

def logit_normal_integral(
    t: torch.Tensor, num_train_timesteps: int = 1000, mu: float = 0, sigma: float = 1.0
) -> torch.Tensor:
    """
    Compute the integral of the logit normal distribution over unit intervals for given time steps.
    This is a legacy version that works directly with time steps rather than normalized intervals.

    The function computes the probability mass in each unit interval [floor(t), floor(t)+1]
    under a logit normal distribution. The time steps are first normalized to [0,1] range
    before computing the integrals.

    Args:
        t (torch.Tensor): Time steps tensor of shape [batch_size,]. Values should be in range [0, num_train_timesteps-1].
        num_train_timesteps (int, optional): Total number of training timesteps. Defaults to 1000.
        mu (float, optional): Mean of the underlying normal distribution in logit space. Defaults to 0.
        sigma (float, optional): Standard deviation of the underlying normal distribution in logit space. Defaults to 1.0.

    Returns:
        torch.Tensor: Integral values of shape [batch_size,], representing the probability mass
                     in each unit interval [floor(t), floor(t)+1].

    Note:
        - The function uses numerical stability techniques (clamping) to prevent issues with logit transformation
        - The integral is computed using the error function (erf) based on the CDF of the logit normal distribution
        - This is a legacy version; consider using logit_normal_integral() for newer code
    """
    floor_t = t.floor()
    int_begin = floor_t / num_train_timesteps
    int_end = (floor_t + 1) / num_train_timesteps

    # Clip values to prevent numerical instability in logit transformation
    eps = 1e-7
    int_begin = torch.clamp(int_begin, min=eps, max=1 - eps)
    int_end = torch.clamp(int_end, min=eps, max=1 - eps)

    # Convert interval bounds to logit space
    logit_begin = torch.log(int_begin / (1 - int_begin))
    logit_end = torch.log(int_end / (1 - int_end))

    # Compute the integral using the error function
    # The CDF of logit normal is: 0.5 * (1 + erf((logit(x) - mu) / (sqrt(2) * sigma)))
    # The integral is the difference between CDF evaluated at end and begin
    integral = 0.5 * (
        torch.erf((logit_end - mu) / (torch.sqrt(torch.tensor(2.0)) * sigma))
        - torch.erf((logit_begin - mu) / (torch.sqrt(torch.tensor(2.0)) * sigma))
    )

    return integral * num_train_timesteps


def plot_logit_normal_distribution(num_train_timesteps: int = 1000, mu: float = 0.0, sigma: float = 1.0):
    """
    Plot logit normal distribution weighting function
    
    Args:
        num_train_timesteps: Number of training timesteps
        mu: Mean in logit space
        sigma: Standard deviation in logit space
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculate timesteps and corresponding loss weights
    timesteps = torch.arange(0, num_train_timesteps, device="cpu")
    loss_weights = logit_normal_integral(timesteps, num_train_timesteps, mu, sigma)
    
    # Print statistics
    total_weight = loss_weights.sum().item()
    avg_weight = total_weight / num_train_timesteps
    print(f"Total weight: {total_weight:.4f}, Average weight: {avg_weight:.4f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps.numpy(), loss_weights.numpy(), 'b-', linewidth=2, label=f'μ={mu}, σ={sigma}')
    
    # Set plot properties
    plt.xlabel('Timestep (t)')
    plt.ylabel('Loss Weight')
    plt.title('Logit Normal Weighting')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    filename = f"logit_normal_weighting_{mu}_{sigma}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close plot to free memory
    print(f"Plot saved as: {filename}")


if __name__ == "__main__":
    plot_logit_normal_distribution(num_train_timesteps=1000, mu=0.0, sigma=1.0)