import torch 
import matplotlib.pyplot as plt 

def plot_weight_distribution(model: torch.nn.Module, bins: int = 256, count_non_zero: bool = Flase):
    fig, axes = plt.subplots(figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0

    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_non_zero:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view()
                ax.hist(param_cpu, 
                bins=bins,
                density=True,
                color = 'blue',
                alpha = 0.5
                )
            else:
                ax.hist(
                    param.detch().view(-1).cpu(),
                    bins=bins,
                    density=True,
                    color = 'blue',
                    alpha = 0.5
                )
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            plot_index += 1 
    fig.suptitle("histogram of weight distribution")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()