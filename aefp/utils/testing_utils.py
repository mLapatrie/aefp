
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from aefp.utils.meg_utils import compute_psd_torch, compute_aec_torch


def test_ae_sampling(ae, dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    ae.eval()
    dataset.set_val()

    sample_idx = np.random.randint(0, len(dataset))
    posterior = ae.encode(dataset[sample_idx].unsqueeze(0).unsqueeze(0).to(device))
    x_hat = ae.decode(posterior.sample()).detach().cpu()
    
    fig, axes = plt.subplots(2, 1)
    # vmin and vmax as 5th and 95th percentile of the data
    vmin = min(torch.quantile(dataset[sample_idx], 0.05).item(), torch.quantile(x_hat[0][0], 0.05).item())
    vmax = max(torch.quantile(dataset[sample_idx], 0.95).item(), torch.quantile(x_hat[0][0], 0.95).item())
    axes[0].imshow(dataset[sample_idx].numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].imshow(x_hat[0][0].numpy(), aspect="auto", vmin=vmin, vmax=vmax)

    wandb.log({"test_reconstruction": [wandb.Image(plt)]})
    plt.close()

    fig, axes = plt.subplots(2, 1)
    num_timeseries = 20
    offset = 3

    for i in range(num_timeseries):
        axes[0].plot(dataset[sample_idx].numpy()[i] + i * offset, color="black", alpha=0.8, linewidth=0.5)
        axes[1].plot(x_hat[0][0].numpy()[i] + i * offset, color="red", alpha=0.8, linewidth=0.5)

    axes[0].set_title("Real")
    axes[1].set_title("Generated")

    wandb.log({"test_timeseries": [wandb.Image(plt)]})
    plt.close()
    

    # Plot random generated window
    
    z_rand = torch.randn_like(posterior.sample()).to(device)
    x_hat_rand = ae.decode(z_rand).detach().cpu()[0]

    fig, axes = plt.subplots(1, 1)
    # vmin and vmax as 5th and 95th percentile of the data
    vmin = torch.quantile(x_hat[0], 0.05).item()
    vmax = torch.quantile(x_hat[0], 0.95).item()
    axes.imshow(x_hat_rand[0].numpy(), aspect="auto", vmin=vmin, vmax=vmax)

    wandb.log({"test_generation": [wandb.Image(plt)]})
    plt.close()

    Pxx, f = compute_psd_torch(x_hat_rand, log=True, fs=150)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(f, Pxx[0, :10].mean(axis=0), color="red", label="Generated")
    ax[0].fill_between(f, Pxx[0, :10].mean(axis=0) - Pxx[0, :10].std(axis=0), 
                      Pxx[0, :10].mean(axis=0) + Pxx[0, :10].std(axis=0), 
                      color="red", alpha=0.2)
    ax[1].plot(f, Pxx[0, -10:].mean(axis=0), color="red", label="Generated")
    ax[1].fill_between(f, Pxx[0, -10:].mean(axis=0) - Pxx[0, -10:].std(axis=0), 
                      Pxx[0, -10:].mean(axis=0) + Pxx[0, -10:].std(axis=0), 
                      color="red", alpha=0.2)
    
    ax[0].set_title(f"PSD of first 10 ROIs")
    ax[1].set_title(f"PSD of last 10 ROIs")
    ax[0].set_xlabel("Frequency (Hz)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[0].legend()
    ax[1].legend()
    plt.suptitle(f"PSD of generated window")
    wandb.log({"test_psd_generation": [wandb.Image(plt)]})
    plt.close()


    # Plot psd
    x = dataset[sample_idx].unsqueeze(0)
    Pxx, f = compute_psd_torch(x, log=True, fs=150)
    Pxx_hat, f = compute_psd_torch(x_hat.squeeze(1), log=True, fs=150)

    mean_real_psd_network1 = Pxx[0, :10].mean(axis=0)
    std_real_psd_network1 = Pxx[0, :10].std(axis=0)

    mean_real_psd_network2 = Pxx[0, -10:].mean(axis=0)
    std_real_psd_network2 = Pxx[0, -10:].std(axis=0)

    mean_generated_psd_network1 = Pxx_hat[0, :10].mean(axis=0)
    std_generated_psd_network1 = Pxx_hat[0, :10].std(axis=0)

    mean_generated_psd_network2 = Pxx_hat[0, -10:].mean(axis=0)
    std_generated_psd_network2 = Pxx_hat[0, -10:].std(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(f, mean_real_psd_network1, color="black", label="Real")
    ax[0].fill_between(f, mean_real_psd_network1 - std_real_psd_network1, mean_real_psd_network1 + std_real_psd_network1, color="black", alpha=0.2)

    ax[0].plot(f, mean_generated_psd_network1, color="red", label="Generated")
    ax[0].fill_between(f, mean_generated_psd_network1 - std_generated_psd_network1, mean_generated_psd_network1 + std_generated_psd_network1, color="red", alpha=0.2)

    ax[1].plot(f, mean_real_psd_network2, color="black", label="Real")
    ax[1].fill_between(f, mean_real_psd_network2 - std_real_psd_network2, mean_real_psd_network2 + std_real_psd_network2, color="black", alpha=0.2)

    ax[1].plot(f, mean_generated_psd_network2, color="red", label="Generated")
    ax[1].fill_between(f, mean_generated_psd_network2 - std_generated_psd_network2, mean_generated_psd_network2 + std_generated_psd_network2, color="red", alpha=0.2)

    ax[0].set_title(f"PSD of first 10 ROIs")
    ax[1].set_title(f"PSD of last 10 ROIs")

    ax[0].set_xlabel("Frequency (Hz)")
    ax[1].set_xlabel("Frequency (Hz)")

    ax[0].legend()
    ax[1].legend()

    plt.suptitle(f"PSD of real and generated windows")

    wandb.log({"test_psd": [wandb.Image(plt)]})
    plt.close()

    # plot fc
    x = dataset[sample_idx].unsqueeze(0)

    aec = compute_aec_torch(x)
    aec_hat = compute_aec_torch(x_hat.squeeze(1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(aec[0], vmin=0, vmax=1)
    ax[1].imshow(aec_hat[0], vmin=0, vmax=1)

    ax[0].set_title("Real")
    ax[1].set_title("Generated")

    plt.suptitle("AEC of real and generated windows")

    wandb.log({"test_aec": [wandb.Image(plt)]})
    plt.close()


    ae.train()
    dataset.set_train()