import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jpcm.draw as draw
cmap = draw.cmap

from einops import rearrange

from torchvision.utils import save_image

def plot(recon, output_dir, name, nrow=4):
    if recon.shape[1] == 1:
        cmap = draw.cmap
        recon = recon.squeeze(1)
        # normalize to [0,1] for colormap
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        # cmap returns RGBA, take RGB and convert to tensor in CHW format
        recon = torch.from_numpy(cmap(recon)[...,:3]).permute(0, 3, 1, 2).float()
        save_image(recon[:8], os.path.join(output_dir, name), nrow=nrow)
    else:
        raise NotImplementedError("Plotting only implemented for single-channel images")

def rplot(recon, output_dir, name):
    # recon is B Y C H W, make grid of recon, error, input
    recon = rearrange(recon, 'b y c h w -> b c h (y w)')
    plot(recon[:8], output_dir, name, nrow=8)

def reconstruction_step(x, net, level=0.1):
    t = torch.tensor(level, device=x.device)
    n = net.noise(x)
    x_hat = net.denoise(net.mix(x, n, t), t)
    # B Y C H W
    loss = F.mse_loss(x_hat, x) / F.mse_loss(x, x.mean(dim=(-2,-1), keepdim=True))
    stack = torch.stack([x, x_hat, x_hat - x], dim=1)  # B Y C H W
    return loss, stack

def reconstruction(net, loader, dirs, level=0.1):
    with torch.no_grad():
        n = len(loader)
        results = []
        loss = 0.0
        for batch in loader:
            x = batch[0].to(next(net.parameters()).device)
            zloss, stack = reconstruction_step(x, net, level)
            results.append(stack.detach().cpu())
            loss += zloss.item() / n
        results = torch.cat(results, dim=0)
        torch.save(results, os.path.join(dirs[1], "reconstructions.pt"))
        rplot(results[:8], dirs[0], "reconstructions.png")
    
def generation(net, loader, dirs, level=0.1, warmup=3, n_samples=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    x = next(iter(loader))[0].to(next(net.parameters()).device)
    
    # with profiler.profile(
    #     activities=[
    #         profiler.ProfilerActivity.CPU,
    #         profiler.ProfilerActivity.CUDA
    #     ],
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True
    # ) as prof:
    
    times = []
    data = []

    steps = (n_samples // x.shape[0]) + 1
    with torch.no_grad():
        
        torch.cuda.empty_cache()
        
        for _ in range(warmup):
            _ = net.gen(x)
        
        torch.cuda.synchronize()
        
        for _ in range(steps):
            start.record()
            x_hat = net.gen(x)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  #
            
            data.append(x_hat.detach())
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        data = torch.cat(data, dim=0).cpu()
        torch.save(data, os.path.join(dirs[1], "generated_samples.pt"))
        
        times = np.array(times)
        min_time = times.min()
        max_time = times.max()
        median_time = np.median(times)
        net.log('gen_time_min', min_time)
        net.log('gen_time_max', max_time)
        net.log('gen_time', median_time)
        
        with open(os.path.join(dirs[1], "generation_times_summary.txt"), "w") as f:
            f.write(f"Min time: {min_time:.2f} ms\n")
            f.write(f"Max time: {max_time:.2f} ms\n")
            f.write(f"Median time: {median_time:.2f} ms\n")
        np.save(os.path.join(dirs[1], "generation_times.npy"), times)
        
        plot(torch.from_numpy(data[:8]), dirs[0], "generated_samples.png")      
