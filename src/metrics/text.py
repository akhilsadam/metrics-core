import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math

from matplotlib import pyplot as plt

def token_accuracy(tks, rpns):
    acc = 0.0
    num_mse = 0.0
    num_rmse = 0.0
    nums = 0.0
    total = 0.0

    for tk_sent, dec_sent in zip(tks, rpns):
        tokens = tk_sent.split(' ')
        dec_tokens = dec_sent.split(' ')

        min_len = min(len(tokens), len(dec_tokens))

        # Compare overlapping tokens
        for tk, dec_tk in zip(tokens[:min_len], dec_tokens[:min_len]):
            total += 1.0

            if tk == dec_tk:
                acc += 1.0

            if '.' in tk and '.' in dec_tk:
                num_mse += (float(tk) - float(dec_tk)) ** 2
                nums += 1.0

        # Count length mismatch only once per sentence
        if len(tokens) != len(dec_tokens):
            total += 1.0

    acc = acc / total if total > 0 else 0.0
    if nums > 0:
        num_rmse = math.sqrt(num_mse / nums) # rmse
    
    return acc, num_rmse


def metrics(net, i, batch, dirs):
    tks = net.detokenize(*net.tokenize(batch))
    encoded = net.encode(batch)
    rpns = net.decode(encoded)
    samples = net.decode(net.sample(encoded))
    
    acc, num_rmse = token_accuracy(tks, rpns)

    d = {'in':batch, 'out':rpns, 'sampled': samples, 'token_check':tks, 'token_accuracy':acc, 'numerical_rmse':num_rmse}
    
    with open(os.path.join(dirs[0], f'rpn_gen_{i:04d}.json'), 'w') as f:
        json.dump(d, f, indent=4)

def generation(net, loader, dirs):
    for i, batch in enumerate(loader):
        metrics(net, i, batch, dirs)
        break
    
def inverse_metrics(net, i, batch, d, dirs):
    rpns, seq = batch
    encoding_hat, recon_rpns, losses, forward = net.inverse_solver(seq)
    encodings = net.encode_LLM(rpns)

    sem_hat = net.llm.crpn.gen.semantic(encoding_hat)
    sem_true = net.llm.crpn.gen.semantic(encodings)
    
    dloss_hat = forward(encoding_hat).item()
    dloss_true = forward(encodings).item()

    relMSE = F.mse_loss(encoding_hat, encodings).item() / (torch.mean(encodings**2).item() + 1e-8)
    relMSE_sem = F.mse_loss(sem_hat, sem_true).item() / (torch.mean(sem_true**2).item() + 1e-8)
    print(f"Sample {i}: Relative MSE in latent space (enc, sem): {relMSE:.6f}, {relMSE_sem:.6f}, with loss ratio (hat/true): {dloss_hat / dloss_true:.6f}") 


    plt.plot(losses['d_loss'], label='Diffusion Loss')
    plt.plot(losses['p_loss'], label='Perceptual Loss')
    plt.title(f'Optimization Loss Curve for Sample {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(dirs[0], f'opt_loss_curve_{i:04d}.png'))
    plt.close()
    
    acc, num_rmse = token_accuracy(rpns, recon_rpns)
    
    d[i] = {'in':rpns, 'out':recon_rpns, 'token_accuracy':acc, 'numerical_rmse':num_rmse,
             'relative_vect_embd_mse': relMSE, 'relative_sem_embd_mse': relMSE_sem, 'loss_ratio': dloss_hat / dloss_true}
    

def inverse_metrics_all(net, loader, dirs):
    d = {}
    n = len(loader)
    _range = list(range(0, n, max(1, n//4)))
    print(f"Selected indices for inversion: {_range}")
    for i, batch in enumerate(loader):
        if i not in _range:
            continue
        inverse_metrics(net, i, batch, d, dirs)
            
    with open(os.path.join(dirs[0], f'inverse_gen.json'), 'w') as f:
        json.dump(d, f, indent=4)