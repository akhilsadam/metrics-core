import torch
import torch.nn as nn
import math


class Derivative(nn.Module):
    """Spectral derivative operator for computing energy/enstrophy spectra."""

    def __init__(self, shape=(512, 512), L=(1, 1)):
        super().__init__()

        self.Ny, self.Nx = shape[-2], shape[-1]
        self.Lx, self.Ly = 2 * torch.pi * L[-2], 2 * torch.pi * L[-1]
        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny
        self.register_buffer("x", torch.arange(-self.Lx / 2, self.Lx / 2, dx))
        self.register_buffer("y", torch.arange(-self.Ly / 2, self.Ly / 2, dy))

        # Number of wavenumber components
        self.dk = int(self.Nx / 2 + 1)

        # Pure wavenumbers
        self.register_buffer(
            "ky",
            torch.reshape((torch.fft.fftfreq(self.Ny, self.Ly / (self.Ny * 2 * torch.pi))), (self.Ny, 1))[
                None, None, :, :
            ],
        )

        self.register_buffer(
            "kx",
            torch.reshape((torch.fft.rfftfreq(self.Nx, self.Lx / (self.Nx * 2 * torch.pi))), (1, self.dk))[
                None, None, :, :
            ],
        )

        ksq = self.kx**2 + self.ky**2
        irsq = 1 / ksq
        irsq[..., 0, 0] = 0.0
        self.register_buffer("ksq", ksq)
        self.register_buffer("irsq", irsq)

        self.register_buffer("dx", 1j * self.kx)
        self.register_buffer("dy", 1j * self.ky)

    def _dx(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.dx)

    def _dy(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.dy)

    def phi(self, w):
        return torch.fft.irfft2(torch.fft.rfft2(w) * self.irsq)

    def uv(self, w):
        phi = self.phi(w)
        return (-self._dy(phi), self._dx(phi))

    def adv(self, w, w_nz):
        u, v = self.uv(w_nz)
        wx = self._dx(w)
        wy = self._dy(w)
        return -u * wx - v * wy

    def spectrum(self, x):
        """Compute energy and enstrophy spectra."""
        with torch.no_grad():
            qh = torch.fft.rfft2(x)
            ph = -qh * self.irsq
            uh = -self.dy * ph
            vh = self.dx * ph

            zh = qh.real**2 + qh.imag**2
            eh = uh.real**2 + uh.imag**2 + vh.real**2 + vh.imag**2

            zh = zh / zh.sum()
            eh = eh / eh.sum()

            y = [eh.detach().cpu()[0, 0], zh.detach().cpu()[0, 0]]

            K = torch.sqrt(self.ksq)[0, 0, :, :].detach().cpu()
            d = 0.5
            k = torch.arange(1, int(K[-1, -1]) - 1)
            m = torch.zeros(k.size())

            e = [torch.zeros(k.size()) for _ in range(len(y))]
            for ik in range(len(k)):
                n = k[ik]
                i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
                m[ik] = i[0].numel()
                for j, yj in enumerate(y):
                    e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
        return k, e

    def plot_spectra(self, x, axes, timestep, label="gen", set_lim=True):
        """Plot energy and enstrophy spectra."""
        k, (ek, zk) = self.spectrum(x)

        # Energy spectrum
        axes[0].loglog(k, ek, label=label)

        # Enstrophy spectrum
        axes[1].loglog(k, zk, label=label)

        if set_lim:
            axes[0].set_xlim(1, 0.75 * max(k))
            axes[0].set_ylim(10 ** (-12), 10 ** (-0.5))
            axes[0].set_aspect(0.20, adjustable="box")
            axes[0].set_title(f"Energy Spectrum at Epoch {timestep}")
            axes[0].set_xlabel("Wavenumber k")
            axes[0].set_ylabel("Energy Spectrum E(k)")

            axes[1].set_xlim(1, 0.75 * max(k))
            axes[1].set_ylim(10 ** (-12), 10 ** (-0.5))
            axes[1].set_aspect(0.20, adjustable="box")
            axes[1].set_title(f"Enstrophy Spectrum at Epoch {timestep}")
            axes[1].set_xlabel("Wavenumber k")
            axes[1].set_ylabel("Enstrophy Spectrum Z(k)")
