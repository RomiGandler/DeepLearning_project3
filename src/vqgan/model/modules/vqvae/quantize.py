"""Vector Quantization module for VQGAN."""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class VectorQuantizer2(nn.Module):
    """
    Improved Vector Quantizer for VQ-VAE/VQGAN.
    
    Avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    
    NOTE: due to a bug the beta term was applied to the wrong term in the original.
    For backwards compatibility we use the buggy version by default, but you can
    specify legacy=False to fix it.
    """
    
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        remap=None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
    ):
        """
        Args:
            n_e: Number of embeddings (codebook size)
            e_dim: Embedding dimension
            beta: Commitment loss weight
            remap: Path to numpy file with used indices (for remapping)
            unknown_index: How to handle unknown indices ("random" or "extra")
            sane_index_shape: If True, return indices as (B, H, W) instead of flat
            legacy: Use legacy (buggy) beta application for backwards compatibility
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert not rescale_logits, "Only for interface compatible with Gumbel"
        assert not return_logits, "Only for interface compatible with Gumbel"
        
        # Reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Distances from z to embeddings e_j: (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # Compute loss for embedding
        if not self.legacy:
            loss = (
                self.beta * torch.mean((z_q.detach() - z) ** 2) +
                torch.mean((z_q - z.detach()) ** 2)
            )
        else:
            loss = (
                torch.mean((z_q.detach() - z) ** 2) +
                self.beta * torch.mean((z_q - z.detach()) ** 2)
            )

        # Preserve gradients via straight-through estimator
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """Get quantized latent vectors from indices."""
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q