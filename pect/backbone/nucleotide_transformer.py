
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .cache import CacheWriter


def _pre_attention_ln(layer: nn.Module) -> nn.Module:
    """Locate the frozen pre-attention LayerNorm LN_l of an ESM-style block."""
    att = getattr(layer, "attention", None)
    if att is not None and isinstance(getattr(att, "LayerNorm", None), nn.LayerNorm):
        return att.LayerNorm
    for name, mod in layer.named_modules():  # fallback for refactored implementations
        if isinstance(mod, nn.LayerNorm) and "output" not in name:
            return mod
    raise RuntimeError("Could not find the pre-attention LayerNorm of a transformer block")


def _encoder_layers(model: nn.Module):
    for attr in ("esm", "base_model", "model"):
        enc = getattr(model, attr, None)
        if enc is not None and hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            return enc, enc.encoder.layer
    raise RuntimeError("Unsupported model: cannot find encoder.layer")


@torch.no_grad()
def cache_nucleotide_transformer(
    genes: Sequence[str],
    seqs: Sequence[str],
    out_dir: str,
    model_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    blocks: List[int] = (21, 22, 23, 24),
    device: str = "cuda",
    batch_size: int = 8,
    fp16: bool = True,
    logger=None,
):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(dev)
    if fp16 and dev.type == "cuda":
        model.half()
    encoder, layers = _encoder_layers(model)
    K_b = len(layers)
    blocks = list(blocks)
    assert all(1 <= l <= K_b for l in blocks), f"blocks must lie in 1..{K_b}"
    d_s = model.config.hidden_size

    n_tokens = max(len(tok(s)["input_ids"]) for s in seqs)
    if logger:
        logger.info(f"{model_name}: K_b={K_b}, d_s={d_s}, tokens per gene <= {n_tokens}")

    cap = {}
    hooks = []
    for l in blocks:
        ln = _pre_attention_ln(layers[l - 1])
        hooks.append(ln.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(("x", l), o)))
    for l in sorted(set(blocks) | {K_b}):
        hooks.append(layers[l - 1].register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(("z", l), o[0] if isinstance(o, tuple) else o)))

    writer = CacheWriter(out_dir, genes, n_tokens, d_s, blocks, K_b, model_name)
    for s in range(0, len(seqs), batch_size):
        batch = list(seqs[s : s + batch_size])
        enc = tok(batch, padding="max_length", max_length=n_tokens, truncation=True,
                  return_tensors="pt", return_special_tokens_mask=True)
        ids = enc["input_ids"].to(dev)
        att = enc["attention_mask"].to(dev)
        m = (att * (1 - enc["special_tokens_mask"].to(dev))).float()  # nucleotide tokens only
        cap.clear()
        encoder(input_ids=ids, attention_mask=att)
        w = m / m.sum(1, keepdim=True).clamp_min(1.0)

        def rho(z):
            return torch.einsum("bn,bnd->bd", w, z.float()).cpu().numpy()

        writer.write(
            s,
            mask=m.cpu().numpy().astype(np.uint8),
            tokens={l: cap[("x", l)].float().cpu().numpy() for l in blocks},
            pooled={l: rho(cap[("z", l)]) for l in blocks},
            final=rho(cap[("z", K_b)]),
        )
        if logger and (s // batch_size) % 50 == 0:
            logger.info(f"  cached {min(s + batch_size, len(seqs))}/{len(seqs)} genes")
    for h in hooks:
        h.remove()
    writer.close()
