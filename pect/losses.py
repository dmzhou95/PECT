
def zinb_nll(x, mu, theta, pi_logits, eps: float = 1e-8):
    """Element-wise negative log-likelihood of ZINB(x | mu, theta, pi), pi = sigmoid(pi_logits).

    Numerically stable form (as in scVI) working with the logits of pi.
    """
    softplus_pi = F.softplus(-pi_logits)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi_logits + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
    )
    ll = torch.where(x < eps, case_zero, case_non_zero)
    return -ll


def zinb_loss(x, mu, theta, pi_logits):
    """L_ZINB = -1/(|B| G) sum_i sum_g log ZINB(x_ig | mu_ig, theta_ig, pi_ig)."""
    return zinb_nll(x, mu, theta, pi_logits).mean()


# ---------------------------------------------------------------- Eq. (31)
def graph_infomax_loss(H: torch.Tensor, H_shuf: torch.Tensor, W_D: torch.Tensor):
    """Deep graph infomax: s = sigmoid(mean_i h_i), D(h, s) = sigmoid(h W_D s^T)."""
    s = torch.sigmoid(H.mean(0))
    pos = H @ (W_D @ s)
    neg = H_shuf @ (W_D @ s)
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))


# ---------------------------------------------------------------- Eq. (32)
def _std_cols(A: torch.Tensor, eps: float = 1e-8):
    """Centre each column and scale to unit RMS; constant columns map to zero."""
    A = A - A.mean(0, keepdim=True)
    rms = A.pow(2).mean(0, keepdim=True).sqrt()
    return torch.where(rms > eps, A / rms.clamp_min(eps), torch.zeros_like(A))


def decorrelation_loss(Hg: torch.Tensor, Zs: torch.Tensor):
    """C = |B|^-1 std(H_bar W_g)^T std(Z_bar W_s); sum_{p!=q} C_pq^2 + sum_p (C_pp - 1)^2."""
    C = _std_cols(Hg).t() @ _std_cols(Zs) / Hg.shape[0]
    diag = torch.diagonal(C)
    off = C.pow(2).sum() - diag.pow(2).sum()
    return off + (diag - 1.0).pow(2).sum()


# ---------------------------------------------------------------- Eq. (33)
def infonce_loss(Hg: torch.Tensor, Zs: torch.Tensor, tau: float):
    """sim(h, z) = cos(h W_g, z W_s); the two modalities of one spot are a positive pair."""
    logits = F.normalize(Hg, dim=1) @ F.normalize(Zs, dim=1).t() / tau
    target = torch.arange(Hg.shape[0], device=Hg.device)
    return F.cross_entropy(logits, target)


# ---------------------------------------------------------------- Eq. (34)
def adapter_reg_loss(delta_sqnorms: List[torch.Tensor], Hg: torch.Tensor, Zs: torch.Tensor,
                     lam: float, gamma: float):
    """lambda sum_l ||Delta W^(l)||_F^2 + gamma/|B| sum_i KL(p_i || q_i),
    p_i = softmax(z_bar_i W_s), q_i = softmax(h_bar_i W_g)."""
    mag = torch.stack(delta_sqnorms).sum()
    logp = F.log_softmax(Zs, dim=1)
    logq = F.log_softmax(Hg, dim=1)
    kl = (logp.exp() * (logp - logq)).sum(1).mean()
    return lam * mag + gamma * kl, mag, kl


# ---------------------------------------------------------------- Eq. (35)
def smoothness_loss(M: torch.Tensor, edges: torch.Tensor):
    """1/|E| sum_{(i,j) in E} ||M_i - M_j||^2 over ordered spatial edges inside B+."""
    if edges.numel() == 0:
        return M.new_zeros(())
    d = M.index_select(0, edges[:, 0]) - M.index_select(0, edges[:, 1])
    return d.pow(2).sum(1).mean()
