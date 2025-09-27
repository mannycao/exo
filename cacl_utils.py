import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)

# Helper for TransformerEncoderLayer to match d_model with input_dim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import random

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_feature_partitions(X, K):
    """
    Create K semantic feature partitions {P₁, ..., P_K} using correlation-based clustering.
    """
    corr = np.corrcoef(X.T)
    reduced = PCA(n_components=K).fit_transform(corr)
    clusters = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(reduced)
    partitions = [[] for _ in range(K)]
    for i, label in enumerate(clusters.labels_):
        partitions[label].append(i)
    return partitions

def _feature_to_partition_map(partitions: List[List[int]]) -> Dict[int, int]:
    """
    Construct a reverse mapping from feature index j to its assigned partition P_k.
    """
    fmap = {}
    for k, feats in enumerate(partitions):
        for f in feats:
            fmap[int(f)] = int(k)
    return fmap

def build_context_groups(X, k: int = 5, mode: str = "static"):
    """
    Construct context groups G(i) for each record xᵢ ∈ ℝ^D, based on the modality.
    """
    N = len(X)
    if N == 0:
        return []

    mode_l = (mode or "tabular").lower()
    if mode_l == "static":
        mode_l = "tabular"

    if mode_l == "temporal":
        k_win = int(max(1, min(k, max(1, N - 1))))
        ctx = []
        for i in range(N):
            left = list(range(max(0, i - k_win), i))
            right = list(range(i + 1, min(N, i + k_win + 1)))
            ctx.append(left + right)
        return ctx

    def _knn_neighbors(X_arr, k_neighbors):
        k_eff = int(min(max(1, k_neighbors) + 1, len(X_arr)))
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X_arr)
        _, inds = nbrs.kneighbors(X_arr)
        out = []
        for row in inds:
            out.append([int(j) for j in row if j != row[0]][:k_neighbors])
        return out

    X_arr = np.asarray(X)

    if mode_l == "spatial":
        return _knn_neighbors(X_arr, k_neighbors=int(max(1, min(k, N - 1))))

    if mode_l == "tabular":
        try:
            n_clusters = int(max(2, min(N, round(np.sqrt(N)))))
            km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            labels = km.fit_predict(X_arr)
        except Exception:
            return _knn_neighbors(X_arr, k_neighbors=int(max(1, min(k, N - 1))))

        from collections import defaultdict
        cluster_members = defaultdict(list)
        for idx, lab in enumerate(labels):
            cluster_members[int(lab)].append(idx)

        ctx = [[] for _ in range(N)]
        global_knn = _knn_neighbors(X_arr, k_neighbors=int(max(1, min(k, N - 1))))

        for i in range(N):
            lab = int(labels[i])
            members = cluster_members[lab]
            cand = [j for j in members if j != i]
            if len(cand) == 0:
                ctx[i] = global_knn[i][:k]
                continue

            diffs = X_arr[cand] - X_arr[i]
            dists = np.einsum("nd,nd->n", diffs, diffs)
            order = np.argsort(dists)
            local_neighbors = [int(cand[o]) for o in order[:k]]

            if len(local_neighbors) < k:
                extras = [j for j in global_knn[i] if j not in local_neighbors and j != i]
                need = k - len(local_neighbors)
                local_neighbors += extras[:need]

            ctx[i] = local_neighbors[:k]
        return ctx

    return _knn_neighbors(X_arr, k_neighbors=int(max(1, min(k, N - 1))))


class PartitionedDataset(Dataset):
    """
    Torch dataset for multi-view contrastive training under CACL.
    """
    def __init__(self, X, partitions):
        self.X = X
        self.partitions = partitions

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        views = tuple(torch.tensor(x[part], dtype=torch.float32) for part in self.partitions)
        return views, idx

def collate_fn(batch):
    """
    Collate function for multi-view batches.
    """
    views_batch, indices = zip(*batch)
    num_parts = len(views_batch[0])
    batch_views = [torch.stack([v[i] for v in views_batch]) for i in range(num_parts)]
    return batch_views, list(indices)

class MLPEncoder(nn.Module):
    """
    Partition encoder f_θ for tabular views.
    """
    def __init__(self, input_dim, hidden=64, output=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output),
        )

    def forward(self, x):
        return self.net(x)





class Projector(nn.Module):
    """
    Projection head g_ϕ for latent embeddings.
    """
    def __init__(self, input_dim=64, proj_dim=32):
        super().__init__()
        self.fc = nn.Linear(input_dim, proj_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)



class PartitionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        projected_input = self.input_proj(x)
        encoded_output = self.transformer(projected_input.unsqueeze(1), mask=mask, src_key_padding_mask=src_key_padding_mask).squeeze(1)
        output_projected = self.output_proj(self.norm(encoded_output))
        return output_projected

class MultiViewModel(nn.Module):
    """
    Multi-partition encoder and projector for CACL.
    """
    def __init__(self, partitions, encoder_type="mlp", output_dim=64, proj_dim=32, nhead=2, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.proj_dim = proj_dim
        self.partitions = partitions
        
        self.encoders = nn.ModuleList()
        self.projectors = nn.ModuleList()

        for i, partition in enumerate(partitions):
            input_dim = len(partition)
            if encoder_type == "transformer":
                encoder = PartitionEncoder(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
            self.encoders.append(encoder)
            self.projectors.append(Projector(output_dim, proj_dim))

    def forward(self, views):
        projected_embeddings = []
        for i, view in enumerate(views):
            encoded_view = self.encoders[i](view)
            projected_embedding = self.projectors[i](encoded_view)
            projected_embeddings.append(projected_embedding)
        return projected_embeddings

    def get_feature_extractor(self):
        # This method returns the encoders without the projection heads
        # for fine-tuning.
        return self.encoders


def contrastive_loss(embeddings, tau=0.1):
    """
    Multi-view InfoNCE loss (L_mv) for intra-record contrastive learning.
    """
    B, K = embeddings[0].shape[0], len(embeddings)
    all_views = torch.stack(embeddings, dim=1)
    loss = 0.0
    for i in range(B):
        for k in range(K):
            anchor = all_views[i, k]
            pos = [all_views[i, l] for l in range(K) if l != k]
            sim_pos = torch.stack([
                torch.exp(F.cosine_similarity(anchor, p, dim=0) / tau)
                for p in pos]).sum()
            neg = [all_views[j, l] for j in range(B) if j != i for l in range(K)]
            sim_neg = torch.stack([
                torch.exp(F.cosine_similarity(anchor, n, dim=0) / tau)
                for n in neg]).sum()
            loss -= torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))
    return loss / (B * K)

def contrastive_loss_ctx(embeddings, ctx_idx, batch_idx,
                         alpha_mv=1.0, alpha_ctx=0.5, tau=0.1, max_ctx_samples=3):
    """
    Context-aware contrastive loss with safe fallbacks.
    """
    B, K = embeddings[0].shape[0], len(embeddings)
    all_views = torch.stack(embeddings, dim=1)
    loss = 0.0

    has_ctx = ctx_idx is not None

    for i in range(B):
        anchor_gid = int(batch_idx[i])
        neigh_set = set(ctx_idx[anchor_gid]) if has_ctx else set()
        anchor_views = all_views[i]

        for k in range(K):
            anchor = anchor_views[k]

            pos = [anchor_views[l] for l in range(K) if l != k]
            weights = [torch.tensor(alpha_mv, device=anchor.device)] * len(pos)

            if has_ctx:
                ctx_cands = [j for j, gid in enumerate(batch_idx) if gid in neigh_set and gid != anchor_gid]
                if max_ctx_samples is not None and max_ctx_samples > 0:
                    ctx_cands = random.sample(ctx_cands, min(max_ctx_samples, len(ctx_cands)))
                for j in ctx_cands:
                    pos.append(all_views[j, k])
                    weights.append(torch.tensor(alpha_ctx, device=anchor.device))

            sim_pos = torch.stack([
                w * torch.exp(F.cosine_similarity(anchor, p, dim=0) / tau)
                for p, w in zip(pos, weights)
            ]).sum()

            neg = [all_views[j, l]
                   for j, gid in enumerate(batch_idx)
                   if gid != anchor_gid and (not has_ctx or gid not in neigh_set)
                   for l in range(K)]
            if len(neg) == 0:
                neg = [all_views[j, l] for j in range(B) if j != i for l in range(K)]

            sim_neg = torch.stack([
                torch.exp(F.cosine_similarity(anchor, n, dim=0) / tau)
                for n in neg
            ]).sum()

            loss = loss - torch.log(sim_pos / (sim_pos + sim_neg + 1e-8))

    return loss / (B * K)

# Helper functions for evaluation (from run_experiment)
def _cos(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def _to1d(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).ravel()

def _pad_or_truncate(vec, d):
    v = _to1d(vec)
    if v.shape[0] == d:
        return v
    out = np.zeros(d, dtype=float)
    n = min(d, v.shape[0])
    out[:n] = v[:n]
    return out

def _avg_partition_embeddings(model, X, partitions):
    N = X.shape[0]
    Kp = len(partitions)

    if hasattr(model, "encode_partitions"):
        out = model.encode_partitions(X, partitions)
        if isinstance(out, (list, tuple)) and len(out) > 0:
            stacks = [np.asarray(o) for o in out]
            return np.mean(np.stack(stacks, axis=0), axis=0)
        if hasattr(out, "detach") and getattr(out, "ndim", 0) == 3:
            return out.detach().cpu().numpy().mean(axis=1)
        out = np.asarray(out)
        if out.ndim == 3 and out.shape[1] == Kp:
            return out.mean(axis=1)

    if hasattr(model, "encode_partition"):
        d = _to1d(model.encode_partition(_to1d(X[0, partitions[0]]), 0)).shape[0]
        E = np.zeros((N, Kp, d), dtype=float)
        for i in range(N):
            for k, part in enumerate(partitions):
                vec = model.encode_partition(_to1d(X[i, part]), k)
                E[i, k] = _pad_or_truncate(vec, d)
        return E.mean(axis=1)

    with torch.no_grad():
        E = []
        for i in range(N):
            views = [torch.tensor(X[i, p], dtype=torch.float32, device=device).unsqueeze(0)
                     for p in partitions]
            out = model(views)
            z   = torch.stack([o[0] for o in out], dim=0)
            E.append(z.mean(dim=0).cpu().numpy())
    return np.vstack(E)

def _get_partition_embeddings(model, X, partitions, i):
    with torch.no_grad():
        views = [torch.tensor(X[i, p], dtype=torch.float32, device=device).unsqueeze(0)
                 for p in partitions]
        out = model(views)
    return [out[k][0].detach().cpu().numpy() for k in range(len(partitions))]

def _pick_threshold(scores: np.ndarray, y_true: np.ndarray, normal_percentile: float = 95.0):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    neg = (y_true == 0)

    if neg.any():
        thr = float(np.nanpercentile(scores[neg], normal_percentile))
    else:
        thr = float(np.nanpercentile(scores, normal_percentile))
    preds = (scores >= thr).astype(int)

    if preds.sum() == 0 or preds.sum() == len(preds):
        qs = np.linspace(50, 99.5, 100)
        best_f1, best_thr = -1.0, thr
        for q in qs:
            t = float(np.nanpercentile(scores, q))
            p = (scores >= t).astype(int)
            tp = int(((p == 1) & (y_true == 1)).sum())
            fp = int(((p == 1) & (y_true == 0)).sum())
            fn = int(((p == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        thr = best_thr
        preds = (scores >= thr).astype(int)
    return thr, preds

def _upper_triangle_argmax(mat: np.ndarray) -> Tuple[int, int]:
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], "Expected square matrix"
    tri = np.triu(mat, k=1)
    if not np.isfinite(tri).any():
        return (0, 1) if mat.shape[0] > 1 else (0, 0)
    idx = np.nanargmax(tri)
    k_star, l_star = np.unravel_index(idx, tri.shape)
    return int(k_star), int(l_star)

def compute_dependency_matrix(X_inliers, model, partitions, eta=0.2):
    N = len(X_inliers)
    K = len(partitions)
    Lambda_sum = np.zeros((K, K))

    model.eval()
    with torch.no_grad():
        for i in range(N):
            views = [torch.tensor(X_inliers[i][p], dtype=torch.float32).unsqueeze(0).to(device)
                    for p in partitions]
            embs = model(views)
            embeddings = [e.squeeze(0).cpu().numpy() for e in embs]
            D = np.zeros((K, K))
            for a in range(K):
                for b in range(K):
                    if a != b:
                        D[a, b] = 1 - F.cosine_similarity(
                            torch.tensor(embeddings[a]),
                            torch.tensor(embeddings[b]),
                            dim=0
                        ).item()
            Lambda = (D < eta).astype(int)
            Lambda_sum += Lambda

    return Lambda_sum / N

def compute_context_embeddings(model, X, partitions):
    model.eval()
    N = len(X)
    K = len(partitions)

    embeddings = []

    with torch.no_grad():
        for i in range(N):
            views = [
                torch.tensor(X[i, part], dtype=torch.float32).unsqueeze(0).to(device)
                for part in partitions
            ]
            out = model(views)
            vecs = [e.squeeze(0).cpu().numpy() for e in out]
            avg = np.mean(np.stack(vecs, axis=0), axis=0)
            embeddings.append(avg)

    return np.vstack(embeddings)

def explain_record(
    record_embeddings: List[np.ndarray],
    context_avg_embeddings: Optional[np.ndarray] = None,
    context_indices: Optional[List[int]] = None,
    eta: float = 0.2,
    dependency_matrix: Optional[np.ndarray] = None,
    context_threshold_quantile: Optional[float] = None,
    ref_ctx_sim_distribution: Optional[np.ndarray] = None,
) -> Dict:
    def _cos_sim(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    K = len(record_embeddings)
    H = np.vstack(record_embeddings)
    D = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            D[i, j] = 1.0 - _cos_sim(H[i], H[j])
    k_star, l_star = _upper_triangle_argmax(D)
    max_disagreement = float(D[k_star, l_star])

    violated_deps = []
    violated_deps_features = []
    if dependency_matrix is not None:
        Kdm = dependency_matrix.shape[0]
        if Kdm != K:
            pass
        else:
            rho = 0.9
            for i in range(K):
                for j in range(i+1, K):
                    if dependency_matrix[i, j] > rho and D[i, j] > eta:
                        violated_deps.append((i, j))

    context_similarity = None
    ctx_flag = False
    if context_avg_embeddings is not None and context_indices:
        avg = H.mean(axis=0)
        sims = []
        for j in context_indices:
            sims.append(_cos_sim(avg, context_avg_embeddings[j]))
        if sims:
            context_similarity = float(np.mean(sims))
            if context_threshold_quantile is not None and ref_ctx_sim_distribution is not None:
                thr = float(np.quantile(ref_ctx_sim_distribution, context_threshold_quantile))
                ctx_flag = context_similarity < thr

    return {
        "disagreement_matrix": D,
        "most_conflicting_partitions": (k_star, l_star),
        "max_disagreement": max_disagreement,
        "violated_dependencies": violated_deps,
        "violated_dependencies_features": violated_deps_features,
        "context_similarity": context_similarity,
        "context_flag": ctx_flag,
    }

def mutate_test_set(
    X_test: np.ndarray,
    partitions=None,
    mutation_rate: float = 0.10,
    seed: int = 0,
    a4_noise_std: float = 0.02,
    ensure_all_types: bool = True,
    *,
    data_type: str = "tabular",
    k_ctx: int = 5,
    n_clusters: int | None = None,
    window: int | None = None,
    extreme_mult: float = 0.75
) -> tuple[np.ndarray, dict[int, object]]:
    rng = np.random.default_rng(seed)
    X_mut = np.array(X_test, copy=True)
    N, D  = X_mut.shape

    if N == 0:
        return X_mut, {}

    requested = int(round(mutation_rate * N))
    if ensure_all_types and N >= 4:
        requested = max(requested, 4)
    requested = max(1, min(requested, N))

    sel = rng.choice(N, size=requested, replace=False)

    if ensure_all_types and N >= 4:
        base, rem = divmod(requested, 4)
        counts = [base + (i < rem) for i in range(4)]
        type_order = (["A1"] * counts[0] +
                      ["A2"] * counts[1] +
                      ["A3"] * counts[2] +
                      ["A4"] * counts[3])
    else:
        all_types = ["A1", "A2", "A3", "A4"]
        type_order = [all_types[i % 4] for i in range(requested)]

    def _lo_hi_span(col: np.ndarray) -> tuple[float, float, float]:
        lo, hi = np.nanpercentile(col, [0.1, 99.9])
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
        if lo == hi:
            hi = lo + 1e-6
        return lo, hi, (hi - lo)

    metas: dict[int, object] = {}

    mode = (data_type or "tabular").lower()
    if mode == "static":
        mode = "tabular"

    if mode == "tabular" and N >= 2:
        try:
            C = n_clusters if n_clusters is not None else int(max(2, min(N, round(np.sqrt(N)))))
            km = KMeans(n_clusters=C, random_state=seed, n_init="auto")
            labels = km.fit_predict(X_test)
            centroids = km.cluster_centers_
            from collections import defaultdict
            cluster_members = defaultdict(list)
            for i, lab in enumerate(labels):
                cluster_members[int(lab)].append(i)
        except Exception:
            labels, centroids, cluster_members = None, None, None

    if mode == "spatial" and N >= 2:
        diffs = X_test[:, None, :] - X_test[None, :, :]
        distM = np.sqrt(np.sum(diffs * diffs, axis=2))

    if mode == "temporal":
        if window is None:
            window = max(5, int(round(0.10 * N)))
        window = int(max(1, min(window, max(1, N-1))))

    for idx, mtype in zip(sel, type_order):
        idx = int(idx)

        if mtype == "A1":
            f = int(rng.integers(0, max(1, D)))
            lo, hi, span = _lo_hi_span(X_test[:, f])
            med = np.nanmedian(X_test[:, f])
            if X_test[idx, f] < med:
                new_val = hi + extreme_mult * span
            else:
                new_val = lo - extreme_mult * span
            jitter = float(rng.normal(0.0, 0.02 * span))
            X_mut[idx, f] = float(new_val + jitter)
            metas[idx] = (f, f)

        elif mtype == "A2":
            if D >= 2:
                if partitions and len(partitions) >= 2 and sum(len(p) for p in partitions) == D:
                    p1, p2 = rng.choice(len(partitions), size=2, replace=False)
                    f1 = int(rng.choice(partitions[p1]))
                    f2 = int(rng.choice(partitions[p2]))
                else:
                    f1, f2 = rng.choice(D, size=2, replace=False)
                lo1, hi1, s1 = _lo_hi_span(X_test[:, f1])
                lo2, hi2, s2 = _lo_hi_span(X_test[:, f2])
                X_mut[idx, f1] = hi1 + extreme_mult * s1
                X_mut[idx, f2] = lo2 - extreme_mult * s2
                metas[idx] = (int(f1), int(f2))
            else:
                f = 0
                lo, hi, span = _lo_hi_span(X_test[:, f])
                X_mut[idx, f] = hi + extreme_mult * span
                metas[idx] = (f, f)

        elif mtype == "A3":
            if N < 2:
                noise = rng.normal(0.0, a4_noise_std, size=D).astype(float)
                X_mut[idx] = X_mut[idx] + noise
                metas[idx] = None
                continue

            if mode == "tabular" and labels is not None and centroids is not None:
                lab_i = int(labels[idx])
                d2c = np.linalg.norm(centroids - X_test[idx], axis=1)
                far_lab = int(np.argmax(d2c))
                cand = cluster_members[far_lab]
                if len(cand) == 0:
                    cand = [j for j in range(N) if j != idx]
                dists = np.linalg.norm(X_test[cand] - X_test[idx], axis=1)
                donor = int(cand[int(np.argmax(dists))])

            elif mode == "spatial":
                drow = distM[idx].copy()
                drow[idx] = -np.inf
                k = max(1, int(round(0.05 * N)))
                farset = np.argsort(drow)[-k:]
                donor = int(rng.choice(farset))

            elif mode == "temporal":
                donor = int((idx + window) % N)

            else:
                drow = np.linalg.norm(X_test - X_test[idx], axis=1)
                drow[idx] = -np.inf
                donor = int(np.argmax(drow))

            X_mut[idx] = X_test[donor]
            metas[idx] = "context"

        elif mtype == "A4":
            noise = rng.normal(0.0, a4_noise_std, size=D).astype(float)
            X_mut[idx] = X_mut[idx] + noise
            metas[idx] = None

        else:
            noise = rng.normal(0.0, a4_noise_std, size=D).astype(float)
            X_mut[idx] = X_mut[idx] + noise
            metas[idx] = None

    return X_mut, metas

def evaluate_explanations_on_mutations(
    explanations: Dict[int, Dict],
    metas: Dict[int, Optional[Tuple[int,int]]],
    partitions: List[List[int]],
    topM: int = 3,
    use_violated_deps: bool = True,
    y_true: Optional[np.ndarray] = None,
    ref_max_dis_clean: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    def _top_pairs_from_D(D: np.ndarray, M: int = 3) -> set:
        if D is None:
            return set()
        K = D.shape[0]
        tri = np.triu(D, 1)
        if not np.isfinite(tri).any():
            return set()
        order = np.argsort(tri, axis=None)[::-1]
        pairs = set()
        for idx in order:
            k, l = np.unravel_index(idx, (K, K))
            if k < l:
                pairs.add((int(k), int(l)))
                if len(pairs) >= M:
                    break
        return pairs

    f2p = _feature_to_partition_map(partitions)
    total = {"A1":0, "A2":0, "A3":0, "A4":0}
    correct = {"A1":0, "A2":0, "A3":0, "A4":0}

    if y_true is not None:
        clean_idxs = [i for i in explanations.keys() if y_true[i] == 0]
        ref_max_dis = [explanations[i]["max_disagreement"] for i in clean_idxs]
    else:
        ref_max_dis = [
            ex["max_disagreement"]
            for ex in explanations.values()
            if not ex.get("violated_dependencies", []) and not ex.get("context_flag", False)
        ]

    if ref_max_dis_clean is not None and len(ref_max_dis_clean) >= 20:
        p90 = float(np.quantile(ref_max_dis_clean, 0.90))
    elif y_true is not None:
        clean_idxs = [i for i in explanations.keys() if y_true[i] == 0]
        ref_max_dis = [explanations[i]["max_disagreement"] for i in clean_idxs]
        p90 = float(np.quantile(ref_max_dis, 0.90)) if len(ref_max_dis) >= 20 else 0.2
    else:
        ref_max_dis = [
            ex["max_disagreement"]
            for ex in explanations.values()
            if not ex.get("violated_dependencies", []) and not ex.get("context_flag", False)
        ]
        p90 = float(np.quantile(ref_max_dis, 0.90)) if len(ref_max_dis) >= 20 else 0.2

    for idx, expected in metas.items():
        ex = explanations.get(idx)
        if ex is None:
            continue

        D = ex.get("disagreement_matrix", None)
        cand_pairs = _top_pairs_from_D(D, M=topM)
        if use_violated_deps:
            viol = [tuple(sorted(p)) for p in ex.get("violated_dependencies", [])]
            cand_pairs |= set(viol)

        if expected == "context":
            total["A3"] += 1
            ok = bool(ex.get("context_flag", False))
            if not ok and ex.get("context_similarity") is not None:
                ok = float(ex["context_similarity"]) < 0.9
            correct["A3"] += int(ok)

        elif expected is None:
            total["A4"] += 1
            ok = (len(ex.get("violated_dependencies", [])) == 0) and (ex["max_disagreement"] <= p90)
            correct["A4"] += int(ok)

        else:
            f1, f2 = expected
            k1, k2 = f2p[int(f1)], f2p[int(f2)]
            exp_pair = tuple(sorted((k1, k2)))

            if k1 == k2:
                total["A1"] += 1
                ok = any((k1 == a or k1 == b) for (a, b) in cand_pairs)
                correct["A1"] += int(ok)
            else:
                total["A2"] += 1
                ok = (exp_pair in cand_pairs)
                correct["A2"] += int(ok)

    accs = {}
    for t in ["A1","A2","A3","A4"]:
        accs[t] = (correct[t] / total[t]) if total[t] > 0 else np.nan
    tot = sum(total.values())
    cor = sum(correct.values())
    accs["overall"] = (cor / tot) if tot > 0 else np.nan
    return accs
