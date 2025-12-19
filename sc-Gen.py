#!/usr/bin/env python3
"""
scgen_overfitting_fixed.py

Modified scGen-style pipeline with:
 - EarlyStopping
 - ReduceLROnPlateau scheduler (no verbose argument)
 - Optional KL-annealing (linear warmup)
 - Model regularization (dropout, LayerNorm, GELU)
 - Gradient clipping
 - Export training curve + artifacts for R visualization
 - No blocking matplotlib calls (figures are saved)

Inputs: same as original sc-Gen.py exports (mtx/genes/meta or lognorm CSV)
Outputs (in out_dir):
 - training_curve.csv
 - train_history.json
 - latent_mu.npy, latents_umap.csv (with stim labels)
 - synthetic_ctrl_to_stim_lognorm.csv, synthetic_ctrl_to_stim_meta.csv
 - marker_expression_real.csv, marker_expression_pred.csv
 - heatmap_input.csv (top variable genes expression matrix)
 - umap_real_generated.png, gene_pred_vs_orig_<gene>.png

Usage example:
 python scgen_overfitting_fixed.py --data_dir output --use_pca --epochs 200

"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import io, sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import umap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import r2_score

# -------------------------
# Utilities & Dataset
# -------------------------
class ExprDataset(Dataset):
    def __init__(self, X, meta, cond_idx):
        # X: numpy array (cells x features)
        self.X = X.astype(np.float32)
        self.meta = meta.reset_index(drop=True)
        self.cond_idx = np.array(cond_idx, dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.cond_idx[idx], idx


def load_from_mtx(data_dir, mtx_fname, genes_fname, meta_fname, use_lognorm_csv=None):
    # returns X (cells x genes), genes list, meta dataframe
    if use_lognorm_csv:
        path = os.path.join(data_dir, use_lognorm_csv)
        df = pd.read_csv(path, index_col=0)
        genes = pd.read_csv(os.path.join(data_dir, genes_fname), header=None).iloc[:,0].astype(str).tolist()
        if df.shape[0] == len(genes):
            # genes x cells -> transpose
            X = df.values.T
            barcodes = df.columns.tolist()
        elif df.shape[1] == len(genes):
            X = df.values
            barcodes = df.index.tolist()
        else:
            raise ValueError("lognorm CSV shape inconsistent with genes file")
    else:
        mtx_path = os.path.join(data_dir, mtx_fname)
        mat = io.mmread(mtx_path)
        if sparse.issparse(mat):
            mat = mat.tocsc()
        # mtX is often genes x cells -> transpose
        X = np.array(mat.T.todense())
        genes = pd.read_csv(os.path.join(data_dir, genes_fname), header=None).iloc[:,0].astype(str).tolist()
        barcodes = None

    meta = pd.read_csv(os.path.join(data_dir, meta_fname), index_col=None)
    # Align barcodes if available in meta
    if 'barcode' in meta.columns and barcodes is not None and len(barcodes) == X.shape[0]:
        # if ordering mismatches, try to reorder
        if not all(meta['barcode'].astype(str).values == np.array(barcodes)):
            # attempt to reorder X to match meta
            order = [barcodes.index(b) for b in meta['barcode'].astype(str).tolist()]
            X = X[order, :]
    return X, genes, meta

# -------------------------
# Model: Regularized VAE (encoder/decoder with dropout/LayerNorm)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim//2, 16)),
            nn.LayerNorm(max(hidden_dim//2, 16)),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(max(hidden_dim//2, 16), latent_dim)
        self.fc_logvar = nn.Linear(max(hidden_dim//2, 16), latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, max(hidden_dim//2, 16)),
            nn.LayerNorm(max(hidden_dim//2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim//2, 16), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, use_vae=True, dropout=0.1):
        super().__init__()
        self.use_vae = use_vae
        self.enc = Encoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.dec = Decoder(latent_dim, hidden_dim, input_dim, dropout=dropout)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        if self.use_vae:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            logvar = torch.zeros_like(mu)
        recon = self.dec(z)
        return recon, mu, logvar, z

# -------------------------
# Loss
# -------------------------

def vae_loss(recon, x, mu, logvar, recon_loss='mse', kl_weight=1.0):
    if recon_loss == 'mse':
        rec = nn.functional.mse_loss(recon, x, reduction='mean')
    elif recon_loss == 'mae':
        rec = nn.functional.l1_loss(recon, x, reduction='mean')
    else:
        raise ValueError
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl_weight * kld, rec.item(), kld.item()

# -------------------------
# EarlyStopping
# -------------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.wait = 0
        self.best_state = None

    def step(self, val_loss, model=None):
        stop = False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if model is not None:
                self.best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop = True
                if self.restore_best and model is not None and self.best_state is not None:
                    model.load_state_dict(self.best_state)
        return stop

# -------------------------
# Train loop with KL-annealing and scheduler
# -------------------------

def train(model, loader, val_loader, optimizer, device, epochs=100, clip_norm=1.0, recon_loss='mse', kl_weight=1.0,
          kl_warmup_epochs=0, scheduler=None, early_stopper=None, out_dir='out'):
    history = {'train_loss': [], 'val_loss': [], 'kl_weight': []}
    best_val = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, conds, idx in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(xb)
            # compute annealed kl weight
            if kl_warmup_epochs > 0:
                anneal = min(1.0, epoch / float(kl_warmup_epochs))
            else:
                anneal = 1.0
            loss, rec_val, kld_val = vae_loss(recon, xb, mu, logvar, recon_loss=recon_loss, kl_weight=kl_weight*anneal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if len(train_losses)>0 else 0.0

        # validation
        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, conds, idx in val_loader:
                xb = xb.to(device)
                recon, mu, logvar, z = model(xb)
                val_loss, rec_val, kld_val = vae_loss(recon, xb, mu, logvar, recon_loss=recon_loss, kl_weight=kl_weight*anneal)
                vlosses.append(val_loss.item())
        val_loss = float(np.mean(vlosses)) if len(vlosses)>0 else 0.0

        # scheduler step (ReduceLROnPlateau expects metric)
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except TypeError:
                # some PyTorch versions expect only metric
                scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['kl_weight'].append(kl_weight*anneal)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} kl_w={kl_weight*anneal:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))

        # early stopping
        if early_stopper is not None:
            stop = early_stopper.step(val_loss, model=model)
            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

    return history

# -------------------------
# Perturbation & generation helpers (same as before)
# -------------------------

def compute_latents(model, X, device, batch_size=1024):
    model.eval()
    mus = []
    zs = []
    logvars = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            recon, mu, logvar, z = model(xb)
            mus.append(mu.cpu().numpy())
            zs.append(z.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    mus = np.vstack(mus)
    zs = np.vstack(zs)
    logvars = np.vstack(logvars)
    return zs, mus, logvars


def decode_from_z(model, Z, device, batch_size=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], batch_size):
            zb = torch.tensor(Z[i:i+batch_size], dtype=torch.float32).to(device)
            out = model.dec(zb)
            outs.append(out.cpu().numpy())
    return np.vstack(outs)

# -------------------------
# Main pipeline
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='output', help='where Seurat exports are')
    parser.add_argument('--mtx', default='raw_counts_train.mtx', help='matrix market raw counts (will be transposed)')
    parser.add_argument('--genes', default='genes_raw_counts.csv')
    parser.add_argument('--meta', default='metadata_train.csv')
    parser.add_argument('--use_lognorm_csv', default=None, help='optional lognorm CSV to load directly (overrides mtx)')
    parser.add_argument('--top_n_genes', type=int, default=2000)
    parser.add_argument('--use_pca', action='store_true')
    parser.add_argument('--pca_n', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_vae', action='store_true', help='use VAE (else deterministic AE)')
    parser.add_argument('--recon_loss', default='mse', choices=['mse','mae'])
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--kl_warmup', type=int, default=10, help='epochs for linear KL warmup')
    parser.add_argument('--out_dir', default='scgen_out')
    parser.add_argument('--celltype_col', default='seurat_clusters', help='metadata column for cell type (optional)')
    parser.add_argument('--stim_col', default='stim', help='metadata column with STIM/CTRL labels')
    parser.add_argument('--per_celltype', action='store_true', help='compute perturbation per celltype (requires celltype_col)')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout in encoder/decoder')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data...")
    X, genes, meta = load_from_mtx(args.data_dir, args.mtx, args.genes, args.meta, use_lognorm_csv=args.use_lognorm_csv)

    # ensure stim column
    if args.stim_col not in meta.columns:
        raise ValueError(f"{args.stim_col} not in metadata columns: {meta.columns.tolist()}")

    # gene selection
    print("Gene selection (variance)...")
    gene_var = np.var(X, axis=0)
    top_idx = np.argsort(-gene_var)[:args.top_n_genes]
    X_sel = X[:, top_idx]
    genes_sel = [genes[i] for i in top_idx]
    print("Selected genes:", len(genes_sel))

    # scaling
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_sel)

    # optional PCA
    if args.use_pca:
        print("Running PCA...")
        pca = PCA(n_components=args.pca_n, random_state=args.seed)
        X_feat = pca.fit_transform(X_scaled)
        print("PCA shape:", X_feat.shape)
    else:
        X_feat = X_scaled

    # prepare conditions
    stim_vals = meta[args.stim_col].astype(str).values
    unique_stim = sorted(pd.unique(stim_vals))
    stim2idx = {s:i for i,s in enumerate(unique_stim)}
    cond_idx = [stim2idx[s] for s in stim_vals]

    # train/val split
    train_idx, val_idx = train_test_split(np.arange(X_feat.shape[0]), test_size=0.1, random_state=args.seed, shuffle=True)
    X_train = X_feat[train_idx]
    X_val = X_feat[val_idx]
    cond_train = [cond_idx[i] for i in train_idx]
    cond_val = [cond_idx[i] for i in val_idx]

    train_ds = ExprDataset(X_train, meta.iloc[train_idx].reset_index(drop=True), cond_train)
    val_ds = ExprDataset(X_val, meta.iloc[val_idx].reset_index(drop=True), cond_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(input_dim=X_feat.shape[1], hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, use_vae=args.use_vae, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler without verbose kwarg (older PyTorch compatibility)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-5, restore_best=True)

    print("Start training AE/VAE...")
    history = train(model, train_loader, val_loader, optimizer, device,
                    epochs=args.epochs, clip_norm=1.0, recon_loss=args.recon_loss, kl_weight=args.kl_weight,
                    kl_warmup_epochs=args.kl_warmup, scheduler=scheduler, early_stopper=early_stopper, out_dir=args.out_dir)

    # save history
    hist_df = pd.DataFrame({'epoch': np.arange(1, len(history['train_loss'])+1),
                             'train_loss': history['train_loss'],
                             'val_loss': history['val_loss'],
                             'kl_weight': history['kl_weight']})
    hist_df.to_csv(os.path.join(args.out_dir, 'training_curve.csv'), index=False)
    with open(os.path.join(args.out_dir, 'train_history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    # compute latents for all cells
    print("Computing latents for all cells...")
    zs, mus, logvars = compute_latents(model, X_feat, device)
    latent = mus
    np.save(os.path.join(args.out_dir, 'latent_mu.npy'), latent)

    # save latent + stim labels for R UMAP
    latent_df = pd.DataFrame(latent)
    latent_df['stim'] = stim_vals
    latent_df.to_csv(os.path.join(args.out_dir, 'latents_umap_input.csv'), index=False)

    # compute perturbation vectors
    print("Computing perturbation vectors...")
    stim_labels = stim_vals
    if args.per_celltype:
        if args.celltype_col not in meta.columns:
            raise ValueError("per_celltype requested but celltype_col not in metadata")
        celltypes = meta[args.celltype_col].astype(str).values
        unique_ct = sorted(pd.unique(celltypes))
        deltas = {}
        for ct in unique_ct:
            mask = (celltypes == ct)
            mask_stim = mask & (stim_labels == unique_stim[1]) if len(unique_stim)>1 else mask
            mask_ctrl = mask & (stim_labels == unique_stim[0])
            if mask_stim.sum() < 1 or mask_ctrl.sum() < 1:
                continue
            mean_stim = latent[mask_stim].mean(axis=0)
            mean_ctrl = latent[mask_ctrl].mean(axis=0)
            deltas[ct] = mean_stim - mean_ctrl
        pd.DataFrame({k: v for k,v in deltas.items()}).to_csv(os.path.join(args.out_dir, 'deltas_per_celltype.csv'))
    else:
        if len(unique_stim) < 2:
            raise ValueError("Need at least 2 conditions (CTRL and STIM) to compute global delta")
        stim_idx = stim2idx[unique_stim[1]]
        ctrl_idx = stim2idx[unique_stim[0]]
        mask_stim = np.array([1 if s==stim_idx else 0 for s in cond_idx], dtype=bool)
        mask_ctrl = np.array([1 if s==ctrl_idx else 0 for s in cond_idx], dtype=bool)
        mean_stim = latent[mask_stim].mean(axis=0)
        mean_ctrl = latent[mask_ctrl].mean(axis=0)
        delta = mean_stim - mean_ctrl
        np.save(os.path.join(args.out_dir, 'delta_global.npy'), delta)
        deltas = {'global': delta}

    # Generate predicted stimulated cells from control cells
    print("Generating predicted STIM cells from CTRL cells...")
    ctrl_mask = np.array([s==unique_stim[0] for s in stim_vals])
    ctrl_idx_list = np.where(ctrl_mask)[0]
    if args.per_celltype:
        generated = []
        gen_meta = []
        for i in ctrl_idx_list:
            ct = meta.iloc[i][args.celltype_col]
            if ct in deltas:
                z = latent[i] + deltas[ct]
            else:
                if 'global' in deltas:
                    z = latent[i] + deltas['global']
                else:
                    continue
            generated.append(z)
            gen_meta.append({'orig_idx': i, 'celltype': ct})
        generated = np.vstack(generated) if len(generated)>0 else np.zeros((0, latent.shape[1]))
    else:
        z_ctrl = latent[ctrl_idx_list]
        delta = deltas['global']
        generated = z_ctrl + delta
        gen_meta = [{'orig_idx': int(i)} for i in ctrl_idx_list]

    # decode generated latents to feature space
    decoded = decode_from_z(model, generated, device)
    # inverse transforms
    if args.use_pca:
        # need to reconstruct using the same pca and scaler objects: we saved them temporarily in local scope
        # Here, we assume pca and scaler exist
        decoded_orig_space = pca.inverse_transform(decoded)
        decoded_unscaled = scaler.inverse_transform(decoded_orig_space)
    else:
        decoded_unscaled = scaler.inverse_transform(decoded)

    synth_df = pd.DataFrame(decoded_unscaled, columns=genes_sel)
    synth_meta = pd.DataFrame(gen_meta)
    synth_df.to_csv(os.path.join(args.out_dir, 'synthetic_ctrl_to_stim_lognorm.csv'), index=False)
    synth_meta.to_csv(os.path.join(args.out_dir, 'synthetic_ctrl_to_stim_meta.csv'), index=False)
    print("Saved synthetic data:", synth_df.shape)

    # Save heatmap input: top variable genes (unscaled lognorm values)
    # ------------------------------------
# Heatmap input: top variable genes (within selected HVGs)
# ------------------------------------
    print("Preparing heatmap input matrix...")
    top_hvgs = np.argsort(-np.var(X_sel, axis=0))[:100]  # top 100 among selected genes
    heat_mat = X_sel[:, top_hvgs]

    heat_df = pd.DataFrame(heat_mat, columns=[genes_sel[i] for i in top_hvgs])
    heat_df['stim'] = stim_vals

    heat_df.to_csv(os.path.join(args.out_dir, 'heatmap_input.csv'), index=False)
    print("Saved heatmap_input.csv (100 HVGs within selected genes)")


    # Marker gene comparisons: save real vs predicted for a set of markers present in genes_sel
    markers = ['CD3D','GNLY','MS4A1','CD14']
    markers = [m for m in markers if m in genes_sel]
    if len(markers) > 0:
        idx_map = {g:i for i,g in enumerate(genes_sel)}
        # original ctrl expression for matching original indices
        orig_idxs = [gm['orig_idx'] for gm in gen_meta]
        real_vals = X_sel[orig_idxs][:, [idx_map[g] for g in markers]]
        pred_vals = decoded_unscaled[:, [idx_map[g] for g in markers]]
        real_df = pd.DataFrame(real_vals, columns=markers)
        pred_df = pd.DataFrame(pred_vals, columns=markers)
        real_df['orig_idx'] = orig_idxs
        pred_df['orig_idx'] = orig_idxs
        real_df.to_csv(os.path.join(args.out_dir, 'marker_expression_real.csv'), index=False)
        pred_df.to_csv(os.path.join(args.out_dir, 'marker_expression_pred.csv'), index=False)

    # -------------------------
    # Visualizations saved to files (non-blocking)
    # -------------------------
    print("UMAP visualization...")
    reducer = umap.UMAP(n_components=2, random_state=args.seed)
    combined_latent = np.vstack([latent, generated]) if generated.shape[0]>0 else latent
    umap_coords = reducer.fit_transform(combined_latent)
    real_coords = umap_coords[:latent.shape[0]]
    gen_coords = umap_coords[latent.shape[0]:] if generated.shape[0]>0 else np.zeros((0,2))

    plt.figure(figsize=(6,5))
    for s in np.unique(stim_vals):
        mask = stim_vals == s
        plt.scatter(real_coords[mask,0], real_coords[mask,1], s=4, label=s, alpha=0.6)
    if gen_coords.shape[0]>0:
        plt.scatter(gen_coords[:,0], gen_coords[:,1], s=10, c='k', marker='x', label='generated')
    plt.legend(markerscale=3)
    plt.title('UMAP of latent space (real + generated)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'umap_real_generated.png'), dpi=150)
    plt.close()

    # Gene-level scatter saves
    if len(markers) > 0:
        for g in markers:
            gi = idx_map[g]
            gen_vals = decoded_unscaled[:, gi]
            orig_vals = X_sel[orig_idxs][:, gi]
            nplot = min(len(orig_vals), len(gen_vals))
            plt.figure(figsize=(4,4))
            plt.scatter(orig_vals[:nplot], gen_vals[:nplot], s=6, alpha=0.6)
            plt.xlabel('Original CTRL (lognorm)')
            plt.ylabel('Predicted STIM (lognorm)')
            plt.title(f'Predicted vs Original for {g}')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f'gene_pred_vs_orig_{g}.png'), dpi=150)
            plt.close()
            try:
                print(f"R2 for {g}: {r2_score(orig_vals[:nplot], gen_vals[:nplot]):.4f}")
            except Exception:
                pass

    # Save run config
    with open(os.path.join(args.out_dir, 'run_config.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    print("Done. All outputs in:", args.out_dir)

if __name__ == '__main__':
    main()
