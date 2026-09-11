#!/usr/bin/env python3
"""
Minimal overlap visualiser (stratified, no-CLI)

Outputs ONLY:
  1) tsne_overlap_mapped.png
  2) knn_pseudo_confusion_heatmap.png
  3) 3d_scatter_fluorescence.html

Edit the CONFIG block and run:  python inspect_overlap_min.py
"""

# =====================
# CONFIG — EDIT THESE
# =====================
DATA_PATH   = r"C:\\Users\\JR13\\Downloads\\EXP5_merged.csv"                 # <-- set to your merged labelled CSV (~200MB)
LABEL_COL   = "Label"                          # <-- source label column in DATA_PATH
MAP_CSV     = r"C:\\Users\\JR13\\Downloads\\labelmap.csv"  # <-- label map with columns: Label, Mapto
MAPPED_COL  = "Label_mapped"                   # <-- new/used label name
UNMAPPED    = "keep"                           # options: "keep" | "not_recognized" | "drop"
OUTDIR      = "overlap_report"                 # output folder

# Sampling/compute knobs
CHUNK_SIZE  = 100_000                           # chunk size for reading
SAMPLE_MAX  = 50_000                            # max total sample for kNN/3D (before t-SNE downsampling)
TSNE_MAX    = 20_000                            # max points used for t-SNE
K_NEIGHBORS = 5                                 # k for pseudo-confusion
RANDOM_STATE= 42
# =====================

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

np.random.seed(RANDOM_STATE)

DATA_PATH = Path(DATA_PATH)
MAP_PATH  = Path(MAP_CSV)
OUTDIR    = Path(OUTDIR); OUTDIR.mkdir(exist_ok=True, parents=True)

# ---------- Load label map ----------
if not MAP_PATH.exists():
    raise SystemExit(f"Mapping file not found: {MAP_PATH}")
map_df = pd.read_csv(MAP_PATH)
map_df.columns = [c.strip() for c in map_df.columns]
if not {'Label','Mapto'}.issubset(map_df.columns):
    raise SystemExit("Map CSV must have columns: 'Label','Mapto'")
map_df['Label'] = map_df['Label'].astype(str).str.strip()
map_df['Mapto'] = map_df['Mapto'].astype(str).str.strip()
label_to_map = {}
for _, r in map_df.iterrows():
    src, tgt = r['Label'], r['Mapto']
    if tgt and tgt.lower() not in ('nan','none',''):
        label_to_map[src] = tgt

# ---------- Discover columns & numeric features ----------
first = pd.read_csv(DATA_PATH, nrows=10)
all_cols = list(first.columns)
if LABEL_COL not in all_cols:
    raise SystemExit(f"Label column '{LABEL_COL}' not found in data. Columns: {all_cols}")
# Keep everything that's not label/id/datetime; cast-to-numeric later when sampling
DROP_LIKE = {LABEL_COL, 'id', 'ID', 'datetime', 'timestamp', 'time', 'DateTime', 'Datetime'}
feature_cols = [c for c in all_cols if c not in DROP_LIKE]

# ---------- PASS 1: label counts after mapping ----------
label_counts = {}
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    src = chunk[LABEL_COL].astype(str).str.strip()
    mapped = src.map(label_to_map)
    if UNMAPPED == 'keep':
        y = mapped.fillna(src)
    elif UNMAPPED == 'not_recognized':
        y = mapped.fillna('not_recognized')
    else:  # drop
        y = mapped.dropna()
    for k, v in y.value_counts().items():
        label_counts[k] = label_counts.get(k, 0) + int(v)

if not label_counts:
    raise SystemExit("No labels found after mapping; check MAP_CSV and UNMAPPED policy.")

# Allocate per-class quotas proportional to counts, with a small floor
label_series = pd.Series(label_counts)
label_prop   = label_series / label_series.sum()
per_class_target = (label_prop * SAMPLE_MAX).clip(lower=200).astype(int).to_dict()
collected = {k: 0 for k in per_class_target}

# ---------- PASS 2: build stratified sample ----------
Xs_list, ys_list = [], []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    # numeric coercion
    Xc = chunk[feature_cols].apply(pd.to_numeric, errors='coerce')
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    src = chunk[LABEL_COL].astype(str).str.strip()
    mapped = src.map(label_to_map)

    if UNMAPPED == 'keep':
        yc = mapped.fillna(src)
    elif UNMAPPED == 'not_recognized':
        yc = mapped.fillna('not_recognized')
    else:
        keep = mapped.notna()
        Xc = Xc.loc[keep]
        yc = mapped.loc[keep]

    dfc = Xc.copy(); dfc['__y__'] = yc.values

    # take up to remaining need per class from this chunk
    takes = []
    for lbl, grp in dfc.groupby('__y__'):
        need = per_class_target.get(lbl, 0) - collected.get(lbl, 0)
        if need > 0:
            n_take = min(len(grp), need)
            if n_take > 0:
                takes.append(grp.sample(n=n_take, random_state=RANDOM_STATE))
                collected[lbl] = collected.get(lbl, 0) + n_take
    if takes:
        take = pd.concat(takes, axis=0)
        ys_list.append(take['__y__'].values)
        Xs_list.append(take.drop(columns=['__y__']).values)

    if sum(collected.values()) >= SAMPLE_MAX:
        break

if not Xs_list:
    raise SystemExit("Stratified sampling returned zero rows. Loosen SAMPLE_MAX or UNMAPPED policy.")

Xsamp = np.vstack(Xs_list)
ysamp = np.concatenate(ys_list)

# ---------- Scale the sample ----------
scaler = StandardScaler(with_mean=True, with_std=True)
Xsamp_scaled = scaler.fit_transform(Xsamp)

# ---------- t-SNE (downsampled) ----------
if Xsamp.shape[0] > TSNE_MAX:
    samp_df = pd.DataFrame(Xsamp_scaled)
    samp_df[MAPPED_COL] = ysamp
    ds = []
    for lbl, grp in samp_df.groupby(MAPPED_COL):
        frac = TSNE_MAX / Xsamp.shape[0]
        take = int(max(100, math.floor(len(grp) * frac)))
        ds.append(grp.sample(n=min(len(grp), take), random_state=RANDOM_STATE))
    ts_df = pd.concat(ds, axis=0)
    Xts = ts_df.drop(columns=[MAPPED_COL]).values
    yts = ts_df[MAPPED_COL].values
else:
    Xts, yts = Xsamp_scaled, ysamp

print(f"t-SNE on {len(yts)} points…")
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=RANDOM_STATE)
emb = tsne.fit_transform(Xts)

plt.figure(figsize=(9,7), dpi=150)
plot_df = pd.DataFrame({'TSNE1': emb[:,0], 'TSNE2': emb[:,1], MAPPED_COL: yts})
sns.scatterplot(data=plot_df, x='TSNE1', y='TSNE2', hue=MAPPED_COL, s=10, linewidth=0, alpha=0.6, legend=False)
plt.title('t-SNE – Class Overlap (mapped labels, stratified sample)')
plt.tight_layout()
plt.savefig(OUTDIR / 'tsne_overlap_mapped.png')
plt.close()

# ---------- k-NN pseudo-confusion + heatmap ----------
print("Computing k-NN pseudo-confusion…")
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='euclidean')
nbrs.fit(Xsamp_scaled)
_, idxs = nbrs.kneighbors(Xsamp_scaled)

labels_cat = pd.Categorical(ysamp)
classes = list(labels_cat.categories)

nbr_lab = ysamp[idxs[:, 1:]]  # exclude self
n_classes = len(classes)
C_counts = np.zeros((n_classes, n_classes), dtype=float)
row_counts = np.zeros(n_classes, dtype=float)

from collections import Counter
for i, y in enumerate(labels_cat.codes):
    row_counts[y] += 1
    neigh_counts = Counter(pd.Categorical(nbr_lab[i], categories=classes).codes)
    for b, cnt in neigh_counts.items():
        if b >= 0:
            C_counts[y, b] += cnt
k_eff = idxs.shape[1] - 1
C_frac = C_counts / np.maximum(row_counts[:, None] * k_eff, 1)

conf_df = pd.DataFrame(C_frac, index=classes, columns=classes)
plt.figure(figsize=(10, 8), dpi=150)
sns.heatmap(conf_df, annot=False, cmap='mako', vmin=0, vmax=1)
plt.title("k-NN Pseudo-Confusion (row = actual, col = 'neighbour class') — mapped labels")
plt.xlabel('Neighbour class (B)')
plt.ylabel('Actual class (A)')
plt.tight_layout()
plt.savefig(OUTDIR / 'knn_pseudo_confusion_heatmap.png')
plt.close()


# ----------------- 3D fluorescence scatter (interactive HTML) -----------------
print("Creating 3D fluorescence scatter…")
req_feats = ["Fl Yellow_total", "Fl Red_total", "Fl Orange_total"]
missing = [f for f in req_feats if f not in feature_cols]
if missing:
    print("Cannot create 3D fluorescence plot; missing:", missing)
else:
    import plotly.graph_objects as go
    import plotly.io as pio

    df3d = pd.DataFrame(Xsamp, columns=feature_cols)
    df3d[MAPPED_COL] = ysamp

    # Clip extreme values for nicer axes (as in your original)
    x_99 = np.percentile(df3d["Fl Yellow_total"], 99.5)
    y_99 = np.percentile(df3d["Fl Red_total"], 99.5)
    z_99 = np.percentile(df3d["Fl Orange_total"], 99.5)

    # ---- Colour + Symbol mapping per class ----
    uniq = sorted(df3d[MAPPED_COL].unique())

    # Colours: use seaborn 'husl' palette as before
    palette = sns.color_palette("husl", len(uniq)).as_hex()
    color_map = dict(zip(uniq, palette))

    # Symbols: cycle through Plotly 3D-friendly marker symbols
    base_symbols = ['circle', 'circle-open', 'cross', 'diamond',
            'diamond-open', 'square', 'square-open', 'x']
    
    # Repeat/cycle to cover many classes
    sym_list = (base_symbols * ((len(uniq) // len(base_symbols)) + 1))[:len(uniq)]
    symbol_map = dict(zip(uniq, sym_list))

    # Vectorised maps
    df3d["_color"]  = df3d[MAPPED_COL].map(color_map)
    df3d["_symbol"] = df3d[MAPPED_COL].map(symbol_map)

    # ---- Main data trace as one array-backed Scatter3d (fast for many points) ----
    scatter = go.Scatter3d(
        x=df3d["Fl Yellow_total"],
        y=df3d["Fl Red_total"],
        z=df3d["Fl Orange_total"],
        mode="markers",
        marker=dict(
            size=3,
            color=df3d["_color"],
            symbol=df3d["_symbol"],        # <= per-point symbols
            opacity=0.65,
            line=dict(width=0.5, color="rgba(20,20,20,0.5)")  # subtle outline for contrast
        ),
        text=df3d[MAPPED_COL],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Fl Yellow: %{x:.2f}<br>"
            "Fl Red: %{y:.2f}<br>"
            "Fl Orange: %{z:.2f}<extra></extra>"
        ),
        name="Data"
    )

    # ---- Legend: add one tiny (invisible in scene) trace per class so users can click to filter ----
    legend_traces = []
    for cls in uniq:
        legend_traces.append(
            go.Scatter3d(
                x=[None], y=[None], z=[None],  # no data in scene; legend only
                mode="markers",
                marker=dict(
                    size=6,
                    color=color_map[cls],
                    symbol=symbol_map[cls],
                    line=dict(width=1, color="rgba(20,20,20,0.6)")
                ),
                name=str(cls),
                showlegend=True
            )
        )

    camera = dict(
        eye=dict(x=-1.5, y=-1.5, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    fig = go.Figure(data=[scatter] + legend_traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, x_99], title="Fl Yellow_total"),
            yaxis=dict(range=[0, y_99], title="Fl Red_total"),
            zaxis=dict(range=[0, z_99], title="Fl Orange_total"),
            camera=camera
        ),
        title='3D Fluorescence Scatter (Sampled points, mapped labels)',
        height=800,
        legend=dict(
            title="Classes",
            itemsizing="trace",
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.6)"
        )
    )
    pio.write_html(fig, file=str(OUTDIR / '3d_scatter_fluorescence.html'), auto_open=False)

print("Done. Wrote:")
print(" -", OUTDIR / 'tsne_overlap_mapped.png')
print(" -", OUTDIR / 'knn_pseudo_confusion_heatmap.png')
print(" -", OUTDIR / '3d_scatter_fluorescence.html')
