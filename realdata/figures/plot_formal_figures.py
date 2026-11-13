#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
TIERS = ("High", "Medium", "Low", "Cub Fighter")
LOADING_ANNOTATION_MIN_ABS = 0.25



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=PACKAGE / "results", help="Results root containing full_sample/coefficients.npz.")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--metadata", type=Path, default=PACKAGE / "data" / "panel_metadata.json")
    parser.add_argument("--font", help="Optional installed font family, e.g. 'DejaVu Sans'.")
    parser.add_argument("--dpi", type=int, default=220, help="Resolution for rasterized content within PDF figures.")
    return parser.parse_args()

def normalize_tier(value):
    key = str(value).lower().replace(" ", "").replace("_", "").replace("-", "")
    names = {x.lower().replace(" ", ""): x for x in TIERS}
    if key not in names:
        raise ValueError(f"Unknown price tier: {value!r}; expected {TIERS}.")
    return names[key]

def read_inputs(results_dir, metadata_path):
    """Read category labels and store price tiers, matching coordinates by ID."""
    coefficient_path = results_dir / "full_sample" / "coefficients.npz"
    with np.load(coefficient_path, allow_pickle=False) as bundle:
        required = ("A_clients", "B", "S0", "S_local", "client_ids", "family_ids")
        missing = [key for key in required if key not in bundle]
        if missing:
            raise ValueError(f"Missing coefficient arrays: {missing}")
        arrays = {key: bundle[key].copy() for key in required}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ids = arrays["client_ids"].astype(int).tolist()
    codes = [str(code).lower() for code in arrays["family_ids"]]
    n, p = len(ids), len(codes)
    if len(set(ids)) != n or len(set(codes)) != p:
        raise ValueError("Coefficient bundle contains duplicated store or category IDs.")
    for key, shape in (("A_clients", (n, p, p)), ("S_local", (n, p, p)), ("B", (p, p)), ("S0", (p, p))):
        if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Invalid {key}: expected finite coefficients of shape {shape}.")
    reconstruction_error = float(np.max(np.abs(arrays["A_clients"] - arrays["B"] - arrays["S0"] - arrays["S_local"])))
    if reconstruction_error > 1e-8:
        raise ValueError(f"A_clients != B + S0 + S_local; max error {reconstruction_error:.3g}.")
    categories = {str(row["code"]).lower(): row for row in metadata["categories"]}
    stores = {int(row["store_id"]): dict(row) for row in metadata["stores"]}
    if set(ids) - set(stores) or set(codes) - set(categories):
        raise ValueError("Metadata must contain every store and category in the coefficient bundle.")
    if set(("cra", "coo", "sna", "lnd", "oat", "bjc")) - set(codes):
        raise ValueError("This paper's category subsets are missing from the coefficient bundle.")
    for store in ids:
        row = stores[store]
        row["price_tier"] = normalize_tier(row["price_tier"])
    for tier in TIERS:
        if not any(stores[store]["price_tier"] == tier for store in ids):
            raise ValueError(f"No stores for price tier {tier}.")
    return arrays, ids, codes, stores, categories, coefficient_path, reconstruction_error

def setup_style(font):
    if font:
        font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": font or "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "figure.titlesize": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })

def save_figure(fig, output_dir, stem, dpi, pad_inches=.1):
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    # Retain only the PDF for this figure after a successful export.
    for suffix in ("png", "svg"):
        (output_dir / f"{stem}.{suffix}").unlink(missing_ok=True)

def matrix_axes(ax, matrix, codes, *, title, cmap="RdBu_r", limit=None, norm=None):
    if norm is not None:
        im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        limit = max(float(np.max(np.abs(matrix))) if limit is None else limit, 1e-12)
        im = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    labels = [code.upper() for code in codes]
    ax.set_xticks(range(len(codes)), labels, rotation=90)
    ax.set_yticks(range(len(codes)), labels)
    ax.set_xticks(np.arange(len(codes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(codes) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.5)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Lagged predictor (previous week)")
    ax.set_ylabel("Response (current week)")
    return im

def varimax_rotation(loadings, *, gamma=1.0, max_iterations=1000, tolerance=1e-12):
    """Old analysis's orthogonal Varimax, without Kaiser normalization."""
    rows, rank = loadings.shape
    rotation = np.eye(rank)
    previous = 0.0

    def criterion(values):
        squared = values**2
        return float(np.sum(np.sum(squared**2, axis=0)
                            - gamma * np.sum(squared, axis=0)**2 / rows))

    before = criterion(loadings)
    for iteration in range(1, max_iterations + 1):
        rotated = loadings @ rotation
        column_energy = np.diag(rotated.T @ rotated)
        target = rotated**3 - (gamma / rows) * (rotated @ np.diag(column_energy))
        left, singular_values, right = np.linalg.svd(loadings.T @ target, full_matrices=False)
        rotation = left @ right
        objective = float(singular_values.sum())
        if previous > 0 and objective - previous <= tolerance * max(previous, 1.0):
            break
        previous = objective
    else:
        raise RuntimeError("Varimax did not converge within the iteration limit.")
    return rotation, {"iterations": iteration, "objective_before": before,
                      "objective_after": criterion(loadings @ rotation)}

def separate_varimax_factors(B, codes):
    """B = R H P^T, with independently rotated orthonormal R and P columns.

    This matches experiments/plot_low_rank_svd_varimax.py's coordinate convention.
    Singular values remain entirely in H; no data or factor-score scaling is used.
    """
    u, singular, vh = np.linalg.svd(B, full_matrices=False)
    rank = int(np.sum(singular > max(B.shape) * np.finfo(float).eps * singular[0])) if singular[0] else 0
    if rank != 2:
        raise ValueError(f"This paper's two-sided Varimax requires rank 2; found rank {rank}.")
    rotations, loadings, diagnostics = {}, {}, {}
    for side, basis in (("response", u[:, :rank]), ("predictor", vh[:rank].T)):
        rotation, diagnostics[side] = varimax_rotation(basis)
        cra = codes.index("cra")
        if abs((basis @ rotation)[cra, 1]) > abs((basis @ rotation)[cra, 0]):
            rotation = rotation[:, [1, 0]]
        if (basis @ rotation)[cra, 0] < 0:
            rotation[:, 0] *= -1
        anchor = ((basis @ rotation)[codes.index("lnd"), 1] if side == "predictor"
                  else (basis @ rotation)[:, 1].sum())
        if anchor < 0:
            rotation[:, 1] *= -1
        rotations[side] = rotation
        loadings[side] = basis @ rotation
    response, predictor = loadings["response"], loadings["predictor"]
    H = rotations["response"].T @ np.diag(singular[:rank]) @ rotations["predictor"]
    checks = {
        "B_minus_RHPt_max_abs": float(abs(B - response @ H @ predictor.T).max()),
        "RtR_minus_I_max_abs": float(abs(response.T @ response - np.eye(rank)).max()),
        "PtP_minus_I_max_abs": float(abs(predictor.T @ predictor - np.eye(rank)).max()),
        "H_minus_RtBP_max_abs": float(abs(H - response.T @ B @ predictor).max()),
    }
    if max(checks.values()) > 1e-12:
        raise ValueError(f"Two-sided factor decomposition failed validation: {checks}")
    biscuit = [codes.index(code) for code in ("cra", "coo", "sna")]
    response_map = response @ H
    return {
        "method": "Separate orthogonal Varimax of unit-length left/right SVD bases; no Kaiser normalization.",
        "definition": "B = R H P^T; R^T R = P^T P = I; H = Q_R^T diag(s) Q_P. R is response, P is predictor. Singular values are not absorbed into loadings.",
        "rank": rank, "displayed_components": rank, "category_order": codes,
        "singular_values": singular[:rank].tolist(),
        "response_loadings": response.tolist(), "predictor_loadings": predictor.tolist(),
        "H": H.tolist(), "H_orientation": "rows R-1,R-2; columns P-1,P-2",
        "response_rotation": rotations["response"].tolist(),
        "predictor_rotation": rotations["predictor"].tolist(),
        "coordinate_convention": "On each side, axis 1 has the larger absolute CRA loading and CRA is positive. Predictor axis 2 has positive LND; response axis 2 has positive loading sum.",
        "heatmap_column_order": ["P-1", "R-1", "P-2", "R-2"],
        "varimax_diagnostics": diagnostics, "checks": checks,
        "off_diagonal_H_squared_share": float((H[0, 1]**2 + H[1, 0]**2) / np.sum(H**2)),
        "interpretation": {
            "biscuit_codes": [codes[i] for i in biscuit],
            "predictor_biscuit_squared_loading_share": np.sum(predictor[biscuit]**2, axis=0).tolist(),
            "response_biscuit_squared_loading_share": np.sum(response[biscuit]**2, axis=0).tolist(),
            "response_positive_loading_counts": np.sum(response > 0, axis=0).tolist(),
            "response_per_unit_predictor_score": response_map.tolist(),
            "net_response_positive_counts": np.sum(response_map > 0, axis=0).tolist(),
            "scope": "Descriptive coefficient structure, not causal effects or forecast-variance explained. The complete H maps between different input and response coordinates.",
        },
    }


def decomposition_figure(factors, S0, mean_abs, codes, output_dir, dpi):
    """Compact overview with distinct palettes and readable sparse-matrix cells.

    Personalization is mean(abs(S_i)) over all stores, including zeros.
    S0 uses signed blue-light-gray-red with white cell boundaries;
    mean absolute correction uses white-purple.
    Both scales are linear, with independently labeled numerical ranges.
    Color darkness is not intended for direct cross-panel magnitude comparison.
    """
    response = np.asarray(factors["response_loadings"])
    predictor = np.asarray(factors["predictor_loadings"])
    if factors["displayed_components"] != 2:
        raise ValueError("The factor heatmap requires two fitted common factors.")
    # Independently rotated bases: the full H retains all four mappings.
    values = np.column_stack((predictor[:, 0], response[:, 0], predictor[:, 1], response[:, 1]))
    column_labels = ("P-1", "R-1", "P-2", "R-2")
    fig = plt.figure(figsize=(10.2, 3.6))
    bottom, height = .18, .7066666667
    ax = fig.add_axes([.045, bottom, .18, height])
    limit = max(float(abs(values).max()), 1e-12)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                   aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(codes)), [code.upper() for code in codes])
    ax.set_xticks(range(4), column_labels)
    ax.tick_params(which="both", length=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=10, pad=4)
    ax.set_yticks(np.arange(len(codes) + 1) - .5, minor=True)
    ax.set_xticks(np.arange(5) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.45)
    ax.axvline(1.5, color="#555555", linewidth=1.2)
    # One absolute-loading cutoff for all four columns, used only for annotation.
    for col in range(values.shape[1]):
        selected = np.flatnonzero(np.abs(values[:, col]) >= LOADING_ANNOTATION_MIN_ABS)
        for row in selected:
            value = float(values[row, col])
            rgb = np.asarray(im.cmap(im.norm(value))[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            ax.text(col, row, f"{abs(value):.2f}", ha="center", va="center",
                    fontsize=8, color="white" if luminance < .179 else "#111111")
    ax.text(0, 1.015, "(a)", transform=ax.transAxes,
            fontsize=10.5, ha="left", va="bottom")
    cax = fig.add_axes([.238, bottom, .008, height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    # Distinct palettes: S0 has light-gray zero cells with white boundaries;
    # mean absolute correction retains a white zero and purple intensity scale.
    # Independently labeled ranges preserve the structure of smaller corrections.
    coefficient_limit = max(float(np.ceil(np.max(np.abs(S0)) * 10) / 10), .1)
    magnitude_limit = max(float(np.ceil(np.max(mean_abs) * 100) / 100), .01)
    colors = plt.get_cmap("RdBu_r")(np.linspace(0, 1, 512))
    colors[255:257] = matplotlib.colors.to_rgba("#F7F7F7")
    signed_cmap = matplotlib.colors.ListedColormap(colors, name="shared_signed_coefficients")
    magnitude_colors = plt.get_cmap("Purples")(np.linspace(0, 1, 512))
    magnitude_colors[0] = (1, 1, 1, 1)
    magnitude_cmap = matplotlib.colors.ListedColormap(magnitude_colors, name="mean_absolute_corrections")
    signed_norm = matplotlib.colors.Normalize(vmin=-coefficient_limit, vmax=coefficient_limit)
    magnitude_norm = matplotlib.colors.Normalize(vmin=0, vmax=magnitude_limit)
    for matrix, x, color_x, label, cmap, norm, ticks in [
        (S0, .318, .596, "(b)", signed_cmap, signed_norm,
         np.linspace(-coefficient_limit, coefficient_limit, 7)),
        (mean_abs, .69, .968, "(c)", magnitude_cmap, magnitude_norm,
         np.linspace(0, magnitude_limit, 8)),
    ]:
        heat = fig.add_axes([x, bottom, .265, height])
        im = matrix_axes(heat, matrix, codes, title="", cmap=cmap, norm=norm)
        heat.set_ylabel("")
        heat.set_xlabel("Predictor (previous week)", fontsize=9)
        heat.tick_params(axis="both", labelsize=9)
        heat.text(0, 1.015, label, transform=heat.transAxes,
                  fontsize=10.5, ha="left", va="bottom")
        cax = fig.add_axes([color_x, bottom, .008, height])
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.ax.tick_params(labelsize=8)
    save_figure(fig, output_dir, "01_decomposition_varimax", dpi, pad_inches=.03)



def breakfast_pattern_data(S, ids, codes):
    """Preserve the 15 adjusted stores, ordered by support pattern then store ID."""
    ids=np.asarray(ids,dtype=int)
    x=S[:,codes.index('bjc'),codes.index('oat')]
    y=S[:,codes.index('oat'),codes.index('bjc')]
    masks=((x!=0)&(y==0), (x==0)&(y!=0), (x!=0)&(y!=0))
    neither=(x==0)&(y==0)
    assert np.all(np.sum(np.stack((*masks,neither)),axis=0)==1)
    groups=[np.flatnonzero(mask)[np.argsort(ids[mask])] for mask in masks]
    order=np.concatenate(groups)
    values=np.column_stack((x,y))[order]
    assert np.all(np.any(values!=0,axis=1))
    limit=max(float(np.ceil(np.max(np.abs(values))*100)/100),.01)
    summary={'only_oat_to_bjc':int(masks[0].sum()),'only_bjc_to_oat':int(masks[1].sum()),
             'both':int(masks[2].sum()),'omitted_neither':int(neither.sum()),
             'store_order':ids[order].tolist()}
    return values,groups,limit,summary


def draw_breakfast_heatmap(ax, cax, S, ids, codes, *, tick_fontsize=10, show_colorbar_label=True):
    """Show all and only the stores with a correction in either direction."""
    values,groups,limit,summary=breakfast_pattern_data(S,ids,codes)
    im=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit,
                 aspect='auto',interpolation='nearest')
    ax.set_xticks([0,1],['OAT → BJC','BJC → OAT'])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(values)),[str(i) for i in summary['store_order']])
    ax.tick_params(which='both',length=0,labelsize=tick_fontsize)
    ax.tick_params(axis='x',pad=8)
    ax.set_xticks(np.arange(3)-.5,minor=True)
    ax.set_yticks(np.arange(len(values)+1)-.5,minor=True)
    ax.grid(which='minor',color='white',linewidth=.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    position=0
    for group in groups:
        count=len(group)
        if position:
            ax.axhline(position-.5,color='#80898F',linewidth=.8)
        position+=count
    cb=ax.figure.colorbar(im,cax=cax,ticks=[-.12,-.06,0,.06,.12])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9,length=2)
    if show_colorbar_label:
        cb.set_label(r'Signed correction $S_{i,jk}$',fontsize=10,labelpad=8)
    return summary


def draw_breakfast_horizontal(ax, cax, S, ids, codes):
    """Adjacent directional rows with shared store columns and labels at left."""
    values,groups,limit,summary=breakfast_pattern_data(S,ids,codes)
    im=ax.imshow(values.T,cmap='RdBu_r',vmin=-limit,vmax=limit,
                 aspect='auto',interpolation='nearest')
    ax.set_yticks([0,1],['OAT → BJC','BJC → OAT'])
    ax.set_xticks(range(len(values)),[str(i) for i in summary['store_order']])
    ax.xaxis.tick_bottom()
    ax.tick_params(axis='x',pad=5,labelsize=10.5)
    ax.tick_params(axis='y',pad=7,labelsize=12)
    ax.set_xticks(np.arange(len(values)+1)-.5,minor=True)
    ax.set_yticks(np.arange(3)-.5,minor=True)
    ax.grid(which='minor',color='white',linewidth=.65)
    ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    position=0
    for group in groups:
        if position:
            ax.axvline(position-.5,color='#80898F',linewidth=.8)
        position+=len(group)
    cb=ax.figure.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.12,-.06,0,.06,.12])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=10.5,length=2)
    return summary



def draw_coo_magnitude(ax, S, ids, codes, group):
    """COO response-row norm: show every store, without group summary lines."""
    norms=np.linalg.norm(S[:,codes.index('coo'),:],axis=1)
    colors=('#466B9E','#339680','#D1A243','#BB5570')
    for g,tier in enumerate(TIERS):
        which=np.flatnonzero(group==tier)
        which=which[np.argsort(ids[which])]
        offsets=np.linspace(-.18,.18,len(which))
        ax.scatter(g+offsets,norms[which],s=35,facecolors=colors[g],
                   edgecolors='white',linewidths=.55,zorder=3)
    ax.set_xticks(range(4),[f'{tier}\n(n = {np.sum(group==tier)})' for tier in TIERS])
    ax.set_xlim(-.48,3.48)
    ax.set_ylim(-.006,.145)
    ax.set_ylabel(r'$\|S_i[\mathrm{COO},:]\|_2$')
    ax.grid(axis='y',color='#dddddd',linewidth=.6,alpha=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x',length=0)


def breakfast_coo_figure(S, ids, codes, stores, output_dir, dpi):
    """Two adjacent OAT/BJC rows and a shorter COO individual-store panel."""
    ids=np.asarray(ids,dtype=int)
    price_tiers=np.array([stores[int(i)]["price_tier"] for i in ids])
    with plt.rc_context({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,
                         'axes.edgecolor':'#777777','axes.linewidth':.7}):
        fig=plt.figure(figsize=(10.8,3.4))
        ax_heat=fig.add_axes([.13,.35,.44,.46])
        cax=fig.add_axes([.15,.19,.40,.03])
        draw_breakfast_horizontal(ax_heat,cax,S,ids,codes)
        ax_heat.annotate('(a)',xy=(0,1),xycoords='axes fraction',xytext=(0,3),
                         textcoords='offset points',fontsize=13,ha='left',va='bottom')
        ax_mag=fig.add_axes([.64,.18,.295,.65])
        draw_coo_magnitude(ax_mag,S,ids,codes,price_tiers)
        ax_mag.set_ylabel('')
        ax_mag.annotate('(b)',xy=(0,1),xycoords='axes fraction',xytext=(0,3),
                        textcoords='offset points',fontsize=13,ha='left',va='bottom')
        ax_mag.set_yticks(np.arange(0,.141,.02))
        ax_mag.tick_params(axis='both',labelsize=11)
        ax_mag.xaxis.label.set_fontsize(12)
        ax_mag.yaxis.label.set_fontsize(12)
        save_figure(fig,output_dir,'03_breakfast_and_coo_personalization',dpi)



def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    arrays, ids, codes, stores, categories, coefficient_path, error = read_inputs(args.results_dir, args.metadata)
    setup_style(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factors = separate_varimax_factors(arrays["B"], codes)
    mean_abs = np.mean(np.abs(arrays["S_local"]), axis=0)
    decomposition_figure(factors, arrays["S0"], mean_abs, codes, args.output_dir, args.dpi)
    breakfast_coo_figure(arrays["S_local"], ids, codes, stores, args.output_dir, args.dpi)
    print(f"Saved 2 formal PDF figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
