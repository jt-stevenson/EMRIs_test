import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors



def read_kernel_file(filename):
    data = np.genfromtxt(filename, comments="#", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]

    return {
        "z": data[:, 0],
        "K": data[:, 1],
    }


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

alphas = ["0.01", "0.1"]
redshifts = ["EL", "SE"]

base_dir = "outputs/protoMRD"
run_label = "B16_K18-3bb_0.0-IG25-agnostic-tau_x_1."

palette = {
    "0.01": {"EL": "tab:red", "SE": "tab:orange"},
    "0.1":  {"EL": "tab:blue", "SE": "tab:cyan"},
}

markers = {
    "0.01": {"EL": "o", "SE": "s"},
    "0.1":  {"EL": "^", "SE": "D"},
}

# choose a few zform slices to inspect in 1D
selected_zforms = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# --------------------------------------------------
# LOAD ALL KERNELS INTO MATRICES
# --------------------------------------------------

all_outputs = {}

for alpha in alphas:
    for redshift in redshifts:
        zform_list = []
        kernel_list = []
        z_ref = None

        for zform in np.round(np.arange(0.0, 15.0, 0.1), 1):
            folder = f"{base_dir}/alpha_{alpha}/{run_label}/{redshift}/{zform:.1f}"
            if not os.path.isdir(folder):
                continue

            files = glob.glob(f"{folder}/kernel_vs_z*.txt")
            if len(files) == 0:
                continue

            # assume one kernel file per folder
            file = files[0]
            out = read_kernel_file(file)

            z = out["z"]
            K = out["K"]

            if z_ref is None:
                z_ref = z
            else:
                if len(z) != len(z_ref) or not np.allclose(z, z_ref, rtol=0, atol=1e-12):
                    raise ValueError(
                        f"Inconsistent z grid in {file}. "
                        "All kernel_vs_z files must share the same z grid."
                    )

            zform_list.append(zform)
            kernel_list.append(K)

        if len(zform_list) == 0:
            print(f"Skipping empty combination alpha={alpha}, redshift={redshift}")
            continue

        zform_arr = np.array(zform_list, dtype=float)
        K_matrix = np.vstack(kernel_list)   # shape = (Nzform, Nz)

        key = (alpha, redshift)
        all_outputs[key] = {
            "z": z_ref,
            "zform": zform_arr,
            "K_matrix": K_matrix,
        }

# --------------------------------------------------
# 1D PLOT: SELECTED zform SLICES
# --------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(13, 5), sharey=True)

norm = mcolors.Normalize(vmin=0.0, vmax=10.0)
cmap = cm.viridis

for ia, alpha in enumerate(alphas):
    for redshift in redshifts:
        key = (alpha, redshift)
        if key not in all_outputs:
            continue

        z = all_outputs[key]["z"]
        zform_arr = all_outputs[key]["zform"]
        K_matrix = all_outputs[key]["K_matrix"]

        valid_rows = np.any(np.isfinite(K_matrix) & (K_matrix > 0), axis=1)
        valid_cols = np.any(np.isfinite(K_matrix) & (K_matrix > 0), axis=0)

        z = z[valid_cols]
        zform_arr = zform_arr[valid_rows]
        K_matrix = K_matrix[valid_rows][:, valid_cols]    

        
        #color = palette[alpha][redshift]
        marker = markers[alpha][redshift]

        for nzf, zf in enumerate(selected_zforms):
            idx = np.argmin(np.abs(zform_arr - zf))
            if not np.isclose(zform_arr[idx], zf, atol=0.051):
                continue

            color = cmap(norm(zform_arr[idx]))

            K_row = K_matrix[idx]
            good = np.isfinite(K_row) & (K_row > 0)

            if nzf==0:
                label = rf"{redshift}, $\alpha={alpha}$"
            else:
                label = None

            ax.plot(
                z[good],
                K_row[good],
                marker=marker,
                linestyle="-",
                color=color,
                markersize=2.5,
                linewidth=0.5,
                markevery=1,
                alpha=0.9,
                label=label,
            )

            print(np.min(z[good]))

    ax.set_xlabel("z")
    ax.set_yscale("log")

ax.set_ylabel(r"$K(z \mid z_{\rm form})$")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncols=3, frameon=False)
fig.suptitle(run_label, y=0.98)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/protoMRD_{run_label}_slices.pdf")
plt.show()

# --------------------------------------------------
# 2D HEATMAP: x = z, y = zform, color = K(z|zform)
# --------------------------------------------------

fig, axes = plt.subplots(
    len(alphas), len(redshifts),
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

if len(alphas) == 1 and len(redshifts) == 1:
    axes = np.array([[axes]])
elif len(alphas) == 1:
    axes = axes[None, :]
elif len(redshifts) == 1:
    axes = axes[:, None]

# compute global positive min/max for a consistent LogNorm
all_positive = []
for key, out in all_outputs.items():
    vals = out["K_matrix"]
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size > 0:
        all_positive.append(pos)

if len(all_positive) == 0:
    raise ValueError("No positive kernel values found for log-scale heatmap.")

all_positive = np.concatenate(all_positive)
vmin = np.nanpercentile(all_positive, 2)
vmax = np.nanpercentile(all_positive, 99.5)

for i, alpha in enumerate(alphas):
    for j, redshift in enumerate(redshifts):
        ax = axes[i, j]
        key = (alpha, redshift)

        if key not in all_outputs:
            ax.set_visible(False)
            continue

        z = all_outputs[key]["z"]
        zform_arr = all_outputs[key]["zform"]
        K_matrix = all_outputs[key]["K_matrix"]

        # mask non-positive values for LogNorm
        K_plot = np.where(K_matrix > 0, K_matrix, np.nan)

        im = ax.pcolormesh(
            zform_arr,
            z,
            K_plot.T,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )

        ax.set_xlim(zform_arr.min(), z.max())
        ax.set_ylim(z.min(), z.max())
        ax.set_title(rf"$\alpha={alpha}$, {redshift}")


for ax in axes[-1, :]:
    ax.set_xlabel(r"formation redshift $z_{\rm form}$")

for ax in axes[:, 0]:
    ax.set_ylabel("merger redshift z")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
cbar.set_label(r"$K(z \mid z_{\rm form})$")

fig.suptitle(f"Kernel map: {run_label}", y=0.98)
#fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"plots/protoMRD_{run_label}_heatmap.pdf")
plt.show()