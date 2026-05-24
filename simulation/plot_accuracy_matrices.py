import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def normalize_cols(matrix, invert=False):
    # Scale each column to [0, 1] for coloring
    cmin = matrix.min(axis=0)
    cmax = matrix.max(axis=0)
    # Avoid division by zero
    denom = cmax - cmin
    denom[denom == 0] = 1.0
    norm = (matrix - cmin) / denom
    if invert:
        return 1.0 - norm
    return norm

def save_heatmap_pair(matrix_left, matrix_right, labels_left, labels_right, 
                      edges, model_labels, title, filename, cmap="YlGn"):
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(title, fontsize=18)
    
    # Normalize per column for coloring
    norm_left = normalize_cols(matrix_left)
    norm_right = normalize_cols(matrix_right)
    
    # Left Plot
    sns.heatmap(norm_left, annot=matrix_left, fmt=labels_left[1], cmap=cmap, 
                xticklabels=edges, yticklabels=model_labels, ax=axes[0], 
                cbar_kws={'label': 'Relative Scale (per column)'})
    axes[0].set_title(labels_left[0])
    axes[0].set_xlabel("Target Data (Edge ID)")
    axes[0].set_ylabel("Model")
    
    # Right Plot
    sns.heatmap(norm_right, annot=matrix_right, fmt=labels_right[1], cmap=cmap, 
                xticklabels=edges, yticklabels=model_labels, ax=axes[1], 
                cbar_kws={'label': 'Relative Scale (per column)'})
    axes[1].set_title(labels_right[0])
    axes[1].set_xlabel("Target Data (Edge ID)")
    axes[1].set_ylabel("Model")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot to {filename}")

def plot_confusion_matrices(json_path: str, alpha: float, output_dir: str = "eval_out_trace/accuracy_plots"):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find the correct key for the given alpha
    target_key = None
    for key in data.keys():
        match = re.search(r'alpha([0-9.]+)', key)
        if match and float(match.group(1)) == alpha:
            target_key = key
            break
            
    if not target_key:
        print(f"Error: Could not find data for alpha={alpha} in the JSON file.")
        print(f"Available keys: {list(data.keys())}")
        return

    print(f"Processing data for: {target_key}")
    alpha_data = data[target_key]
    
    runs = list(alpha_data.keys())
    sample_metrics = alpha_data[runs[0]]['hybrid_mse_avg']['testing']['tnr']
    edge_names = set()
    for key in sample_metrics.keys():
        if "->" in key:
            src, dst = key.split('->')
            edge_names.add(dst)
    
    edges = sorted(list(edge_names), key=lambda x: int(x[1:]))
    n_edges = len(edges)
    model_labels = ["GM"] + [f"LM{e[1:]}" for e in edges]
    n_models = len(model_labels)
    
    avg_tnr_matrix = np.zeros((n_models, n_edges))
    avg_tpr_matrix = np.zeros((n_models, n_edges))
    
    for run in runs:
        metrics = alpha_data[run]['hybrid_mse_avg']['testing']
        tnr_data = metrics['tnr']
        tpr_data = metrics['tpr']
        
        for j, dst_edge in enumerate(edges):
            gm_key = f"gm->{dst_edge}"
            avg_tnr_matrix[0, j] += tnr_data.get(gm_key, 0.0)
            avg_tpr_matrix[0, j] += tpr_data.get(gm_key, 0.0)
            
            for i, src_edge in enumerate(edges):
                model_id = src_edge[1:]
                src_model = f"lm{model_id}"
                key = f"{src_model}->{dst_edge}"
                avg_tnr_matrix[i+1, j] += tnr_data.get(key, 0.0)
                avg_tpr_matrix[i+1, j] += tpr_data.get(key, 0.0)
                
    avg_tnr_matrix /= len(runs)
    avg_tpr_matrix /= len(runs)
    
    fpr_matrix = 1.0 - avg_tnr_matrix
    fnr_matrix = 1.0 - avg_tpr_matrix
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: TNR and TPR (Higher is Better -> Green)
    save_heatmap_pair(
        avg_tnr_matrix, avg_tpr_matrix, 
        ("True Negative Rate (TNR)", ".3f"), ("True Positive Rate (TPR)", ".4f"),
        edges, model_labels, 
        f"Model Performance (TNR/TPR) - Alpha = {alpha}", 
        Path(output_dir) / f"tnr_tpr_alpha_{alpha}.png",
        cmap="YlGn"
    )
    
    # Plot 2: FPR and FNR (Lower is Better -> Red/Orange for high values)
    # We'll use YlOrRd so that high errors are red
    save_heatmap_pair(
        fpr_matrix, fnr_matrix, 
        ("False Positive Rate (FPR)", ".3f"), ("False Negative Rate (FNR)", ".5f"),
        edges, model_labels, 
        f"Model Error Rates (FPR/FNR) - Alpha = {alpha}", 
        Path(output_dir) / f"fpr_fnr_alpha_{alpha}.png",
        cmap="YlOrRd"
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model performance and error rates from accuracy_matrix.json")
    parser.add_argument("--json", type=str, default="configs/accuracy_matrix.json", help="Path to accuracy_matrix.json")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value to filter by")
    parser.add_argument("--outdir", type=str, default="eval_out_trace/accuracy_plots", help="Output directory for plots")
    
    args = parser.parse_args()
    plot_confusion_matrices(args.json, args.alpha, args.outdir)
