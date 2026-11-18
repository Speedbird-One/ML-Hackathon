"""Plot helpers for PathSnap model comparison and reporting.

Generates mandatory figures required for the Technical Report:
- CV Macro F1-score comparison across models
- Student-testing Top-1 Accuracy & Macro F1-score comparison
- Log Loss comparison across models
- Confusion matrices
- Real loss curves for models with training histories (e.g., MLP)
- Formatted summary comparison table
"""
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR


def set_plot_style():
    """Configure clean plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_cv_macro_f1(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot Cross-validation Macro F1-score across all models."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    colors = sns.color_palette("viridis", len(df))
    bars = ax.barh(df["model_name"], df["cv_macro_f1"], color=colors, edgecolor="black", alpha=0.85)
    
    ax.set_xlabel("Cross-Validation Macro F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("5-Fold CV Macro F1-Score Comparison Across Models", fontsize=12, pad=12)
    ax.set_xlim(0, 1.05)
    
    # Add numerical labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    out = save_path or (RESULTS_DIR / "figures" / "cv_macro_f1_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] Saved CV Macro F1 comparison to {out}")


def plot_test_comparison(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot Student-testing Top-1 Accuracy and Macro F1 side by side."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(df))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, df["test_top1_acc"], width, label="Student-Test Top-1 Acc",
                    color="#2b5c8f", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, df["test_macro_f1"], width, label="Student-Test Macro F1",
                    color="#48a9a6", edgecolor="black", alpha=0.9)
                    
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Student-Testing Performance: Top-1 Accuracy vs Macro F1", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, loc="lower right")
    
    for rects in [rects1, rects2]:
        for bar in rects:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = save_path or (RESULTS_DIR / "figures" / "student_test_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] Saved Student-Test comparison to {out}")


def plot_log_loss_comparison(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot Log Loss comparison for models that output calibrated probabilities."""
    set_plot_style()
    valid_df = df.dropna(subset=["test_log_loss"]).copy()
    if valid_df.empty:
        return
        
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = sns.color_palette("mako", len(valid_df))
    bars = ax.bar(valid_df["model_name"], valid_df["test_log_loss"], color=colors, edgecolor="black", alpha=0.85)
    
    ax.set_ylabel("Student-Test Log Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Student-Testing Log Loss Comparison", fontsize=12, pad=12)
    ax.set_xticklabels(valid_df["model_name"], rotation=25, ha="right", fontsize=9)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    out = save_path or (RESULTS_DIR / "figures" / "log_loss_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] Saved Log Loss comparison to {out}")


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
):
    """Plot heatmap for candidate-level confusion matrix."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    
    labels = ["Candidate (0)", "True Arc (1)"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=11, pad=10)
    
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    out = save_path or (RESULTS_DIR / "figures" / f"cm_{safe_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] Saved confusion matrix for {model_name} to {out}")


def plot_mlp_loss_curve(loss_curve: List[float], save_path: Optional[Path] = None):
    """Plot actual epoch-wise training loss curve for MLP (never fabricated)."""
    if not loss_curve:
        return
        
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    
    epochs = range(1, len(loss_curve) + 1)
    ax.plot(epochs, loss_curve, color="#d95f02", lw=2, label="Training Loss")
    
    ax.set_xlabel("Epoch / Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_title("MLP Training Loss History (Actual Execution)", fontsize=12, pad=10)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    out = save_path or (RESULTS_DIR / "figures" / "mlp_training_loss_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[Plot] Saved MLP loss curve to {out}")


def generate_all_plots_and_tables(log_csv_path: Optional[Path] = None):
    """Load latest experiments log and generate all required report figures & tables."""
    csv_file = log_csv_path or (RESULTS_DIR / "logs" / "experiments.csv")
    if not csv_file.exists():
        print(f"[Plot] No experiments log found at {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    if df.empty:
        return
        
    # Keep only the latest entry per model_name
    latest_df = df.sort_values("timestamp").groupby("model_name", as_index=False).last()
    
    plot_cv_macro_f1(latest_df)
    plot_test_comparison(latest_df)
    plot_log_loss_comparison(latest_df)
    
    # Save a clean markdown table formatted exactly for the report
    table_cols = [
        "model_name", "family", "key_config", "cv_macro_f1",
        "test_top1_acc", "test_macro_f1", "test_log_loss",
        "test_route_continuity", "inference_time_ms_per_point"
    ]
    sub_df = latest_df[[c for c in table_cols if c in latest_df.columns]].copy()
    sub_df.columns = [
        "Model", "Algorithm Family", "Key Configuration", "CV Macro F1",
        "Top-1 Accuracy", "Student-Test Macro F1", "Log Loss",
        "Route Continuity", "Inference Time (ms/pt)"
    ]
    
    md_table = sub_df.to_markdown(index=False)
    table_file = RESULTS_DIR / "figures" / "model_comparison_table.md"
    table_file.parent.mkdir(parents=True, exist_ok=True)
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[Plot] Saved markdown summary table to {table_file}")
