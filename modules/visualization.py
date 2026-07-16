"""
Bagian 9 notebook — Visualisasi, dipindahkan ke Plotly interaktif agar mudah
dipahami orang awam (hover detail, zoom). Tidak ada balloon/bubble chart
sesuai ketentuan #7 — hanya bar, line, scatter, dan error bar.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

PALETTE = ["#3B82F6", "#10B981", "#A855F7", "#EC4899", "#EF4444",
           "#F97316", "#8B5A2B", "#06B6D4", "#84CC16", "#6B7280"]


def line_metric_vs_k(df: pd.DataFrame, y_col: str, title: str, y_title: str,
                       highlight_k=None, color="#3B82F6"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["K"], y=df[y_col], mode="lines+markers",
        line=dict(color=color, width=3), marker=dict(size=9),
        name=y_title,
    ))
    if highlight_k is not None:
        fig.add_vline(x=highlight_k, line_dash="dash", line_color="#EF4444",
                       annotation_text=f"K = {highlight_k}", annotation_position="top")
    fig.update_layout(
        title=title, xaxis_title="Jumlah Cluster (K)", yaxis_title=y_title,
        template="plotly_white", height=380, margin=dict(t=60, b=40),
    )
    return fig


def errorbar_metric_vs_k(df: pd.DataFrame, mean_col: str, std_col: str, title: str,
                           y_title: str, highlight_k=None, color="#10B981"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["K"], y=df[mean_col], mode="lines+markers",
        error_y=dict(type="data", array=df[std_col], visible=True, color=color),
        line=dict(color=color, width=3), marker=dict(size=9), name=y_title,
    ))
    if highlight_k is not None:
        fig.add_vline(x=highlight_k, line_dash="dash", line_color="#EF4444",
                       annotation_text=f"K = {highlight_k}", annotation_position="top")
    fig.update_layout(
        title=title, xaxis_title="Jumlah Cluster (K)", yaxis_title=y_title,
        template="plotly_white", height=380, margin=dict(t=60, b=40),
    )
    return fig


def composite_rank_bar(df: pd.DataFrame, best_k: int, title="Peringkat Komposit Multi-Index Validation"):
    plot_data = df.sort_values("K")
    colors = ["#EF4444" if k == best_k else "#3B82F6" for k in plot_data["K"]]
    fig = go.Figure(go.Bar(
        x=plot_data["K"], y=plot_data["Mean_Rank"], marker_color=colors,
        text=plot_data["Mean_Rank"].round(2), textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Mean Rank (makin kecil makin baik)",
        template="plotly_white", height=400, margin=dict(t=60, b=40),
        xaxis=dict(tickmode="linear"),
    )
    return fig


def cluster_size_bar(labels, title="Jumlah Anggota Tiap Cluster"):
    sizes = pd.Series(labels).value_counts().sort_index()
    x_labels = [f"C{c + 1}" for c in sizes.index]
    fig = go.Figure(go.Bar(
        x=x_labels, y=sizes.values, marker_color=PALETTE[:len(sizes)],
        text=sizes.values, textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Cluster", yaxis_title="Jumlah UMKM",
        template="plotly_white", height=400, margin=dict(t=60, b=40),
    )
    return fig


def compute_pca(X_baseline, random_state=42):
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_baseline)
    var_ratio = pca.explained_variance_ratio_
    return X_pca, var_ratio


def pca_scatter_panel(X_pca, labels, title):
    labels = np.asarray(labels)
    unique_clusters = np.sort(np.unique(labels))
    fig = go.Figure()
    for i, cluster in enumerate(unique_clusters):
        idx = labels == cluster
        fig.add_trace(go.Scatter(
            x=X_pca[idx, 0], y=X_pca[idx, 1], mode="markers",
            marker=dict(size=6, color=PALETTE[i % len(PALETTE)], opacity=0.7),
            name=f"C{cluster + 1}",
        ))
        fig.add_trace(go.Scatter(
            x=[X_pca[idx, 0].mean()], y=[X_pca[idx, 1].mean()], mode="markers",
            marker=dict(size=14, color="black", symbol="x"), showlegend=False,
        ))
    fig.update_layout(
        title=title, xaxis_title="PC1", yaxis_title="PC2",
        template="plotly_white", height=400, margin=dict(t=50, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig


def numeric_distribution(df: pd.DataFrame, col: str):
    fig = px.histogram(df, x=col, nbins=30, template="plotly_white", color_discrete_sequence=["#3B82F6"])
    fig.update_layout(title=f"Sebaran {col}", height=320, margin=dict(t=50, b=40))
    return fig


def categorical_distribution(df: pd.DataFrame, col: str):
    counts = df[col].astype(str).str.strip().str.title().value_counts().reset_index()
    counts.columns = [col, "Jumlah"]
    fig = px.bar(counts, x=col, y="Jumlah", template="plotly_white",
                  color_discrete_sequence=["#10B981"])
    fig.update_layout(title=f"Sebaran {col}", height=320, margin=dict(t=50, b=40))
    return fig
