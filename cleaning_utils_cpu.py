# --- Standard Library Imports ---
import os
from pathlib import Path

# --- Third-Party Imports ---
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Interactive Plotting Imports (for STA analysis) ---
import ipywidgets as widgets
from ipywidgets import HBox
from IPython.display import display

def extract_snippets(dat_path, spike_times, window=(-20, 60), n_channels=512, dtype='int16'):
    """
    Extracts snippets of raw data using memory-mapping for high efficiency.
    This avoids opening and seeking within the file for every individual snippet.
    
    Parameters:
        dat_path (str): Path to the .dat or .bin file.
        spike_times (np.ndarray): Array of spike center times (in samples).
        window (tuple): Tuple (before, after) in samples to define the snippet length.
        n_channels (int): Number of channels in the recording.
        dtype (str): The data type of the raw file (e.g., 'int16').
        
    Returns:
        np.ndarray: A numpy array of shape [n_channels, snippet_len, num_spikes].
    """
    snip_len = window[1] - window[0]
    spike_count = len(spike_times)
    
    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    # 1. Memory-map the file. The data is not read into RAM all at once.
    #    The raw data is treated as a 1D array and then reshaped.
    raw_data = np.memmap(dat_path, dtype=dtype, mode='r').reshape(-1, n_channels)
    total_samples = raw_data.shape[0]

    # 2. Pre-allocate the output array for the snippets.
    #    Note the initial shape for easier filling.
    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)

    # 3. Iterate through spikes and slice from the memory-mapped array.
    #    This is significantly faster than f.seek() and f.read() in a loop.
    for i, spike_time in enumerate(spike_times):
        start_sample = spike_time + window[0]
        end_sample = start_sample + snip_len

        # Boundary check to ensure snippets are not read from outside the file
        if start_sample >= 0 and end_sample < total_samples:
            # Slice the data. Shape is (snip_len, n_channels)
            snippet = raw_data[start_sample:end_sample, :]
            # Transpose to (n_channels, snip_len) and store
            snips[i, :, :] = snippet.T
            
    # 4. Transpose the final array to match the original function's output shape:
    #    (num_spikes, n_channels, snip_len) -> (n_channels, snip_len, num_spikes)
    return snips.transpose(1, 2, 0)

def compare_eis(eis, ei_template=None, max_lag=3):
    """
    Compare a list of EIs to each other or to a template.

    Parameters:
        eis         : list of [C x T] arrays (cluster EIs)
        ei_template : optional [C x T] template EI
        max_lag     : max lag to align on dominant channel

    Returns:
        sim : [k x k] similarity matrix if ei_template is None
              [k x 1] similarity vector if template is given
    """
    k = len(eis)
    if ei_template is not None:
        sim = np.zeros((k, 1))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]
            trace_t = ei_template[dom_chan, :]

            lags = np.arange(-max_lag, max_lag + 1)
            xc = correlate(trace_i, trace_t, mode='full', method='auto')
            center = len(xc) // 2
            xc_window = xc[center - max_lag:center + max_lag + 1]
            shift = lags[np.argmax(xc_window)]

            aligned_t = np.roll(ei_template, shift, axis=1)
            sim[i] = np.dot(ei_i.flatten(), aligned_t.flatten()) / (
                np.linalg.norm(ei_i) * np.linalg.norm(aligned_t))
        return sim

    else:
        sim = np.zeros((k, k))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]

            for j in range(i, k):
                ei_j = eis[j]
                trace_j = ei_j[dom_chan, :]

                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode='full', method='auto')
                center = len(xc) // 2
                xc_window = xc[center - max_lag:center + max_lag + 1]
                shift = lags[np.argmax(xc_window)]

                aligned_j = np.roll(ei_j, shift, axis=1)
                val = np.dot(ei_i.flatten(), aligned_j.flatten()) / (
                    np.linalg.norm(ei_i) * np.linalg.norm(aligned_j))
                sim[i, j] = val
                sim[j, i] = val
        return sim


def plot_ei_python(ei, positions, frame_number=0, cutoff=0.03, scale=1.0, alpha=0.5,
                   neg_color='blue', pos_color='blue', label=None, ax=None,
                   scale_ref_electrode=None):
    """
    Plot an electrical image (EI) as a spatial map of electrode amplitudes.
    """
    n_elec, T = ei.shape
    if positions.shape[0] != n_elec:
        temp_positions = np.zeros((n_elec, 2))
        min_len = min(n_elec, positions.shape[0])
        temp_positions[:min_len, :] = positions[:min_len, :]
        positions = temp_positions

    if frame_number == 0:
        ei_frame = ei[np.arange(n_elec), np.argmax(np.abs(ei), axis=1)]
    else:
        ei_frame = ei[:, frame_number]

    if scale_ref_electrode is not None:
        ref_amp = np.abs(ei_frame[scale_ref_electrode])
    else:
        ref_amp = np.max(np.abs(ei_frame))

    if ref_amp == 0:
        radii = np.zeros_like(ei_frame)
    else:
        raw_radii = scale * np.abs(ei_frame) / ref_amp
        radii = np.minimum(raw_radii, 1.0)
        radii[raw_radii < cutoff] = 0

    radii = radii * 30

    ax = ax or plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(n_elec):
        if radii[i] == 0:
            continue
        color = neg_color if ei_frame[i] < 0 else pos_color
        circle = plt.Circle(positions[i], radii[i], color=color, alpha=alpha, edgecolor='none')
        ax.add_patch(circle)

        if label is not None:
            if isinstance(label, str) and label.lower() == 'all':
                ax.text(*positions[i], str(i + 1), fontsize=6, ha='center', va='center')
            elif isinstance(label, (list, np.ndarray)) and i in label:
                ax.text(*positions[i], str(i + 1), fontsize=6, ha='center', va='center')

    padding = np.max(radii) * 1.5 if np.any(radii) else 1
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)


def plot_ei_waveforms(ei, positions, ref_channel=None, scale=1.0, ax=None,
                      colors='black', alpha=1.0, linewidth=0.5,
                      box_height=1.0, box_width=1.0):
    """
    Plot one or more EI waveforms overlaid at their spatial locations.
    """
    ax = ax or plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    if isinstance(ei, np.ndarray):
        eis = [ei]
    else:
        eis = ei

    if isinstance(colors, str):
        colors = [colors] * len(eis)

    global_max = max(np.max(np.abs(e)) for e in eis if e.size > 0)
    if global_max == 0:
        return

    t = np.linspace(-0.5, 0.5, eis[0].shape[1]) * box_width

    for ei_array, color in zip(eis, colors):
        norm_ei = (ei_array / global_max) * scale * box_height
        p2ps = norm_ei.max(axis=1) - norm_ei.min(axis=1)
        max_p2p = p2ps.max() if p2ps.size > 0 else 0
        p2p_thresh = 0.05 * max_p2p

        for i in range(ei_array.shape[0]):
            x_offset, y_offset = positions[i]
            y = norm_ei[i]
            
            this_alpha = alpha
            this_lw = linewidth
            if p2ps.size > 0 and p2ps[i] < p2p_thresh:
                this_alpha = 0.4
                this_lw = 0.4
            
            if isinstance(ref_channel, int) and i == ref_channel:
                this_alpha = 1
                this_lw = linewidth * 2

            ax.plot(t + x_offset, y + y_offset, color=color, alpha=this_alpha, linewidth=this_lw)

    pad_x, pad_y = box_width, box_height
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    extra_y_pad = box_height * scale
    ax.set_ylim(min_y - pad_y - extra_y_pad, max_y + pad_y + extra_y_pad)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)


def baseline_correct(snips, pre_samples=20):
    """Corrects for baseline shifts in an EI."""
    if snips.ndim == 3:
        # For 3D (C x T x N): Compute mean over time per channel per spike
        baseline = snips[:, :pre_samples, :].mean(axis=1)  # Shape: (C, N)
        return snips - baseline[:, np.newaxis, :]  # Broadcast to (C, T, N)
    else:
        # For 2D (C x T)
        return snips - snips[:, :pre_samples].mean(axis=1, keepdims=True)

def compute_ei(snips, pre_samples=20):
    """Computes the Electrical Image (average waveform) from snippets."""
    snips = baseline_correct(snips, pre_samples=pre_samples)
    snips_torch = torch.from_numpy(snips)
    ei = torch.mean(snips_torch, dim=2).numpy()
    return ei

def select_channels(ei, min_chan=30, max_chan=80, threshold=15):
    """Selects the most significant channels from an EI based on peak-to-peak amplitude."""
    p2p = ei.max(axis=1) - ei.min(axis=1)
    selected = np.where(p2p > threshold)[0]
    if len(selected) > max_chan:
        selected = np.argsort(p2p)[-max_chan:]
    elif len(selected) < min_chan and len(p2p) > min_chan:
        selected = np.argsort(p2p)[-min_chan:]
    return np.sort(selected)

def find_merge_groups(sim, threshold):
    """Finds groups of clusters to merge based on a similarity matrix."""
    G = nx.Graph()
    k = sim.shape[0]
    G.add_nodes_from(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > threshold:
                G.add_edge(i, j)
    return list(nx.connected_components(G))

def plot_kmeans_pca(pcs, labels):
    plt.figure(figsize=(6, 5))
    for i in np.unique(labels):
        cluster_points = pcs[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}", s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA on all concatenated waveforms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_cleaned_clusters(clusters,
                             spike_times,
                             sampling_rate,
                             dat_path,
                             ei_positions,
                             cluster_ids=None,
                             ei_scale=3,
                             ei_cutoff=0.05,
                             isi_max_ms=200,
                             template_ei=None,
                             sigma_ms=2500,
                             dt_ms=1000):
    """
    Analyzes and plots cleaned clusters. (STA analysis has been removed).
    
    Generates a figure with 3 rows of plots for each cluster:
    1. Electrical Image (EI)
    2. Inter-Spike Interval (ISI) Histogram
    3. Smoothed Firing Rate
    """
    if cluster_ids is None:
        # Sort clusters by number of spikes, descending
        sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]['inds']), reverse=True)
    else:
        # Use provided cluster IDs
        sorted_clusters = [(i, clusters[i]) for i in cluster_ids if i < len(clusters)]

    n_clusters = len(sorted_clusters)
    if n_clusters == 0:
        print("No clusters to analyze.")
        return

    # Create a figure with 3 rows for plots
    fig_height = 3
    fig = plt.figure(figsize=(4 * n_clusters, 6))
    gs = gridspec.GridSpec(fig_height, n_clusters, figure=fig)

    for col, (idx, cluster) in enumerate(sorted_clusters):
        inds = cluster['inds']
        spike_samples = spike_times[inds]
        spikes_sec = spike_samples / sampling_rate
        n_channels = ei_positions.shape[0]

        # Ensure EI is calculated
        if 'ei' not in cluster or cluster['ei'] is None:
            snips = extract_snippets(dat_path, spike_samples, n_channels=n_channels)
            ei = compute_ei(snips)
            cluster['ei'] = ei
        else:
            ei = cluster['ei']

        # --- Row 1: Plot EI ---
        ax_ei = fig.add_subplot(gs[0, col])
        title_str = f"Cluster {idx}\n({len(inds)} spikes)"
        ax_ei.set_title(title_str, fontsize=10)

        plot_ei_python(ei, ei_positions, scale=ei_scale, cutoff=ei_cutoff, pos_color='black', neg_color='red', ax=ax_ei, alpha=1.0)
        if template_ei is not None:
            sim = compare_eis([ei], template_ei)[0, 0]
            ax_ei.set_title(f"{title_str}\nSim: {sim:.2f}", fontsize=10)
            plot_ei_python(template_ei, ei_positions, scale=ei_scale, cutoff=ei_cutoff, pos_color='blue', neg_color='blue', ax=ax_ei, alpha=0.5)

        # --- Row 2: Plot ISI ---
        ax_isi = fig.add_subplot(gs[1, col])
        if len(spikes_sec) > 1:
            isi = np.diff(spikes_sec) * 1000 # convert to ms
            bins = np.arange(0, isi_max_ms + 0.5, 0.5)
            hist, _ = np.histogram(isi, bins=bins)
            fractions = hist / len(isi) if len(isi) > 0 else np.zeros_like(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax_isi.plot(bin_centers, fractions, color='blue')
            ax_isi.set_xlim(0, isi_max_ms)
        sub_spike_samples = spike_times[inds]
        isi_violation_rate = calculate_isi_violations(sub_spike_samples, sampling_rate)

        ax_isi.set_title(f"ISI ({isi_max_ms} ms)\nViolations: {isi_violation_rate:.2%}", fontsize=9)
        if col == 0:
            ax_isi.set_ylabel("Fraction of Spikes")

        # --- Row 3: Plot Smoothed Firing Rate ---
        ax_rate = fig.add_subplot(gs[2, col])
        if len(spikes_sec) > 0:
            dt = dt_ms / 1000.0
            sigma_samples = sigma_ms / dt_ms
            total_duration = spikes_sec.max()
            time_vector = np.arange(0, total_duration, dt)
            counts, _ = np.histogram(spikes_sec, bins=np.append(time_vector, time_vector[-1] + dt))
            rate = gaussian_filter1d(counts / dt, sigma=sigma_samples)
            ax_rate.plot(time_vector, rate, color='darkorange')
            ax_rate.set_xlim(0, total_duration)
            ax_rate.set_xlabel("Time (s)")
        ax_rate.set_title("Smoothed Firing Rate", fontsize=10)
        if col == 0:
            ax_rate.set_ylabel("Hz")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()



def calculate_isi_violations(spike_times_samples, sampling_rate, refractory_period_ms=2.0):
    """
    Calculates the rate of refractory period violations for a spike train.
    A violation is defined as an inter-spike interval shorter than the refractory period.
    """
    if len(spike_times_samples) < 2:
        return 0.0

    # Convert refractory period to samples
    refractory_period_samples = (refractory_period_ms / 1000.0) * sampling_rate

    # Ensure spikes are sorted before calculating intervals
    sorted_spikes = np.sort(spike_times_samples)
    isis_samples = np.diff(sorted_spikes)

    violation_count = np.sum(isis_samples < refractory_period_samples)
    violation_rate = violation_count / len(isis_samples)

    return violation_rate

# This function should be added to or replace the existing V2 function in cleaning_utils.py

# cleaning_utils.py

def refine_cluster_v2(spike_times, dat_path, channel_positions, params):
    """
    Recursively refines neural spike clusters using PCA+KMeans clustering.
    
    The algorithm:
    1. Splits clusters using PCA dimensionality reduction + KMeans
    2. Merges clusters with highly similar electrical images (EIs)
    3. Recursively applies the process until convergence
    
    Parameters:
        spike_times: Array of spike times in samples
        dat_path: Path to raw data file or pre-loaded snippets
        channel_positions: Array of channel positions
        params: Dictionary with refinement parameters
        
    Returns:
        List of refined cluster dictionaries with 'inds', 'ei', and 'channels' keys
    """
    print(f"\n=== Starting Cluster Refinement ===")
    print(f"Input spikes: {len(spike_times)}")
    
    window = params.get('window', (-20, 60))
    min_spikes = params.get('min_spikes', 100)
    k_start = params.get('k_start', 8)
    k_refine = params.get('k_refine', 2)
    ei_sim_threshold = params.get('ei_sim_threshold', 0.95)
    max_depth = params.get('max_depth', 10)
    
    print(f"Parameters: min_spikes={min_spikes}, k_start={k_start}, k_refine={k_refine}")
    print(f"EI similarity threshold: {ei_sim_threshold}, max_depth: {max_depth}")

    # Extract snippets
    if isinstance(dat_path, np.ndarray):
        snips = dat_path
        print(f"Using pre-loaded snippets: {snips.shape}")
    elif isinstance(dat_path, str):
        print(f"Loading snippets from: {dat_path}")
        snips = extract_snippets(dat_path, spike_times, window)
        print(f"Extracted snippets: {snips.shape}")
    else:
        print(f"Error: Unknown dat_path type: {type(dat_path)}")
        return []

    min_spikes = params.get('min_spikes', 100)
    k_start = params.get('k_start', 8)
    k_refine = params.get('k_refine', 2)
    ei_sim_threshold = params.get('ei_sim_threshold', 0.95)
    max_depth = params.get('max_depth', 10)

    full_inds = np.arange(snips.shape[2])
    cluster_pool = [{'inds': full_inds, 'depth': 0}]
    final_clusters = []

    while cluster_pool:
        cl = cluster_pool.pop(0)
        inds = cl['inds']
        depth = cl['depth']
        
        print(f"  [Depth {depth}] Processing cluster with {len(inds)} spikes...")

        if depth >= max_depth:
            print(f"    → Reached max depth ({max_depth}) - finalizing cluster")
            final_clusters.append({
                'inds': inds,
                'ei': compute_ei(snips[:, :, inds]),
                'channels': select_channels(compute_ei(snips[:, :, inds]))
            })
            continue

        if len(inds) < min_spikes:
            print(f"    → Cluster too small ({len(inds)} < {min_spikes}) - skipping")
            continue

        k = k_start if depth == 0 else k_refine
        print(f"    → Splitting with KMeans (k={k})...")

        snips_cl = snips[:, :, inds]
        ei = compute_ei(snips_cl)
        selected = select_channels(ei)
        print(f"    → Selected {len(selected)} channels for clustering")
        snips_sel = snips[np.ix_(selected, np.arange(snips.shape[1]), inds)]

        snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
        flat = snips_centered.transpose(2, 0, 1).reshape(len(inds), -1)
        print(f"    → Computing PCA on {flat.shape[1]} features...")
        pcs = PCA(n_components=10).fit_transform(flat)
        print(f"    → Running KMeans with {k} clusters...")
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(pcs)

        subclusters = []
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) > 0:
                sub_inds = inds[idx]
                ei_i = compute_ei(snips[:, :, sub_inds])
                subclusters.append({
                    'inds': sub_inds,
                    'ei': ei_i,
                })
        print(f"    → KMeans produced {len(subclusters)} subclusters")
        
        if len(subclusters) <= 1:
            print(f"    → Only one subcluster - finalizing as-is")
            final_clusters.append({
                'inds': inds,
                'ei': ei,
                'channels': selected
            })
            continue

        large_subclusters = [(i, cl) for i, cl in enumerate(subclusters) if len(cl['inds']) >= min_spikes]
        print(f"    → {len(large_subclusters)} subclusters have ≥{min_spikes} spikes")

        if len(large_subclusters) == 1:
            print(f"    → Only one large subcluster - finalizing")
            final_clusters.append({
                'inds': large_subclusters[0][1]['inds'],
                'ei': large_subclusters[0][1]['ei'],
                'channels': select_channels(large_subclusters[0][1]['ei'])
            })
            continue

        if len(large_subclusters) > 1:
            eis_large = [cl['ei'] for _, cl in large_subclusters]
            sim_large = compare_eis(eis_large, None)
            merge_groups = find_merge_groups(sim_large, ei_sim_threshold)
            print(f"    → EI similarity check: {len(merge_groups)} merge groups found")
            if len(merge_groups) == 1:
                print(f"    → All subclusters are mergeable - aborting split")
                final_clusters.append({
                    'inds': inds,
                    'ei': ei,
                    'channels': selected
                })
                continue

        eis = [cl['ei'] for _, cl in large_subclusters]
        sim = compare_eis(eis, None)
        groups = find_merge_groups(sim, ei_sim_threshold)

        if len(groups) == 0:
            print(f"    → No valid subclusters after merge - keeping current cluster")
            final_clusters.append({
                'inds': inds,
                'ei': ei,
                'channels': selected
            })
            continue

        print(f"    → Processing {len(groups)} merge groups for recursive refinement")
        for i, group in enumerate(groups):
            group = list(group)
            all_inds = np.concatenate([large_subclusters[j][1]['inds'] for j in group])
            print(f"      Group {i+1}: {len(all_inds)} spikes")
            if len(all_inds) >= min_spikes:
                cluster_pool.append({
                    'inds': all_inds,
                    'depth': depth + 1
                })

    print(f"\n=== Final Merge Phase ===")
    print(f"Initial clusters: {len(final_clusters)}")
    
    if len(final_clusters) == 1:
        print(f"Only one final cluster - no merging needed")
        return final_clusters

    all_final_eis = [compute_ei(snips[:, :, cl['inds']]) for cl in final_clusters]
    sim = compare_eis(all_final_eis, None)
    groups = find_merge_groups(sim, ei_sim_threshold)
    
    print(f"Final EI similarity matrix:")
    print(f"{np.round(sim, 2)}")
    print(f"Merge groups found: {len(groups)}")

    kept = []
    for i, group in enumerate(groups):
        group = list(group)
        all_inds = np.concatenate([final_clusters[j]['inds'] for j in group])
        ei_final = compute_ei(snips[:, :, all_inds])
        chans_final = select_channels(ei_final)
        kept.append({
            'inds': all_inds,
            'ei': ei_final,
            'channels': chans_final
        })
        print(f"  Final cluster {i+1}: {len(all_inds)} spikes")

    print(f"\n=== Refinement Complete ===")
    print(f"Original cluster: {snips.shape[2]} spikes")
    print(f"Final clusters: {len(kept)}")
    total_refined_spikes = sum(len(cluster['inds']) for cluster in kept)
    recovery_rate = total_refined_spikes / snips.shape[2] * 100
    
    for i, cluster in enumerate(kept):
        print(f"  Cluster {i+1}: {len(cluster['inds'])} spikes ({len(cluster['inds'])/snips.shape[2]*100:.1f}%)")
    
    print(f"Total refined spikes: {total_refined_spikes} ({recovery_rate:.1f}% recovery)")
    print(f"Spikes lost: {snips.shape[2] - total_refined_spikes} ({(1-recovery_rate/100)*100:.1f}%)")
    
    return kept

def plot_rich_ei(fig, ei, channel_positions, sampling_rate=20000, pre_samples=40):
    """
    Creates a rich, multi-panel EI visualization on a given matplotlib Figure.
    Directly inspired by the ENHANCED plots in analysis.pdf (pages 6-8).
    """
    fig.clear()

    # --- 1. Feature Extraction ---
    peak_negative = ei.min(axis=1)
    peak_times = ei.argmin(axis=1)
    peak_times_ms = (peak_times - pre_samples) / sampling_rate * 1000
    
    # Apply spatial smoothing for the amplitude map
    peak_negative_smooth = _spatial_smooth(peak_negative, channel_positions)
    
    # Define active channels
    amplitude_threshold = np.percentile(np.abs(peak_negative), 80)
    active_channels = np.abs(peak_negative) > amplitude_threshold
    active_idx = np.where(active_channels)[0]

    if active_channels.sum() == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No significant channels found.", ha='center', va='center', color='white')
        ax.set_facecolor('black')
        return

    # --- 2. Setup Enhanced Figure Layout ---
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[1,1], hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Smoothed Amplitude
    ax2 = fig.add_subplot(gs[0, 1])  # Propagation Delay
    ax3 = fig.add_subplot(gs[1, :]) # Waveform Heatmap (spanning both columns)

    # --- 3. Panel 1: Smoothed Spatial Amplitude ---
    scatter1 = ax1.scatter(
        channel_positions[:, 0], channel_positions[:, 1],
        c=peak_negative_smooth, cmap='RdBu_r', s=30,
        vmin=np.percentile(peak_negative_smooth, 5),
        vmax=np.percentile(peak_negative_smooth, 95)
    )
    fig.colorbar(scatter1, ax=ax1, label='Smoothed Peak Amp (uV)', shrink=0.8)
    ax1.set_title('Spatial Amplitude')

    # --- 4. Panel 2: Propagation Delay ---
    scatter2 = ax2.scatter(
        channel_positions[active_channels, 0], channel_positions[active_channels, 1],
        c=peak_times_ms[active_channels],
        cmap='viridis', s=80, edgecolor='white', linewidth=0.5
    )
    fig.colorbar(scatter2, ax=ax2, label='Time to Peak (ms)', shrink=0.8)
    ax2.set_title('Spike Propagation')

    # --- 5. Panel 3: Waveform Heatmap ---
    time_axis_ms = (np.arange(ei.shape[1]) - pre_samples) / sampling_rate * 1000
    if active_channels.sum() > 0:
        sorted_channel_idx = active_idx[np.argsort(peak_times[active_idx])]
        waveform_matrix = ei[sorted_channel_idx]
        
        im = ax3.imshow(waveform_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-np.percentile(np.abs(waveform_matrix), 98),
                        vmax=np.percentile(np.abs(waveform_matrix), 98),
                        extent=[time_axis_ms[0], time_axis_ms[-1], len(sorted_channel_idx), 0])
        
        ax3.axvline(0, color='black', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Channels (sorted by time)')
        ax3.set_title('Waveform Heatmap')
        fig.colorbar(im, ax=ax3, label='Amplitude (uV)', shrink=0.8, orientation='horizontal', pad=0.2)

    # --- 6. Final Styling ---
    for ax in [ax1, ax2]:
        ax.scatter(channel_positions[:, 0], channel_positions[:, 1], c='#333333', s=10, alpha=0.5, zorder=-1)
        ax.axis('equal')

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

def _spatial_smooth(values, positions, sigma=30):
    """Spatially smooth values based on channel positions, from analysis.pdf."""
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        weights = np.exp(-distances**2 / (2 * sigma**2))
        smoothed[i] = np.sum(values * weights) / np.sum(weights)
    return smoothed



def get_binned_spikes_from_kilosort(kilosort_path, target_cluster_id, M, protocol_id, noise_file_name, bins_per_frame=1, pre_filtered_spike_times=None):
    """
    Loads Kilosort spike data and bins it according to the precise frame timing
    from Symphony metadata, replicating the robust logic from sta_analysis.py.
    This function is immune to display timing jitter.

    Parameters:
        pre_filtered_spike_times (np.ndarray, optional): If provided, these spikes are used directly
                                                          instead of loading from kilosort_path and filtering by target_cluster_id.
    """
    print("--> Inside get_binned_spikes_from_kilosort function...")
    SAMPLE_RATE = 20000.0 # This should ideally come from params, or be a fixed constant

    if pre_filtered_spike_times is not None:
        target_spike_times_samples = pre_filtered_spike_times
        print(f"    Using pre-filtered spikes (count: {len(target_spike_times_samples)})")
    else:
        try:
            spike_times_all = np.load(os.path.join(kilosort_path, 'spike_times.npy')).flatten()
            spike_clusters = np.load(os.path.join(kilosort_path, 'spike_clusters.npy')).flatten()
        except FileNotFoundError as e:
            print(f"Error loading Kilosort files: {e}")
            return np.array([]), {}, 0, 0 # Return empty data on error
        target_spike_times_samples = spike_times_all[spike_clusters == target_cluster_id]
        print(f"    Loading spikes for original cluster ID {target_cluster_id} (count: {len(target_spike_times_samples)})")


    if len(target_spike_times_samples) == 0:
        print("Warning: No target spike times found. Cannot bin spikes.")
        return np.array([]), {}, 0, 0

    protocol = M.search_data_file(protocol_id, file_name=noise_file_name)
    if not protocol:
        print(f"Error: Protocol '{protocol_id}' with file name '{noise_file_name}' not found.")
        return np.array([]), {}, 0, 0

    param_names = ['preTime', 'stimTime', 'uniqueTime', 'numXChecks', 'numYChecks', 'numXStixels', 'numYStixels', 'chromaticClass', 'stepsPerStixel', 'seed', 'frameDwell']
    params, _ = M.get_stimulus_parameters(protocol, param_names)
    epoch_params = {key: val[0] for key, val in params.items()}

    all_frame_times_samples = []
    if 'group' in protocol and protocol['group']:
        for group in protocol['group']:
            if 'block' in group and group['block']:
                for block in group['block']:
                    epoch_frames_ms = M.get_frame_times_ms(block)
                    for epoch_idx in epoch_frames_ms:
                        frame_times_samples = (epoch_frames_ms[epoch_idx] / 1000.0 * SAMPLE_RATE).astype(np.int64)
                        all_frame_times_samples.extend(frame_times_samples)

    if not all_frame_times_samples:
        print("Warning: No frame times found in metadata. Cannot bin spikes.")
        return np.array([]), {}, 0, 0

    all_frame_times_samples = np.array(all_frame_times_samples)

    if len(all_frame_times_samples) == 0:
        print("Warning: No frame times to bin against.")
        return np.array([]), {}, 0, 0

    if not np.all(np.diff(all_frame_times_samples) >= 0):
        print("Warning: Frame times are not strictly increasing. Sorting them.")
        all_frame_times_samples.sort()

    binned_spikes_full = np.histogram(target_spike_times_samples, bins=all_frame_times_samples)[0]

    mean_interval_ms = np.mean(np.diff(all_frame_times_samples) / SAMPLE_RATE * 1000) if len(all_frame_times_samples) > 1 else 0
    mean_frame_rate = 1000.0 / mean_interval_ms if mean_interval_ms > 0 else 0

    unique_frames = int(np.ceil(float(epoch_params.get('uniqueTime', 0)) / 1000.0 * mean_frame_rate)) if mean_frame_rate > 0 else 0

    if bins_per_frame > 1:
        upsampled_spikes = np.zeros(len(binned_spikes_full) * bins_per_frame)
        for i in range(bins_per_frame):
            upsampled_spikes[i::bins_per_frame] = binned_spikes_full / bins_per_frame
        binned_spikes_final = upsampled_spikes
    else:
        binned_spikes_final = binned_spikes_full

    print(f"    Binned {np.sum(binned_spikes_final):.0f} spikes against {len(binned_spikes_final)} total bins.")
    return binned_spikes_final[:, np.newaxis], epoch_params, mean_frame_rate, unique_frames

# Modularized STA plotting function

def get_and_plot_sta_analysis(kilosort_output_path, target_cluster_id, raw_data_base_path, protocol_id, sta_depth, bins_per_frame, sample_rate, frame_offset,
                              experiment_name=None, noise_file_name_list=None, sub_cluster_spike_times=None): # Added new parameters
    """
    Performs STA analysis and generates interactive plots for a given Kilosort cluster or sub-cluster.

    Parameters:
        sub_cluster_spike_times (np.ndarray, optional): If provided, STA will be computed for these
                                                       specific spike times. Otherwise, it loads based on target_cluster_id.
    """
    print("--- STA Analysis Pipeline (Final Interactive Version) ---")

    # 1. Determine experiment_name and raw_data_name (your noise_file_name)
    current_experiment_name = experiment_name
    current_noise_file_name = noise_file_name_list

    if current_experiment_name is None or current_noise_file_name is None:
        kilosort_output_path_obj = Path(kilosort_output_path)
        try:
            inferred_raw_data_name_str = kilosort_output_path_obj.parent.name
            inferred_experiment_name = kilosort_output_path_obj.parent.parent.name

            if current_experiment_name is None:
                current_experiment_name = inferred_experiment_name
            if current_noise_file_name is None:
                current_noise_file_name = [inferred_raw_data_name_str] # Convert to list as expected
        except IndexError:
            print("Could not infer experiment_name or raw_data_name from kilosort_output_path. Please provide them explicitly if inference fails.")
            if current_experiment_name is None: current_experiment_name = "default_experiment" # Fallback to prevent error
            if current_noise_file_name is None: current_noise_file_name = ["default_data"] # Fallback

    if not current_experiment_name:
         print("Error: Experiment name could not be determined. Aborting STA analysis.")
         return
    if not current_noise_file_name:
        print("Error: Raw data name could not be determined. Aborting STA analysis.")
        return

    print(f"Inferred/Using Experiment Name: {current_experiment_name}")
    print(f"Inferred/Using Raw Data Name: {current_noise_file_name[0]}")

    # 2. Data Loading (or using provided spike times)
    print("--> Step 2: Preparing spikes and JSON metadata...")
    if sub_cluster_spike_times is not None:
        target_spike_times_samples = sub_cluster_spike_times
        print(f"    Using provided sub-cluster spike times (count: {len(target_spike_times_samples)})")
    else:
        try:
            spike_times_all = np.load(os.path.join(kilosort_output_path, 'spike_times.npy')).flatten()
            spike_clusters = np.load(os.path.join(kilosort_output_path, 'spike_clusters.npy')).flatten()
        except FileNotFoundError:
            print(f"Error: Could not find Kilosort files in '{kilosort_output_path}'. Aborting STA analysis.")
            return
        target_spike_times_samples = spike_times_all[spike_clusters == target_cluster_id]
        print(f"    Loading spikes for original cluster ID {target_cluster_id} (count: {len(target_spike_times_samples)})")

    if len(target_spike_times_samples) == 0:
        print(f"No spikes found for cluster/sub-cluster {target_cluster_id}. Aborting STA analysis.")
        return

    M = Metadata(experimentName=current_experiment_name)
    M.datapath = raw_data_base_path
    json_path = os.path.join(M.datapath, current_experiment_name + '.json')

    try:
        M.loadJSON(json_path)
    except FileNotFoundError:
        print(f"Error: JSON metadata file '{json_path}' not found. Aborting STA analysis.")
        return

    print("--> Step 3: Binning spikes...")
    # Pass the pre_filtered_spike_times to get_binned_spikes_from_kilosort
    binned_spikes_full_array, epoch_params, mean_frame_rate, unique_frames = \
        get_binned_spikes_from_kilosort(
            kilosort_path=kilosort_output_path,
            target_cluster_id=target_cluster_id,
            M=M,
            protocol_id=protocol_id,
            noise_file_name=current_noise_file_name[0],
            bins_per_frame=bins_per_frame,
            pre_filtered_spike_times=target_spike_times_samples # <--- Pass the actual spikes here!
        )

    if binned_spikes_full_array.size == 0:
        print("Error during spike binning. Aborting STA analysis.")
        return

    pre_bins = int(np.round(epoch_params.get('preTime', 0) / 1000.0 * mean_frame_rate))

    start_bin = pre_bins * bins_per_frame + frame_offset
    end_bin = start_bin + (unique_frames * bins_per_frame) - frame_offset

    if start_bin < 0 or end_bin > binned_spikes_full_array.shape[0]:
        print(f"Error: Calculated bin range [{start_bin}:{end_bin}] out of bounds for binned spikes (size {binned_spikes_full_array.shape[0]}). Aborting STA analysis.")
        return

    binned_spikes_for_sta = binned_spikes_full_array[start_bin:end_bin]

    print("--> Step 4: Reconstructing stimulus and calculating STA...")
    S = Stimulus()
    numXStixels = int(epoch_params.get('numXStixels', 1))
    numYStixels = int(epoch_params.get('numYStixels', 1))
    numXChecks = int(epoch_params.get('numXChecks', 1))
    numYChecks = int(epoch_params.get('numYChecks', 1))
    chromaticClass = epoch_params.get('chromaticClass', 'rgb')
    stepsPerStixel = int(epoch_params.get('stepsPerStixel', 1))
    seed = int(epoch_params.get('seed', 0))
    frameDwell = int(epoch_params.get('frameDwell', 1))

    stimulus_frames_full = S.getFastNoiseFrames(numXStixels, numYStixels, numXChecks, numYChecks, chromaticClass, unique_frames, stepsPerStixel, seed, frameDwell)

    if bins_per_frame > 1:
        stimulus_frames_full = upsample_frames(stimulus_frames_full, bins_per_frame)
    stimulus_frames_for_sta = stimulus_frames_full[frame_offset:]

    min_len = min(len(stimulus_frames_for_sta), len(binned_spikes_for_sta))
    stimulus_frames_for_sta, binned_spikes_for_sta = stimulus_frames_for_sta[:min_len], binned_spikes_for_sta[:min_len]

    if stimulus_frames_for_sta.size == 0 or binned_spikes_for_sta.size == 0:
        print("Error: Stimulus frames or binned spikes are empty after alignment. Aborting STA analysis.")
        return

    sta_single_cell = compute_sta(stimulus=stimulus_frames_for_sta.astype(np.float32), binned_spikes=binned_spikes_for_sta, depth=sta_depth).squeeze()
    sta_single_cell_flipped = np.flipud(sta_single_cell.copy())

    print("--> Step 5: Fitting receptive field...")
    hull_params = None
    peak_time_idx = 0
    try:
        if sta_single_cell.ndim >= 3:
            if sta_single_cell.ndim == 4: # (Time, Y, X, Color)
                timecourse_matrix, _, _, hull_parameters, _, _, _ = compute_sta_parameters(sta_single_cell[np.newaxis, ...])
            elif sta_single_cell.ndim == 3: # (Time, Y, X) - grayscale
                timecourse_matrix, _, _, hull_parameters, _, _, _ = compute_sta_parameters(sta_single_cell[np.newaxis, ..., np.newaxis])
            else:
                print(f"    Warning: STA single cell has unexpected dimensions {sta_single_cell.ndim}. Skipping detailed fit.")
                raise ValueError("Unexpected STA dimensions.")
        else:
            print(f"    Warning: STA single cell has insufficient dimensions ({sta_single_cell.ndim}) for RF fit. Skipping detailed fit.")
            raise ValueError("Insufficient STA dimensions.")

        time_course_clean = timecourse_matrix[0]
        hull_params = hull_parameters[0]
        x0, y0, sigma_x, sigma_y, theta = hull_params
        peak_time_idx = np.argmax(np.max(np.abs(time_course_clean), axis=1))
    except Exception as e:
        print(f"    WARNING: STA analysis failed, plotting will be limited. Error: {e}")

    # 6. Draw the static plots
    print("--> Step 6: Generating static summary plots...")
    fig_static, (ax_isi, ax_fr, ax_tc) = plt.subplots(1, 3, figsize=(15, 4))
    fig_static.suptitle(f'Summary Plots for Cluster {target_cluster_id}', fontsize=16) # Title still uses original ID

    # ISI Histogram (uses target_spike_times_samples which is now sub-cluster specific)
    isis_ms = np.diff(target_spike_times_samples) / (sample_rate / 1000)
    ax_isi.hist(isis_ms, bins=np.linspace(0, 50, 50), color='blue', density=True)
    ax_isi.set_title('ISI Histogram'); ax_isi.set_xlabel('ISI (ms)'); ax_isi.set_ylabel('Density')

    # Smoothed Firing Rate (uses target_spike_times_samples which is now sub-cluster specific)
    spike_times_sec = target_spike_times_samples / sample_rate
    if len(spike_times_sec) > 0:
        total_duration = spike_times_sec.max()
        bins = np.arange(0, total_duration, 1)
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5)
        ax_fr.plot(bins[:-1], rate, color='orange')
    ax_fr.set_title('Smoothed Firing Rate'); ax_fr.set_xlabel('Time (s)'); ax_fr.set_ylabel('Hz')

    # STA Time Course
    if hull_params is not None:
        time_per_bin_ms = 1000 / (mean_frame_rate * bins_per_frame) if (mean_frame_rate * bins_per_frame) > 0 else 0
        time_axis_ms = (np.arange(sta_depth) - (sta_depth - 1)) * time_per_bin_ms
        time_course_flipped_for_plot = np.flipud(time_course_clean)
        ax_tc.plot(time_axis_ms, time_course_flipped_for_plot[:, 0], 'r-', label='Red')
        ax_tc.plot(time_axis_ms, time_course_flipped_for_plot[:, 1], 'g-', label='Green')
        ax_tc.plot(time_axis_ms, time_course_flipped_for_plot[:, 2], 'b-', label='Blue')
        ax_tc.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_tc.set_title('STA Time Course'); ax_tc.set_xlabel('Time Before Spike (ms)'); ax_tc.legend()
    else:
        ax_tc.set_title('STA Time Course (Fit Failed)'); ax_tc.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax_tc.transAxes)

    plt.tight_layout(pad=3.0)
    plt.show()

    # 7. Create the Final Interactive Widget
    print("\n--> Step 7: Creating interactive STA frame viewer...")

    def update_interactive_plot(frame_index=0):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        frame_data = sta_single_cell_flipped[frame_index, :, :, :]

        min_val, max_val = frame_data.min(), frame_data.max()
        frame_data_norm = (frame_data - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else frame_data

        ax.imshow(frame_data_norm, cmap='gray')

        if hull_params is not None:
            ellipse = Ellipse(xy=(x0, y0), width=2 * sigma_x, height=2 * sigma_y, angle=np.rad2deg(theta), edgecolor='red', facecolor='none', lw=1.5, alpha=0.8)
            ax.add_patch(ellipse)

        ax.set_title(f'Forward Time Frame: {frame_index + 1}/{sta_depth}')
        ax.axis('off')
        plt.show()

    peak_idx_flipped = (sta_depth - 1) - peak_time_idx if hull_params is not None else 0
    frame_slider = widgets.IntSlider(value=peak_idx_flipped, min=0, max=sta_depth - 1, step=1, description='Frame:')
    prev_button = widgets.Button(description="< Prev")
    next_button = widgets.Button(description="Next >")

    def on_prev_button_clicked(b):
        if frame_slider.value > frame_slider.min:
            frame_slider.value -= 1
    def on_next_button_clicked(b):
        if frame_slider.value < frame_slider.max:
            frame_slider.value += 1

    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)

    out = widgets.interactive_output(update_interactive_plot, {'frame_index': frame_slider})

    print("Use the slider or buttons to explore the STA frames in forward time.")
    display(HBox([prev_button, frame_slider, next_button]), out)