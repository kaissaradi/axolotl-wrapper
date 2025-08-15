# Axolotl - Neural Spike Sorting Cluster Refinement GUI

A high-performance GUI for refining neural spike sorting clusters from Kilosort output. Uses PCA + KMeans clustering with automatic merge detection to split contaminated clusters.

## Features

- **Visualize clusters** with waveform plots, ISI histograms, and spatial analysis
- **Refine contaminated clusters** using advanced clustering algorithms
- **Interactive analysis** with real-time plots and multi-level caching
- **Kilosort integration** - works directly with Kilosort output files

## Installation

```bash
git clone https://github.com/kaissaradi/axolotl-wrapper.git
cd axolotl-wrapper
pip install -r requirements.txt
python gui.py
```

## Quick Start

1. **Load Data**: Select Kilosort output directory and corresponding `.dat/.bin` file
2. **View Clusters**: Browse clusters in the table, filter by 'good' clusters
3. **Select & Refine**: Choose a cluster and click "Refine Selected Cluster"
4. **Monitor Progress**: Watch console output for detailed refinement steps
5. **Save Results**: Overwrite original Kilosort files (with automatic backup)

## Files

- `gui.py` - Main GUI application
- `cleaning_utils_cpu.py` - Core clustering algorithms
- `requirements.txt` - Dependencies

## Dependencies

- Python 3.8+
- PyQt5, numpy, pandas, scipy, matplotlib, scikit-learn, torch

