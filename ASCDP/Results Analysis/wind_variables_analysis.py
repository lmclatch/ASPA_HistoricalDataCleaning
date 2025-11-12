#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 07:54:26 2025

@author: lizamclatchy
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load all wind data in
folder = '/Users/lizamclatchy/ASCDP/Results Analysis'
filelist = [file for file in os.listdir(folder) if file.endswith('.csv')]
#filelist = [file for file in os.listdir(folder) if file.startswith('W') and file.endswith('.csv')]
# Read files into a dictionary keyed by filename
database = {}
for file in filelist:
    filepath = os.path.join(folder, file)
    try:
        database[file] = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# Metrics to plot (exclude MAPE_% to avoid scale issues)
metrics = ['MAE', 'RMSE', 'R2']
colors = ['#3b82f6', '#10b981', '#ef4444']

# Build a single figure with one subplot per file (key)
keys = list(database.keys())
num_keys = len(keys)
if num_keys == 0:
    print("No files to plot.")
else:
    ncols = 2 if num_keys > 1 else 1
    nrows = int(np.ceil(num_keys / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        df = database[key]

        if df is None or df.empty or 'model' not in df.columns:
            ax.set_visible(False)
            continue

        models = df['model'].astype(str).tolist()
        x = np.arange(len(models))
        width = 0.8 / max(1, len(metrics))  # total width split across metrics

        # Plot each metric as an offset bar
        for m_i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric not in df.columns:
                continue
            values = df[metric].values
            offsets = x + (m_i - (len(metrics)-1)/2) * width
            ax.bar(offsets, values, width=width, label=metric, color=color)

        ax.set_title(f'{key}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylabel('Error / Score')
        ax.legend(title='Metric', fontsize=9)

    # Hide any unused subplots
    total_axes = nrows * ncols
    for j in range(num_keys, total_axes):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle('Error Metrics per Model for Each Result File', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()