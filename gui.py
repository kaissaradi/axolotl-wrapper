# -*- coding: utf-8 -*-
"""
axolotl GUI v3.5 (Refinement Implemented)
Developer: 100x
Date: 2025-08-14

A high-performance GUI for neural spike sorting cluster refinement and visualization.
This version re-implements the complete, multithreaded refinement and saving
functionality into the corrected and streamlined UI.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from collections import deque

# --- Core QtPy Imports ---
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QPushButton, QFileDialog, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget, QLabel, QFrame
)
from qtpy.QtCore import (
    QAbstractTableModel, Qt, QModelIndex, QObject, QThread, Signal
)
from qtpy.QtGui import QFont

# --- Visualization Libraries ---
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Custom Utility Library ---
try:
    import cleaning_utils_cpu
except ImportError:
    app = QApplication(sys.argv)
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setText("CRITICAL ERROR: 'cleaning_utils_cpu.py' not found.")
    error_box.setInformativeText("Please place the cleaning_utils_cpu.py file in the same directory as this application.")
    error_box.setWindowTitle("Missing Dependency")
    error_box.exec()
    sys.exit(1)

# --- Global Configuration ---
pg.setConfigOption('background', '#1f1f1f')
pg.setConfigOption('foreground', 'd')

# =============================================================================
# 0. BACKGROUND WORKERS
# =============================================================================
class SpatialWorker(QObject):
    """
    Runs in a separate thread to compute heavyweight features without freezing the UI.
    """
    result_ready = Signal(int, dict)

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.is_running = True
        self.queue = deque()

    def run(self):
        while self.is_running:
            if self.queue:
                cluster_id = self.queue.popleft()
                if cluster_id not in self.data_manager.heavyweight_cache:
                    features = self.data_manager.get_heavyweight_features(cluster_id)
                    if features:
                        self.result_ready.emit(cluster_id, features)
            else:
                QThread.msleep(100)

    def add_to_queue(self, cluster_id, high_priority=False):
        if cluster_id in self.queue: return
        if high_priority: self.queue.appendleft(cluster_id)
        else: self.queue.append(cluster_id)

    def stop(self):
        self.is_running = False

class RefinementWorker(QObject):
    """
    Runs the `refine_cluster_v2` function in a background thread.
    """
    finished = Signal(int, list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id

    def run(self):
        try:
            spike_times_cluster = self.data_manager.get_cluster_spikes(self.cluster_id)
            params = {'min_spikes': 500, 'ei_sim_threshold': 0.90}
            refined_clusters = cleaning_utils_cpu.refine_cluster_v2(
                spike_times_cluster,
                str(self.data_manager.dat_path),
                self.data_manager.channel_positions,
                params
            )
            self.finished.emit(self.cluster_id, refined_clusters)
        except Exception as e:
            self.error.emit(f"Refinement failed for cluster {self.cluster_id}: {str(e)}")

# =============================================================================
# 1. MODEL: DataManager - The Brains of the Operation
# =============================================================================
class DataManager(QObject):
    """
    Manages all data loading, processing, and caching.
    """
    is_dirty = False

    def __init__(self, kilosort_dir):
        super().__init__()
        self.kilosort_dir = Path(kilosort_dir)
        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.dat_path = None
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195

    def load_kilosort_data(self):
        try:
            self.spike_times = np.load(self.kilosort_dir / 'spike_times.npy').flatten()
            self.spike_clusters = np.load(self.kilosort_dir / 'spike_clusters.npy').flatten()
            self.channel_positions = np.load(self.kilosort_dir / 'channel_positions.npy')
            info_path = self.kilosort_dir / 'cluster_info.tsv'
            group_path = self.kilosort_dir / 'cluster_group.tsv'
            if info_path.exists():
                self.info_path = info_path
                self.cluster_info = pd.read_csv(info_path, sep='\t')
            elif group_path.exists():
                self.info_path = group_path
                self.cluster_info = pd.read_csv(group_path, sep='\t')
            else:
                raise FileNotFoundError("'cluster_info.tsv' or 'cluster_group.tsv' not found.")
            self._load_kilosort_params()
            return True, "Successfully loaded Kilosort data."
        except Exception as e:
            return False, f"Error during Kilosort data loading: {e}"

    def _load_kilosort_params(self):
        params_path = self.kilosort_dir / 'params.py'
        if not params_path.exists(): raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = map(str.strip, line.split('=', 1))
                    try: params[key] = eval(val)
                    except (NameError, SyntaxError): params[key] = val.strip("'\"")
        self.sampling_rate = params.get('fs', 30000)
        self.n_channels = params.get('n_channels_dat', 512)
        dat_path_str = params.get('dat_path', '')
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str: dat_path_str = dat_path_str[0]
        suggested_path = Path(dat_path_str)
        if not suggested_path.is_absolute():
            self.dat_path_suggestion = self.kilosort_dir.parent / suggested_path
        else:
            self.dat_path_suggestion = suggested_path

    def build_cluster_dataframe(self):
        cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        df = pd.DataFrame({'cluster_id': cluster_ids, 'n_spikes': n_spikes})
        isi_rates_dict = self._calculate_all_isi_violations_vectorized()
        df['isi_violations_pct'] = df['cluster_id'].map(isi_rates_dict).fillna(0)
        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns: self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')
        df['status'] = 'Original'
        self.cluster_df = df[['cluster_id', 'KSLabel', 'n_spikes', 'isi_violations_pct', 'status']]
        self.original_cluster_df = self.cluster_df.copy()

    def _calculate_all_isi_violations_vectorized(self, refractory_period_ms=2.0):
        rates_dict = {}
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        for cid in np.unique(self.spike_clusters):
            cluster_spikes = np.sort(self.spike_times[self.spike_clusters == cid])
            if len(cluster_spikes) < 2:
                rates_dict[cid] = 0.0
                continue
            isis = np.diff(cluster_spikes)
            violations = np.sum(isis < refractory_period_samples)
            rates_dict[cid] = (violations / (len(cluster_spikes) - 1)) * 100
        return rates_dict

    def get_cluster_spikes(self, cluster_id):
        return self.spike_times[self.spike_clusters == cluster_id]

    def get_lightweight_features(self, cluster_id, max_spikes_for_vis=100, n_raw_snippets=30):
        if cluster_id in self.ei_cache: return self.ei_cache[cluster_id]
        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) == 0: return None
        sample_size = min(len(spike_times_cluster), max_spikes_for_vis)
        spike_sample = np.random.choice(spike_times_cluster, size=sample_size, replace=False)
        snippets_raw = cleaning_utils_cpu.extract_snippets(str(self.dat_path), spike_sample, n_channels=self.n_channels)
        snippets_uV = snippets_raw.astype(np.float32) * self.uV_per_bit
        if snippets_uV.shape[2] == 0: return None
        pre_samples = 20
        snippets_bc = cleaning_utils_cpu.baseline_correct(snippets_uV, pre_samples=pre_samples)
        mean_ei = cleaning_utils_cpu.compute_ei(snippets_bc, pre_samples=pre_samples)
        features = {'mean_ei': mean_ei, 'raw_snippets': snippets_bc[:, :, :min(n_raw_snippets, snippets_bc.shape[2])]}
        self.ei_cache[cluster_id] = features
        return features

    def get_heavyweight_features(self, cluster_id):
        if cluster_id in self.heavyweight_cache: return self.heavyweight_cache[cluster_id]
        lightweight_data = self.get_lightweight_features(cluster_id)
        if not lightweight_data: return None
        features = cleaning_utils_cpu.compute_spatial_features(
            lightweight_data['mean_ei'], self.channel_positions, self.sampling_rate)
        self.heavyweight_cache[cluster_id] = features
        return features
        
    def update_after_refinement(self, parent_id, new_clusters_data):
        self.is_dirty = True
        parent_indices = np.where(self.spike_clusters == parent_id)[0]
        self.cluster_df.loc[self.cluster_df['cluster_id'] == parent_id, 'status'] = 'Refined (Parent)'
        max_id = self.spike_clusters.max()
        new_rows = []
        for i, new_cluster in enumerate(new_clusters_data):
            new_id = max_id + 1 + i
            sub_indices = parent_indices[new_cluster['inds']]
            self.spike_clusters[sub_indices] = new_id
            isi_violations = self._calculate_isi_violations(new_id)
            new_row = {
                'cluster_id': new_id, 'KSLabel': 'good', 'n_spikes': len(sub_indices),
                'isi_violations_pct': isi_violations, 'status': f'Refined (from C{parent_id})'
            }
            new_rows.append(new_row)
        self.cluster_df = pd.concat([self.cluster_df, pd.DataFrame(new_rows)], ignore_index=True)

    def _calculate_isi_violations(self, cluster_id, refractory_period_ms=2.0):
        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2: return 0.0
        isis = np.diff(np.sort(spike_times_cluster))
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        violations = np.sum(isis < refractory_period_samples)
        return (violations / (len(spike_times_cluster) - 1)) * 100

# =============================================================================
# 2. VIEW: MainWindow & Supporting Widgets
# =============================================================================
class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.set_dataframe(dataframe)

    def set_dataframe(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()): return len(self._dataframe)
    def columnCount(self, parent=QModelIndex()): return len(self._dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._dataframe.iloc[index.row(), index.column()]
            if pd.isna(value): return ""
            if isinstance(value, float): return f"{value:.2f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal: return str(self._dataframe.columns[section])
            if orientation == Qt.Orientation.Vertical: return str(self._dataframe.index[section])
        return None

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        colname = self._dataframe.columns[column]
        self._dataframe.sort_values(colname, ascending=(order == Qt.SortOrder.AscendingOrder), inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1f1f1f')
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("axolotl")
        self.setGeometry(50, 50, 1800, 1000)
        self.data_manager = None
        self.refine_thread = None
        self.spatial_plot_dirty = False
        self.worker_thread = None
        self.spatial_worker = None
        self.current_spatial_features = None
        self._setup_style()
        self._setup_ui()
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage("Welcome to axolotl. Please load a Kilosort directory to begin.")

    def _setup_style(self):
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet("""
            QWidget { color: white; background-color: #2D2D2D; }
            QTableView { background-color: #191919; alternate-background-color: #252525; gridline-color: #454545; }
            QHeaderView::section { background-color: #353535; padding: 4px; border: 1px solid #555555; }
            QPushButton { background-color: #353535; border: 1px solid #555555; padding: 5px; border-radius: 4px; }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #252525; }
            QTabWidget::pane { border: 1px solid #4282DA; }
            QTabBar::tab { color: white; background: #353535; padding: 8px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #4282DA; }
            QStatusBar { color: white; }
        """)

    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_pane.setFixedWidth(450)
        filter_box = QHBoxLayout()
        self.filter_button = QPushButton("Filter 'Good'")
        self.reset_button = QPushButton("Reset View")
        filter_box.addWidget(self.filter_button)
        filter_box.addWidget(self.reset_button)
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.refine_button = QPushButton("Refine Selected Cluster")
        self.refine_button.setFixedHeight(40)
        self.refine_button.setStyleSheet("font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;")
        left_layout.addLayout(filter_box)
        left_layout.addWidget(self.table_view)
        left_layout.addWidget(self.refine_button)
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        self.analysis_tabs = QTabWidget()
        self.waveforms_tab = QWidget()
        waveforms_layout = QVBoxLayout(self.waveforms_tab)
        wf_splitter = QSplitter(Qt.Orientation.Vertical)
        self.waveform_plot = pg.PlotWidget(title="Waveforms (Sampled)")
        bottom_panel_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.isi_plot = pg.PlotWidget(title="Inter-Spike Interval (ISI) Histogram")
        self.fr_plot = pg.PlotWidget(title="Smoothed Firing Rate")
        bottom_panel_splitter.addWidget(self.isi_plot)
        bottom_panel_splitter.addWidget(self.fr_plot)
        wf_splitter.addWidget(self.waveform_plot)
        wf_splitter.addWidget(bottom_panel_splitter)
        wf_splitter.setSizes([600, 400])
        waveforms_layout.addWidget(wf_splitter)
        self.analysis_tabs.addTab(self.waveforms_tab, "Waveform Details")
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        summary_layout.addWidget(self.summary_canvas)
        self.analysis_tabs.addTab(self.summary_tab, "Spatial Analysis")
        right_layout.addWidget(self.analysis_tabs)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setSizes([450, 1350])
        main_layout.addWidget(splitter)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        load_action = file_menu.addAction("&Load Kilosort Directory...")
        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)
        
        # --- Connect Signals to Slots ---
        load_action.triggered.connect(self.load_directory)
        self.save_action.triggered.connect(self.on_save_action)
        self.filter_button.clicked.connect(self._apply_good_filter)
        self.reset_button.clicked.connect(self._reset_table_view)
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)
        self.summary_canvas.fig.canvas.mpl_connect('motion_notify_event', self._on_summary_plot_hover)
        self.refine_button.clicked.connect(self.on_refine_cluster)

    # --- Controller & Worker Logic ---
    def load_directory(self):
        ks_dir_name = QFileDialog.getExistingDirectory(self, "Select Kilosort Output Directory")
        if not ks_dir_name: return
        self.status_bar.showMessage("Loading Kilosort files...")
        QApplication.processEvents()
        self.data_manager = DataManager(ks_dir_name)
        success, message = self.data_manager.load_kilosort_data()
        if not success:
            QMessageBox.critical(self, "Loading Error", message)
            self.status_bar.showMessage("Loading failed.", 5000)
            return
        self.status_bar.showMessage("Kilosort files loaded. Please select the raw data file.")
        QApplication.processEvents()
        dat_file, _ = QFileDialog.getOpenFileName(self, "Select Raw Data File (.dat or .bin)",
                                                  str(self.data_manager.dat_path_suggestion.parent),
                                                  "Binary Files (*.dat *.bin)")
        if not dat_file:
            self.status_bar.showMessage("Data loading cancelled by user.", 5000)
            return
        self.data_manager.dat_path = Path(dat_file)
        self.status_bar.showMessage("Building cluster dataframe...")
        QApplication.processEvents()
        self.data_manager.build_cluster_dataframe()
        self._setup_gui_with_data()
        self._start_worker()
        self.central_widget.setEnabled(True)
        self.status_bar.showMessage(f"Successfully loaded {len(self.data_manager.cluster_df)} clusters.", 5000)

    def _start_worker(self):
        if self.worker_thread is not None: self._stop_worker()
        self.worker_thread = QThread()
        self.spatial_worker = SpatialWorker(self.data_manager)
        self.spatial_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.spatial_worker.run)
        self.spatial_worker.result_ready.connect(self.on_spatial_data_ready)
        self.worker_thread.start()

    def _stop_worker(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.spatial_worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()

    def closeEvent(self, event):
        if self.data_manager and self.data_manager.is_dirty:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                "You have unsaved refinement changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                self.on_save_action()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        self._stop_worker()
        event.accept()

    def _setup_gui_with_data(self):
        self.pandas_model = PandasModel(self.data_manager.cluster_df)
        self.table_view.setModel(self.pandas_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.selectionModel().selectionChanged.connect(self.on_cluster_selection_changed)

    def on_cluster_selection_changed(self):
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None: return
        self.status_bar.showMessage(f"Loading data for Cluster ID: {cluster_id}", 2000)
        QApplication.processEvents()
        lightweight_features = self.data_manager.get_lightweight_features(cluster_id)
        if lightweight_features is None:
            self.status_bar.showMessage(f"Could not generate EI for cluster {cluster_id}.", 3000)
            self.waveform_plot.clear(); self.isi_plot.clear(); self.fr_plot.clear()
            self.summary_canvas.fig.clear()
            self.summary_canvas.fig.text(0.5, 0.5, "Select a cluster", ha='center', va='center', color='gray')
            self.summary_canvas.draw()
            return
        self._update_waveform_plot(cluster_id, lightweight_features)
        self._update_isi_plot(cluster_id)
        self._update_fr_plot(cluster_id)
        self.spatial_plot_dirty = True
        self.summary_canvas.fig.clear()
        self.summary_canvas.fig.text(0.5, 0.5, "Click 'Spatial Analysis' tab to load", ha='center', va='center', color='gray')
        self.summary_canvas.draw()
        if self.spatial_worker: self.spatial_worker.add_to_queue(cluster_id, high_priority=False)
        self.on_tab_changed(self.analysis_tabs.currentIndex())
        self.status_bar.showMessage("Ready.", 2000)

    def on_tab_changed(self, index):
        is_summary_tab = self.analysis_tabs.widget(index) == self.summary_tab
        if is_summary_tab and self.spatial_plot_dirty:
            cluster_id = self._get_selected_cluster_id()
            if cluster_id is None: return
            if cluster_id in self.data_manager.heavyweight_cache:
                self._draw_summary_plot(cluster_id)
                self.spatial_plot_dirty = False
                return
            self.status_bar.showMessage(f"Requesting spatial analysis for C{cluster_id}...", 3000)
            self.summary_canvas.fig.clear()
            self.summary_canvas.fig.text(0.5, 0.5, f"Loading C{cluster_id}...", ha='center', va='center', color='white')
            self.summary_canvas.draw()
            QApplication.processEvents()
            if self.spatial_worker: self.spatial_worker.add_to_queue(cluster_id, high_priority=True)
            self.spatial_plot_dirty = False

    def on_spatial_data_ready(self, cluster_id, features):
        current_id = self._get_selected_cluster_id()
        current_tab_widget = self.analysis_tabs.currentWidget()
        if cluster_id == current_id and current_tab_widget == self.summary_tab:
            self._draw_summary_plot(cluster_id)
            self.status_bar.showMessage("Spatial analysis complete.", 2000)

    # --- Refinement Logic ---
    def on_refine_cluster(self):
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            QMessageBox.warning(self, "No Cluster Selected", "Please select a cluster from the table to refine.")
            return
        self.refine_button.setEnabled(False)
        self.status_bar.showMessage(f"Starting refinement for Cluster {cluster_id}...")
        self.refine_thread = QThread()
        self.refinement_worker = RefinementWorker(self.data_manager, cluster_id)
        self.refinement_worker.moveToThread(self.refine_thread)
        self.refinement_worker.finished.connect(self.handle_refinement_results)
        self.refinement_worker.error.connect(self.handle_refinement_error)
        self.refinement_worker.progress.connect(lambda msg: self.status_bar.showMessage(msg, 3000))
        self.refine_thread.started.connect(self.refinement_worker.run)
        self.refine_thread.start()

    def handle_refinement_results(self, parent_id, new_clusters):
        self.status_bar.showMessage(f"Refinement of C{parent_id} complete. Found {len(new_clusters)} sub-clusters.", 5000)
        self.data_manager.update_after_refinement(parent_id, new_clusters)
        self.pandas_model.set_dataframe(self.data_manager.cluster_df)
        self.refine_button.setEnabled(True)
        self.save_action.setEnabled(True)
        self.setWindowTitle("*axolotl (unsaved changes)")
        self.refine_thread.quit()
        self.refine_thread.wait()

    def handle_refinement_error(self, error_message):
        QMessageBox.critical(self, "Refinement Error", error_message)
        self.status_bar.showMessage("Refinement failed.", 5000)
        self.refine_button.setEnabled(True)
        self.refine_thread.quit()
        self.refine_thread.wait()

    # --- File Saving Logic ---
    def on_save_action(self):
        if self.data_manager and self.data_manager.info_path:
            original_path = self.data_manager.info_path
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Refined Cluster Info",
                str(original_path.parent / f"{original_path.stem}_refined.tsv"), "TSV Files (*.tsv)")
            if save_path:
                self.save_results(save_path)

    def save_results(self, output_path):
        try:
            col = 'KSLabel' if 'KSLabel' in self.data_manager.cluster_info.columns else 'group'
            final_df = self.data_manager.cluster_df[['cluster_id', 'KSLabel']].copy()
            final_df.rename(columns={'KSLabel': col}, inplace=True)
            final_df.to_csv(output_path, sep='\t', index=False)
            self.data_manager.is_dirty = False
            self.setWindowTitle("axolotl")
            self.save_action.setEnabled(False)
            self.status_bar.showMessage(f"Results saved to {output_path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the file: {e}")
            self.status_bar.showMessage("Save failed.", 5000)

    # --- Plotting Methods ---
    def _draw_summary_plot(self, cluster_id):
        lightweight_features = self.data_manager.get_lightweight_features(cluster_id)
        heavyweight_features = self.data_manager.get_heavyweight_features(cluster_id)
        self.current_spatial_features = heavyweight_features
        if lightweight_features is None or heavyweight_features is None:
            self.summary_canvas.fig.clear()
            self.summary_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            self.summary_canvas.draw()
            return
        self.summary_canvas.fig.clear()
        cleaning_utils_cpu.plot_rich_ei(
            self.summary_canvas.fig, lightweight_features['mean_ei'], self.data_manager.channel_positions,
            heavyweight_features, self.data_manager.sampling_rate, pre_samples=20)
        self.summary_canvas.fig.suptitle(f"Cluster {cluster_id} Spatial Analysis", color='white', fontsize=16)
        self.summary_canvas.draw()

    def _on_summary_plot_hover(self, event):
        if (event.inaxes is None or self.data_manager is None or self.current_spatial_features is None): return
        if event.inaxes == self.summary_canvas.fig.axes[0]:
            positions = self.data_manager.channel_positions
            ptp_amps = self.current_spatial_features.get('ptp_amps')
            if ptp_amps is None: return
            mouse_pos = np.array([[event.xdata, event.ydata]])
            distances = cdist(mouse_pos, positions)[0]
            if distances.min() < 20:
                closest_idx = distances.argmin()
                ptp = ptp_amps[closest_idx]
                self.status_bar.showMessage(f"Channel ID {closest_idx}: PTP = {ptp:.2f} µV")

    def _update_waveform_plot(self, cluster_id, lightweight_features):
        self.waveform_plot.clear()
        self.waveform_plot.setTitle(f"Cluster {cluster_id} | Waveforms (Sampled)")
        mean_ei = lightweight_features['mean_ei']
        snippets = lightweight_features['raw_snippets']
        p2p = mean_ei.max(axis=1) - mean_ei.min(axis=1)
        dom_chan = np.argmax(p2p)
        pre_peak_samples = 20
        time_axis = (np.arange(mean_ei.shape[1]) - pre_peak_samples) / self.data_manager.sampling_rate * 1000
        for i in range(snippets.shape[2]):
            self.waveform_plot.plot(time_axis, snippets[dom_chan, :, i], pen=pg.mkPen(color=(200, 200, 200, 30)))
        self.waveform_plot.plot(time_axis, mean_ei[dom_chan], pen=pg.mkPen('#00A3E0', width=2.5))
        self.waveform_plot.setLabel('bottom', 'Time (ms)')
        self.waveform_plot.setLabel('left', 'Amplitude (uV)')

    def _update_isi_plot(self, cluster_id):
        self.isi_plot.clear()
        violation_rate = self.data_manager._calculate_isi_violations(cluster_id)
        self.isi_plot.setTitle(f"Cluster {cluster_id} | ISI | Violations: {violation_rate:.2f}%")
        spikes = self.data_manager.get_cluster_spikes(cluster_id)
        if len(spikes) < 2: return
        isis_ms = np.diff(np.sort(spikes)) / self.data_manager.sampling_rate * 1000
        y, x = np.histogram(isis_ms, bins=np.linspace(0, 50, 101))
        self.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
        self.isi_plot.addLine(x=2.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
        self.isi_plot.setLabel('bottom', 'ISI (ms)')
        self.isi_plot.setLabel('left', 'Count')
        
    def _update_fr_plot(self, cluster_id):
        self.fr_plot.clear()
        self.fr_plot.setTitle(f"Cluster {cluster_id} | Firing Rate")
        spikes_sec = self.data_manager.get_cluster_spikes(cluster_id) / self.data_manager.sampling_rate
        if len(spikes_sec) == 0: return
        total_duration = self.data_manager.spike_times.max() / self.data_manager.sampling_rate
        bins = np.arange(0, total_duration + 1, 1)
        counts, _ = np.histogram(spikes_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5)
        self.fr_plot.plot(bins[:-1], rate, pen='y')
        self.fr_plot.setLabel('bottom', 'Time (s)')
        self.fr_plot.setLabel('left', 'Firing Rate (Hz)')

    # --- Helper & UI State Methods ---
    def _get_selected_cluster_id(self):
        if not self.table_view.selectionModel().hasSelection(): return None
        row = self.table_view.selectionModel().selectedRows()[0].row()
        return self.pandas_model._dataframe.iloc[row]['cluster_id']

    def _apply_good_filter(self):
        if self.data_manager is None: return
        filtered_df = self.data_manager.original_cluster_df[
            self.data_manager.original_cluster_df['KSLabel'] == 'good'
        ].copy()
        self.pandas_model.set_dataframe(filtered_df)

    def _reset_table_view(self):
        if self.data_manager is None: return
        self.pandas_model.set_dataframe(self.data_manager.original_cluster_df)

# =============================================================================
# 3. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())