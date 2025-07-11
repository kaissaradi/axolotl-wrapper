# -*- coding: utf-8 -*-
"""
axolotl GUI v2.7
Developer: 100x
Date: 2025-07-03

A high-performance GUI for neural spike sorting cluster refinement and visualization.
This version implements a robust multi-level caching system (in-memory and on-disk)
and true lazy loading for expensive plots, ensuring a maximally responsive UI.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d

# --- Core QtPy Imports ---
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QPushButton, QFileDialog, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget, QLabel, QFrame
)
from qtpy.QtCore import (
    QAbstractTableModel, Qt, QModelIndex, QObject, QThread, Signal
)
from qtpy.QtGui import QPalette, QColor, QFont

# --- Visualization Libraries ---
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Custom Utility Library ---
try:
    import cleaning_utils_cpu
except ImportError:
    # Critical error if the utility library is missing
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
# 1. MODEL: DataManager - The Brains of the Operation
# =============================================================================
class DataManager(QObject):
    """
    Manages all data loading, processing, and caching (in-memory and on-disk).
    This is the 'Model' in our MVC architecture.
    """
    is_dirty = False

    def __init__(self, kilosort_dir):
        super().__init__()
        self.kilosort_dir = Path(kilosort_dir)
        self.cache_path = self.kilosort_dir / 'axolotl_cache.npy'
        self.ei_cache = {} # In-memory cache
        self.dat_path = None
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None

    def load_disk_cache(self):
        """Loads the persistent on-disk EI cache if it exists."""
        if self.cache_path.exists():
            backup_path = self.cache_path.with_suffix('.npy.bak')
            if not backup_path.exists():
                self.cache_path.rename(backup_path)
            try:
                self.ei_cache = np.load(self.cache_path, allow_pickle=True).item()
                return True
            except Exception as e:
                pass  # Cache file might be corrupt, will recompute
        self.ei_cache = {}  # Reset to empty to force recompute
        return False

    def save_disk_cache(self):
        """Saves the in-memory EI cache to a persistent file on disk."""
        try:
            np.save(self.cache_path, self.ei_cache)
        except Exception as e:
            pass  # Cache save failed, will continue without caching

    def load_kilosort_data(self):
        """Loads all Kilosort-related files synchronously."""
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
        """Parses the params.py file to get critical recording metadata."""
        params_path = self.kilosort_dir / 'params.py'
        if not params_path.exists(): raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = map(str.strip, line.split('=', 1))
                    try:
                        params[key] = eval(val)
                    except (NameError, SyntaxError):
                        params[key] = val.strip("'\"")
        self.sampling_rate = params.get('fs', 30000)
        self.n_channels = params.get('n_channels_dat', 512)
        dat_path_str = params.get('dat_path', '')
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str:
            dat_path_str = dat_path_str[0]
        suggested_path = Path(dat_path_str)
        if not suggested_path.is_absolute():
            self.dat_path_suggestion = self.kilosort_dir.parent / suggested_path
        else:
            self.dat_path_suggestion = suggested_path

    def _calculate_all_isi_violations_vectorized(self, refractory_period_ms=2.0):
        rates_dict = {}
        all_cluster_ids = np.unique(self.spike_clusters)
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        for cid in all_cluster_ids:
            cluster_spikes = np.sort(self.spike_times[self.spike_clusters == cid])
            if len(cluster_spikes) < 2:
                rates_dict[cid] = 0.0
                continue
            isis = np.diff(cluster_spikes)
            violations = np.sum(isis < refractory_period_samples)
            rate = (violations / (len(cluster_spikes) - 1)) * 100
            rates_dict[cid] = rate
        return rates_dict

    def build_cluster_dataframe(self):
        """Constructs the main pandas DataFrame using the optimized ISI calculation."""
        cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        df = pd.DataFrame({'cluster_id': cluster_ids, 'n_spikes': n_spikes})

        isi_rates_dict = self._calculate_all_isi_violations_vectorized()
        df['isi_violations_pct'] = df['cluster_id'].map(isi_rates_dict).fillna(0)

        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns:
            self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})

        df = pd.merge(df, info_subset, on='cluster_id', how='left')
        df['status'] = 'Original'
        
        self.cluster_df = df[['cluster_id', 'KSLabel', 'n_spikes', 'isi_violations_pct', 'status']]
        self.original_cluster_df = self.cluster_df.copy()
        self.next_new_cluster_id = (self.cluster_df['cluster_id'].max() + 1) if not self.cluster_df.empty else 0

    def get_cluster_spikes(self, cluster_id):
        """Returns spike times for a given cluster ID."""
        return self.spike_times[self.spike_clusters == cluster_id]

    def _find_nearest_channels(self, target_idx, n=16):
        """Finds the n nearest channels to a target channel index."""
        if self.channel_positions is None: return []
        distances = cdist([self.channel_positions[target_idx]], self.channel_positions)[0]
        return np.argsort(distances)[:n]

    def get_ei_features(self, cluster_id, n_sample=500, n_raw_snippets=30):
        """Retrieves EI and snippet data for a cluster, using the multi-level cache."""
        if cluster_id in self.ei_cache:
            return self.ei_cache[cluster_id]

        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) == 0: return None

        sample_indices = np.random.choice(len(spike_times_cluster), size=min(n_sample, len(spike_times_cluster)), replace=False)
        spike_sample = spike_times_cluster[sample_indices]

        snippets = cleaning_utils_cpu.extract_snippets(str(self.dat_path), spike_sample, n_channels=self.n_channels)
        if snippets.shape[2] == 0: return None

        # --- Robust baseline correction for both mean and raw traces ---
        pre_samples = 40  # Default, or derive from window if available
        # If window is available in params, use its negative part
        if hasattr(self, 'params') and 'window' in self.params:
            pre_samples = abs(self.params['window'][0])
        pre_samples = min(pre_samples, snippets.shape[1])  # Safeguard for short snippets
        snippets_bc = cleaning_utils_cpu.baseline_correct(snippets, pre_samples=pre_samples)

        mean_ei = cleaning_utils_cpu.compute_ei(snippets_bc, pre_samples=pre_samples)
        p2p = mean_ei.max(axis=1) - mean_ei.min(axis=1)
        dominant_channel = np.argmax(p2p)

        features = {
            'mean_ei': mean_ei,
            'raw_snippets': snippets_bc[:, :, :min(n_raw_snippets, snippets_bc.shape[2])],
            'dominant_channel': dominant_channel,
            'neighbor_channels': self._find_nearest_channels(dominant_channel)
        }
        self.ei_cache[cluster_id] = features # Add to in-memory cache
        return features

    def _calculate_isi_violations(self, cluster_id, refractory_period_ms=2.0):
        """Calculates ISI for a single cluster. Used for post-refinement updates."""
        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2:
            return 0.0
        
        isis = np.diff(np.sort(spike_times_cluster))
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        violations = np.sum(isis < refractory_period_samples)
        
        return (violations / (len(spike_times_cluster) - 1)) * 100

    def update_after_refinement(self, result_list, parent_id):
        """Updates the internal data state after a cluster has been refined."""
        original_indices = np.where(self.spike_clusters == parent_id)[0]
        parent_row_mask = self.cluster_df['cluster_id'] == parent_id

        if not result_list:
             self.cluster_df.loc[parent_row_mask, 'status'] = 'Refined (Empty)'
             return f"Refinement of {parent_id} resulted in no valid sub-clusters."

        if len(result_list) == 1 and len(result_list[0]['inds']) == len(original_indices):
            self.cluster_df.loc[parent_row_mask, 'status'] = 'Refined (Clean)'
            return f"Cluster {parent_id} was not split (already clean)."

        self.cluster_df.loc[parent_row_mask, 'status'] = 'Refined (Split)'
        for sub_cluster in result_list:
            new_id = self.next_new_cluster_id
            self.next_new_cluster_id += 1
            spike_indices_to_update = original_indices[sub_cluster['inds']]
            self.spike_clusters[spike_indices_to_update] = new_id
            new_row = {
                'cluster_id': new_id, 'KSLabel': 'good',
                'n_spikes': len(sub_cluster['inds']),
                'isi_violations_pct': self._calculate_isi_violations(new_id),
                'status': f'New (from {parent_id})'
            }
            self.cluster_df = pd.concat([self.cluster_df, pd.DataFrame([new_row])], ignore_index=True)
        
        self.is_dirty = True
        self.original_cluster_df = self.cluster_df.copy()
        return f"Cluster {parent_id} was split into {len(result_list)} new clusters."

# =============================================================================
# 2. VIEW: MainWindow & Supporting Widgets
# =============================================================================
class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView."""
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
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._dataframe.index[section])
        return None

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        colname = self._dataframe.columns[column]
        self._dataframe.sort_values(colname, ascending=(order == Qt.SortOrder.AscendingOrder), inplace=True)
        self._dataframe.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

class MplCanvas(FigureCanvas):
    """A Matplotlib canvas widget to embed in a Qt application."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1f1f1f')
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    """The main application window, acting as the 'View' and 'Controller'."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("axolotl")
        self.setGeometry(50, 50, 1800, 1000)
        self.data_manager = None
        self.refine_thread = None
        self.spatial_plot_dirty = True

        self._setup_style()
        self._setup_ui()
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage("Welcome to axolotl. Please load a Kilosort directory to begin.")

    def _setup_style(self):
        """Defines the dark theme for the application."""
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
            QMenuBar { background-color: #2D2D2D; color: white; }
            QMenu { background-color: #2D2D2D; color: white; }
            QMenu::item:selected { background-color: #4282DA; }
        """)

    def _setup_ui(self):
        """Builds the main user interface layout."""
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
        self.waveform_plot = pg.PlotWidget(title="Dominant Channel Waveforms")
        
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

        self.spatial_waveforms_tab = QWidget()
        spatial_waveforms_layout = QVBoxLayout(self.spatial_waveforms_tab)
        self.spatial_waveform_plot = pg.PlotWidget(title="Spatial Waveforms (Nearest 16 Channels)")
        spatial_waveforms_layout.addWidget(self.spatial_waveform_plot)
        self.analysis_tabs.addTab(self.spatial_waveforms_tab, "Spatial Waveforms")

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
        save_action = file_menu.addAction("&Save Refined Results...")
        self.save_action = save_action
        self.save_action.setEnabled(False)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")

        load_action.triggered.connect(self.load_directory)
        save_action.triggered.connect(self.on_save_results)
        exit_action.triggered.connect(self.close)
        self.filter_button.clicked.connect(self._apply_good_filter)
        self.reset_button.clicked.connect(self._reset_table_view)
        self.refine_button.clicked.connect(self.on_refine_cluster)
        
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)

    # --- Controller Logic ---
    def load_directory(self):
        """Handles the 'Load Directory' action with a synchronous, responsive flow."""
        ks_dir_name = QFileDialog.getExistingDirectory(self, "Select Kilosort Output Directory")
        if not ks_dir_name: return

        self.status_bar.showMessage("Loading Kilosort files...")
        QApplication.processEvents()

        self.data_manager = DataManager(ks_dir_name)
        
        if self.data_manager.load_disk_cache():
            self.status_bar.showMessage("Loaded EIs from on-disk cache.")
        else:
            self.status_bar.showMessage("No cache file found. EIs will be computed on demand.")
        QApplication.processEvents()

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
        
        self.status_bar.showMessage("Building cluster dataframe... This may take a moment.")
        QApplication.processEvents()

        self.data_manager.build_cluster_dataframe()
        self._setup_gui_with_data()
        self.central_widget.setEnabled(True)
        self.status_bar.showMessage(f"Successfully loaded {len(self.data_manager.cluster_df)} clusters.", 5000)

    def _setup_gui_with_data(self):
        """Initializes the UI components that depend on loaded data."""
        self.pandas_model = PandasModel(self.data_manager.cluster_df)
        self.table_view.setModel(self.pandas_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.selectionModel().selectionChanged.connect(self.on_cluster_selection_changed)

    def on_cluster_selection_changed(self):
        """
        Updates only the lightweight plots for maximum speed. Marks expensive plots as 'dirty'.
        """
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None: return

        # --- This part is now extremely fast ---
        self._update_fast_plots(cluster_id)
        
        # --- Mark expensive plots to be loaded on demand ---
        self.spatial_plot_dirty = True
        
        # --- Trigger lazy load if the relevant tab is already active ---
        self.on_tab_changed(self.analysis_tabs.currentIndex())

    def _update_fast_plots(self, cluster_id):
        """Helper function to update all the fast pyqtgraph plots at once."""
        self.status_bar.showMessage(f"Visualizing Cluster ID: {cluster_id}", 2000)
        ei_features = self.data_manager.get_ei_features(cluster_id)
        if ei_features is None:
            self.status_bar.showMessage(f"Could not generate EI for cluster {cluster_id}.", 3000)
            return
        
        self._update_waveform_plot(cluster_id, ei_features)
        self._update_isi_plot(cluster_id)
        self._update_fr_plot(cluster_id)
        self._update_spatial_waveform_plot(cluster_id, ei_features)

    def on_tab_changed(self, index):
        """Handles lazy loading of expensive plots when a tab is selected."""
        if self.analysis_tabs.widget(index) == self.summary_tab and self.spatial_plot_dirty:
            cluster_id = self._get_selected_cluster_id()
            if cluster_id is None: return
            
            self.status_bar.showMessage(f"Generating spatial analysis for cluster {cluster_id}...", 2000)
            QApplication.processEvents()

            ei_features = self.data_manager.get_ei_features(cluster_id)
            if ei_features:
                self._update_summary_plot(cluster_id, ei_features)
                self.spatial_plot_dirty = False
            
            self.status_bar.showMessage("Ready.", 2000)

    def on_refine_cluster(self):
        """Starts the cluster refinement process in a background thread."""
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            QMessageBox.warning(self, "No Cluster Selected", "Please select a cluster from the table to refine.")
            return

        self.refine_button.setEnabled(False)
        self.status_bar.showMessage(f"Starting refinement for cluster {cluster_id}...")

        self.refine_thread = QThread()
        self.worker = RefinementWorker(self.data_manager, cluster_id)
        self.worker.moveToThread(self.refine_thread)

        self.refine_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_refinement_results)
        self.worker.error.connect(self.handle_refinement_error)
        self.worker.progress.connect(lambda msg: self.status_bar.showMessage(msg, 2000))
        
        self.worker.finished.connect(self.refine_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.refine_thread.finished.connect(self.refine_thread.deleteLater)
        
        self.refine_thread.start()

    def handle_refinement_results(self, result_list, parent_id):
        """Updates the data model and UI after refinement is complete."""
        message = self.data_manager.update_after_refinement(result_list, parent_id)
        self.status_bar.showMessage(message, 5000)
        self._reset_table_view()
        self.refine_button.setEnabled(True)
        if self.data_manager.is_dirty:
            self.save_action.setEnabled(True)
            self.setWindowTitle("axolotl* (Unsaved Changes)")

    def handle_refinement_error(self, error_message):
        QMessageBox.critical(self, "Refinement Error", error_message)
        self.refine_button.setEnabled(True)
        self.status_bar.showMessage("Refinement failed.", 10000)

    def on_save_results(self):
        """Saves the refined results by overwriting the original Kilosort files."""
        ks_dir = self.data_manager.kilosort_dir
        clusters_path = ks_dir / 'spike_clusters.npy'
        new_info_path = ks_dir / 'cluster_info.tsv' 
        original_info_path = self.data_manager.info_path

        clusters_bak_path = clusters_path.with_suffix('.npy.bak')
        if original_info_path:
            original_info_bak_path = original_info_path.with_suffix(original_info_path.suffix + '.bak')
        else:
            original_info_bak_path = None

        msg = (f"This will overwrite files in:\n{ks_dir}\n\n"
               f"The following files will be modified:\n"
               f"- {clusters_path.name}\n"
               f"- {new_info_path.name}\n\n"
               f"A backup of the original files will be created with a .bak extension if one doesn't already exist.\n\n"
               f"Are you sure you want to proceed?")

        reply = QMessageBox.question(self, "Confirm Overwrite", msg,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if clusters_path.exists() and not clusters_bak_path.exists():
                    os.rename(clusters_path, clusters_bak_path)
                    self.status_bar.showMessage(f"Backed up {clusters_path.name}", 2000)

                if original_info_path and original_info_path.exists() and not original_info_bak_path.exists():
                    os.rename(original_info_path, original_info_bak_path)
                    self.status_bar.showMessage(f"Backed up {original_info_path.name}", 2000)

                np.save(clusters_path, self.data_manager.spike_clusters)
                self.data_manager.cluster_df.to_csv(new_info_path, sep='\t', index=False)
                
                self.status_bar.showMessage("Results saved successfully!", 5000)
                self.data_manager.is_dirty = False
                self.save_action.setEnabled(False)
                self.setWindowTitle("axolotl")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save files: {e}")

    # --- Plotting Methods ---
    def _update_summary_plot(self, cluster_id, ei_features):
        """Updates the rich Matplotlib summary plot."""
        ei = ei_features['mean_ei']
        positions = self.data_manager.channel_positions
        n_elec, _ = ei.shape
        if positions.shape[0] != n_elec:
            temp_positions = np.zeros((n_elec, 2))
            min_len = min(n_elec, positions.shape[0])
            temp_positions[:min_len, :] = positions[:min_len, :]
            positions = temp_positions
        
        self.summary_canvas.fig.clear()
        cleaning_utils_cpu.plot_rich_ei(
            self.summary_canvas.fig,
            ei,
            positions,
            self.data_manager.sampling_rate
        )
        self.summary_canvas.fig.suptitle(f"Cluster {cluster_id} Spatial Analysis", color='white', fontsize=16)
        self.summary_canvas.draw()

    def _update_waveform_plot(self, cluster_id, ei_features):
        """Updates the pyqtgraph waveform plot with cached data."""
        self.waveform_plot.clear()
        self.waveform_plot.setTitle(f"Cluster {cluster_id} | Waveforms (n={ei_features['raw_snippets'].shape[2]})")
        
        mean_ei = ei_features['mean_ei']
        snippets = ei_features['raw_snippets']
        dom_chan = ei_features['dominant_channel']
        time_axis = (np.arange(mean_ei.shape[1]) - 40) / self.data_manager.sampling_rate * 1000

        for i in range(snippets.shape[2]):
            self.waveform_plot.plot(time_axis, snippets[dom_chan, :, i], pen=pg.mkPen(color=(200, 200, 200, 30)))
        
        self.waveform_plot.plot(time_axis, mean_ei[dom_chan], pen=pg.mkPen('#00A3E0', width=2.5))
        self.waveform_plot.setLabel('bottom', 'Time (ms)')
        self.waveform_plot.setLabel('left', 'Amplitude (uV)')

    def _update_isi_plot(self, cluster_id):
        """Updates the ISI histogram plot."""
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
        """Updates the smoothed firing rate plot."""
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

    def _update_spatial_waveform_plot(self, cluster_id, ei_features):
        """Updates the restored spatial waveform plot."""
        self.spatial_waveform_plot.clear()
        self.spatial_waveform_plot.setTitle(f"Cluster {cluster_id} | Spatial Waveforms")
        
        mean_ei = ei_features['mean_ei']
        snippets = ei_features['raw_snippets']
        dom_chan = ei_features['dominant_channel']
        neighbor_channels = ei_features['neighbor_channels']
        time_axis = (np.arange(mean_ei.shape[1]) - 40) / self.data_manager.sampling_rate * 1000

        for ch_idx in neighbor_channels:
            x_pos, y_pos = self.data_manager.channel_positions[ch_idx]
            
            for i in range(snippets.shape[2]):
                self.spatial_waveform_plot.plot(
                    time_axis * 15 + x_pos,
                    snippets[ch_idx, :, i] * 0.05 + y_pos,
                    pen=pg.mkPen(color=(200, 200, 200, 20))
                )
            
            pen_color = '#00A3E0' if ch_idx == dom_chan else (180, 180, 180)
            self.spatial_waveform_plot.plot(
                time_axis * 15 + x_pos,
                mean_ei[ch_idx] * 0.05 + y_pos,
                pen=pg.mkPen(pen_color, width=1.5)
            )
        self.spatial_waveform_plot.setAspectLocked(True)


    # --- Helper & UI State Methods ---
    def _get_selected_cluster_id(self):
        """Safely retrieves the selected cluster ID from the table view."""
        if not self.table_view.selectionModel().hasSelection():
            return None
        row = self.table_view.selectionModel().selectedRows()[0].row()
        return self.pandas_model._dataframe.iloc[row]['cluster_id']

    def _apply_good_filter(self):
        """Filters the table to show only clusters marked 'good'."""
        filtered_df = self.data_manager.original_cluster_df[
            self.data_manager.original_cluster_df['KSLabel'] == 'good'
        ].copy()
        self.pandas_model.set_dataframe(filtered_df)

    def _reset_table_view(self):
        """Resets the table to show all clusters."""
        self.pandas_model.set_dataframe(self.data_manager.cluster_df)

    def closeEvent(self, event):
        """Handles closing the application, checking for unsaved changes and running threads."""
        if self.refine_thread and self.refine_thread.isRunning():
            QMessageBox.warning(self, "Task in Progress", 
                                "A refinement task is still running. Please wait for it to complete before closing.")
            event.ignore()
            return

        if self.data_manager and self.data_manager.is_dirty:
            reply = QMessageBox.question(self, "Unsaved Changes",
                                         "You have unsaved refinement results. Quit without saving?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Save the on-disk cache before exiting
        if self.data_manager:
            self.data_manager.save_disk_cache()

        event.accept()

# =============================================================================
# 3. WORKER THREAD: For CPU-Intensive Tasks
# =============================================================================
class RefinementWorker(QObject):
    """
    A worker object that runs the CPU-intensive cluster refinement task
    in a separate thread to prevent the GUI from freezing.
    """
    finished = Signal(object, int)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, data_manager, cluster_id):
        super().__init__()
        self.data_manager = data_manager
        self.cluster_id = cluster_id

    def run(self):
        """The main work of the thread."""
        try:
            print(f"\n{'='*60}")
            print(f"STARTING REFINEMENT FOR CLUSTER {self.cluster_id}")
            print(f"{'='*60}")
            
            self.progress.emit(f"Refining cluster {self.cluster_id}...")
            spikes_to_clean = self.data_manager.get_cluster_spikes(self.cluster_id)
            print(f"Loaded {len(spikes_to_clean)} spikes for refinement")
            
            cleaning_params = {
                'n_channels': self.data_manager.n_channels,
                'sampling_rate': self.data_manager.sampling_rate
            }
            print(f"Using parameters: {cleaning_params}")

            result = cleaning_utils_cpu.refine_cluster_v2(
                spike_times=spikes_to_clean,
                dat_path=str(self.data_manager.dat_path),
                channel_positions=self.data_manager.channel_positions,
                params=cleaning_params
            )
            
            print(f"\n{'='*60}")
            print(f"REFINEMENT COMPLETE FOR CLUSTER {self.cluster_id}")
            print(f"Result: {len(result)} refined clusters")
            print(f"{'='*60}\n")
            
            self.finished.emit(result, self.cluster_id)
        except Exception as e:
            import traceback
            error_msg = f"Error in refinement thread for cluster {self.cluster_id}:\n{traceback.format_exc()}"
            print(f"ERROR: {error_msg}")
            self.error.emit(error_msg)

# =============================================================================
# 4. APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
