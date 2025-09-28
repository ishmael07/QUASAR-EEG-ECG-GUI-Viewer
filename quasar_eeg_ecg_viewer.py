"""
Ishmael's QUASAR EEG/ECG Interactive GUI Viewer — Enhanced single-GUI version

Features added / changed:
- Single GUI contains file browser, channel selector, plot options, data info, and export
- Robust CSV loading that ignores `#` metadata lines and uses first non-comment row as header
- Channel multi-selection control (Listbox) populated after loading the file
- Unit conversion toggle (show EEG in μV and ECG in mV; option to show ECG in μV if needed)
- Normalize (z-score) option for EEG channels
- Option to subtract CM (reference) from EEG channels before plotting
- Plot layout:
    * EEG channels grouped into an EEG subplot (stacked traces)
    * ECG channels placed in a separate subplot (scaled to mV)
    * CM plotted separately (if selected)
    * Shared x-axis with Plotly range slider + pan/zoom + hover unified mode
- Export: HTML always; PNG export if `kaleido` installed (graceful fallback)
- Background file loading to avoid GUI freeze and improved status messages
- Better channel detection and safe ignoring of unwanted columns
- Small UI tweaks: auto-enable/disable Generate/Export buttons, info panel with summary

Dependencies:
- pandas
- plotly
- tkinter (builtin)
- kaleido (for PNG export)

Run:
    python quasar_eeg_ecg_viewer.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import tempfile
import os
import threading
import math

# Try to detect kaleido for image export
try:
    import plotly.io as pio
    _KAL = "kaleido" in pio.kaleido.scope.map if hasattr(pio, "kaleido") else True
except Exception:
    # we'll try to write a PNG later and catch the error
    _KAL = False


class QuasarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QUASAR EEG/ECG Data Viewer — Enhanced")
        self.root.geometry("980x660")
        self.root.minsize(800, 520)

        # Data
        self.csv_path = None
        self.data = None
        self.metadata = []

        # Channel definitions (assignment-provided lists)
        self.EEG_CHANNELS = ['Fz', 'Cz', 'P3', 'C3', 'F3', 'F4', 'C4', 'P4',
                             'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2',
                             'F7', 'F8', 'A1', 'A2', 'Pz']
        self.ECG_CHANNELS = ['X1:LEOG', 'X2:REOG']  # X1=X2 mapping
        self.REFERENCE_CHANNELS = ['CM']
        self.IGNORE_PATTERNS = ['X3:', 'Trigger', 'Time_Offset', 'ADC_Status', 'ADC_Sequence', 'Event', 'Comments']

        # GUI state variables
        self.file_var = tk.StringVar(value="No file selected")
        self.status_var = tk.StringVar(value="Ready — select a CSV")
        self.unit_choice = tk.StringVar(value="native")  # 'native' (EEG μV, ECG mV) or 'all_uv' (everything μV)
        self.normalize_eeg = tk.BooleanVar(value=False)
        self.subtract_cm = tk.BooleanVar(value=False)
        self.show_cm = tk.BooleanVar(value=True)

        # After plot is created, store it
        self.current_fig = None

        self.setup_ui()

    def setup_ui(self):
        root = self.root
        # Top frame: title + file controls
        top_frame = ttk.Frame(root, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(1, weight=1)

        title = ttk.Label(top_frame, text="QUASAR EEG/ECG Data Viewer", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(top_frame, text="CSV:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.file_display = ttk.Label(top_frame, textvariable=self.file_var, relief="sunken", padding=5)
        self.file_display.grid(row=1, column=1, sticky="ew", padx=(6, 6), pady=(8, 0))

        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_file)
        browse_btn.grid(row=1, column=2, sticky="e", pady=(8, 0))

        # Main panes
        main_pane = ttk.Frame(root, padding=10)
        main_pane.grid(row=1, column=0, sticky="nsew")
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        # Left: controls
        controls_frame = ttk.LabelFrame(main_pane, text="Controls", padding=10)
        controls_frame.grid(row=0, column=0, sticky="nsw")
        controls_frame.columnconfigure(0, weight=1)

        # Channel listbox
        ttk.Label(controls_frame, text="Available Channels (select multiple):", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.channel_listbox = tk.Listbox(controls_frame, selectmode=tk.MULTIPLE, width=28, height=18, exportselection=False)
        self.channel_listbox.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        controls_frame.rowconfigure(1, weight=1)

        ch_btn_frame = ttk.Frame(controls_frame)
        ch_btn_frame.grid(row=2, column=0, sticky="ew", pady=(6, 6))
        ttk.Button(ch_btn_frame, text="Select All", command=self.select_all_channels).grid(row=0, column=0, sticky="ew")
        ttk.Button(ch_btn_frame, text="Clear", command=self.clear_channel_selection).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ch_btn_frame.columnconfigure(0, weight=1)
        ch_btn_frame.columnconfigure(1, weight=1)

        # Plot options
        ttk.Separator(controls_frame).grid(row=3, column=0, sticky="ew", pady=8)
        ttk.Label(controls_frame, text="Plot Options:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(0, 6))

        # Units radio
        ttk.Label(controls_frame, text="Units mode:").grid(row=5, column=0, sticky="w")
        units_frame = ttk.Frame(controls_frame)
        units_frame.grid(row=6, column=0, sticky="w", pady=(2, 8))
        ttk.Radiobutton(units_frame, text="Default (EEG μV, ECG mV)", variable=self.unit_choice, value="native", command=self.on_option_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(units_frame, text="All μV (show ECG in μV)", variable=self.unit_choice, value="all_uv", command=self.on_option_changed).grid(row=1, column=0, sticky="w")

        ttk.Checkbutton(controls_frame, text="Normalize EEG (z-score)", variable=self.normalize_eeg, command=self.on_option_changed).grid(row=7, column=0, sticky="w", pady=(0, 4))
        ttk.Checkbutton(controls_frame, text="Subtract CM from EEG before plotting", variable=self.subtract_cm, command=self.on_option_changed).grid(row=8, column=0, sticky="w", pady=(0, 4))
        ttk.Checkbutton(controls_frame, text="Show CM trace", variable=self.show_cm, command=self.on_option_changed).grid(row=9, column=0, sticky="w")

        # Time window input
        ttk.Separator(controls_frame).grid(row=10, column=0, sticky="ew", pady=8)
        ttk.Label(controls_frame, text="Time window (seconds):", font=("Arial", 10, "bold")).grid(row=11, column=0, sticky="w")
        time_frame = ttk.Frame(controls_frame)
        time_frame.grid(row=12, column=0, sticky="w", pady=(6, 6))
        ttk.Label(time_frame, text="Start:").grid(row=0, column=0)
        self.time_start_entry = ttk.Entry(time_frame, width=8)
        self.time_start_entry.grid(row=0, column=1, padx=(4, 8))
        ttk.Label(time_frame, text="End:").grid(row=0, column=2)
        self.time_end_entry = ttk.Entry(time_frame, width=8)
        self.time_end_entry.grid(row=0, column=3, padx=(4, 4))
        ttk.Button(time_frame, text="Reset", command=self.reset_time_inputs).grid(row=0, column=4, padx=(6, 0))

        # Buttons: Generate + Export
        btns_frame = ttk.Frame(controls_frame)
        btns_frame.grid(row=13, column=0, sticky="ew", pady=(12, 0))
        self.generate_btn = ttk.Button(btns_frame, text="Generate Plot", command=self.generate_plot, state="disabled")
        self.generate_btn.grid(row=0, column=0, sticky="ew")
        self.export_btn = ttk.Button(btns_frame, text="Export...", command=self.export_plot, state="disabled")
        self.export_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        btns_frame.columnconfigure(0, weight=1)

        # Right: info + small preview
        right_frame = ttk.Frame(main_pane)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        main_pane.columnconfigure(1, weight=1)
        main_pane.rowconfigure(0, weight=1)

        info_frame = ttk.LabelFrame(right_frame, text="Data Information", padding=8)
        info_frame.grid(row=0, column=0, sticky="nsew")
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        self.info_text = tk.Text(info_frame, height=20, wrap="word", font=("Consolas", 10))
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set, state="disabled")

        # Status bar
        status = ttk.Label(root, textvariable=self.status_var, relief="sunken", padding=6)
        status.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 10))

    # -------------------------
    # File loading & parsing
    # -------------------------
    def browse_file(self):
        path = filedialog.askopenfilename(title="Select EEG/ECG Data File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.csv_path = path
        self.file_var.set(os.path.basename(path))
        self.start_data_load()

    def start_data_load(self):
        if not self.csv_path:
            return
        self.status_var.set("Loading data...")
        self.generate_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        t = threading.Thread(target=self._load_data_thread, daemon=True)
        t.start()

    def _load_data_thread(self):
        try:
            with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # collect metadata lines starting with '#'
            metadata = []
            data_start = 0
            for i, ln in enumerate(lines):
                if ln.strip().startswith("#"):
                    metadata.append(ln.strip())
                elif ln.strip() == "":
                    # skip blank lines
                    continue
                else:
                    data_start = i
                    break

            # read into pandas, using first non-comment row as header
            # Use skiprows to pass the header index - pandas will use that row as header automatically
            df = pd.read_csv(self.csv_path, skiprows=data_start, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]

            # basic validation
            if "Time" not in df.columns:
                raise ValueError("Missing required 'Time' column in CSV header.")

            # keep numeric columns only for signals where appropriate; but keep other columns in info
            # Convert Time to numeric
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            df = df.dropna(subset=['Time'])
            # convert all other signal columns to numeric if possible
            for col in df.columns:
                if col != "Time":
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Save
            self.data = df.reset_index(drop=True)
            self.metadata = metadata

            # Update GUI in main thread
            self.root.after(0, self._on_data_loaded)
        except Exception as e:
            self.root.after(0, lambda: self._on_data_error(str(e)))

    def _on_data_loaded(self):
        self.status_var.set(f"Data loaded — {len(self.data):,} rows, {len(self.data.columns)} columns")
        self.generate_btn.configure(state="normal")
        self.populate_channel_list()
        self.update_info_panel()
        # clear any previous figure
        self.current_fig = None
        self.export_btn.configure(state="disabled")

    def _on_data_error(self, msg):
        self.status_var.set("Error loading data")
        messagebox.showerror("Data Load Error", f"Failed to load CSV:\n{msg}")

    # -------------------------
    # Channel selection helpers
    # -------------------------
    def populate_channel_list(self):
        """Populate left channel listbox based on detected columns (excluding ignore patterns)"""
        self.channel_listbox.delete(0, tk.END)
        if self.data is None:
            return

        cols = [c for c in self.data.columns if c != "Time"]
        # filter ignore patterns
        def ignore_col(c):
            for pat in self.IGNORE_PATTERNS:
                if pat and pat in c:
                    return True
            return False

        display_cols = [c for c in cols if not ignore_col(c)]
        # Keep a stable sort: put EEG first (in defined order), then ECG, then CM, then others
        ordered = []
        # EEG
        for ch in self.EEG_CHANNELS:
            if ch in display_cols and ch not in ordered:
                ordered.append(ch)
        # ECG
        for ch in self.ECG_CHANNELS:
            if ch in display_cols and ch not in ordered:
                ordered.append(ch)
        # CM
        for ch in self.REFERENCE_CHANNELS:
            if ch in display_cols and ch not in ordered:
                ordered.append(ch)
        # remaining
        for ch in display_cols:
            if ch not in ordered:
                ordered.append(ch)

        for ch in ordered:
            self.channel_listbox.insert(tk.END, ch)

        # By default select all available channels
        self.select_all_channels()

    def select_all_channels(self):
        self.channel_listbox.select_set(0, tk.END)

    def clear_channel_selection(self):
        self.channel_listbox.select_clear(0, tk.END)

    # -------------------------
    # UI helpers
    # -------------------------
    def reset_time_inputs(self):
        self.time_start_entry.delete(0, tk.END)
        self.time_end_entry.delete(0, tk.END)

    def on_option_changed(self):
        # small immediate feedback; don't auto-generate plot (user must press Generate)
        self.status_var.set("Options changed — press 'Generate Plot' to update")

    def update_info_panel(self):
        """Write a helpful summary into info_text"""
        if self.data is None:
            return

        eeg_av, ecg_av, ref_av = self._detect_channels_present()

        tmin = float(self.data['Time'].min())
        tmax = float(self.data['Time'].max())
        duration = tmax - tmin
        sr = 1.0 / float(self.data['Time'].diff().median()) if len(self.data) > 1 else float("nan")

        lines = []
        lines.append("=== DATA SUMMARY ===\n")
        lines.append(f"File: {os.path.basename(self.csv_path)}")
        lines.append(f"Time range: {tmin:.3f} - {tmax:.3f} s")
        lines.append(f"Duration: {duration:.3f} s")
        lines.append(f"Approx. sampling rate: {sr:.1f} Hz")
        lines.append(f"Data rows: {len(self.data):,}")
        lines.append(f"Metadata lines (leading '#'): {len(self.metadata)}\n")

        lines.append("=== AVAILABLE CHANNELS ===")
        lines.append(f"EEG ({len(eeg_av)}): " + (", ".join(eeg_av) if eeg_av else "None"))
        lines.append(f"ECG ({len(ecg_av)}): " + (", ".join(ecg_av) if ecg_av else "None"))
        lines.append(f"Reference (CM): " + (", ".join(ref_av) if ref_av else "None"))

        # Ranges
        try:
            if eeg_av:
                eegvals = self.data[eeg_av].apply(pd.to_numeric, errors='coerce')
                lines.append(f"\nEEG amplitude range (μV): {eegvals.min().min():.1f} to {eegvals.max().max():.1f}")
            if ecg_av:
                ecgvals = self.data[ecg_av].apply(pd.to_numeric, errors='coerce')
                # show in μV and mV
                lines.append(f"ECG amplitude range: {ecgvals.min().min():.1f} to {ecgvals.max().max():.1f} μV (~{ecgvals.min().min()/1000:.3f} to {ecgvals.max().max()/1000:.3f} mV)")
            if ref_av:
                refvals = self.data[ref_av].apply(pd.to_numeric, errors='coerce')
                lines.append(f"CM amplitude range: {refvals.min().min():.1f} to {refvals.max().max():.1f}")
        except Exception:
            pass

        if self.metadata:
            lines.append("\n=== METADATA SAMPLE ===")
            for ln in self.metadata[:6]:
                lines.append(ln)
            if len(self.metadata) > 6:
                lines.append(f"... and {len(self.metadata) - 6} more lines")

        text = "\n".join(lines)
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state="disabled")

    def _detect_channels_present(self):
        if self.data is None:
            return [], [], []
        cols = set(self.data.columns)
        eeg = [c for c in self.EEG_CHANNELS if c in cols]
        ecg = [c for c in self.ECG_CHANNELS if c in cols]
        ref = [c for c in self.REFERENCE_CHANNELS if c in cols]
        return eeg, ecg, ref

    # -------------------------
    # Plot generation
    # -------------------------
    def get_selected_channels(self):
        sel = self.channel_listbox.curselection()
        return [self.channel_listbox.get(i) for i in sel]

    def _parse_time_window(self):
        if self.data is None:
            return None
        tmin = float(self.data['Time'].min())
        tmax = float(self.data['Time'].max())
        s = self.time_start_entry.get().strip()
        e = self.time_end_entry.get().strip()
        try:
            start = float(s) if s else tmin
            end = float(e) if e else tmax
            start = max(start, tmin)
            end = min(end, tmax)
            if start >= end:
                return None
            return (start, end)
        except ValueError:
            return None

    def generate_plot(self):
        """Main entry to build the plot and open it in browser"""
        if self.data is None:
            messagebox.showwarning("No data", "Load a CSV file first.")
            return

        selected = self.get_selected_channels()
        if not selected:
            messagebox.showwarning("No channels", "Select at least one channel to plot.")
            return

        self.status_var.set("Generating plot...")
        self.root.update_idletasks()

        try:
            fig = self._create_plot(selected)
            if fig is None:
                raise RuntimeError("Failed to create figure (no channels after filtering).")

            # write temporary html and open
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
            tmp.close()
            fig.write_html(tmp.name, include_plotlyjs="cdn")
            webbrowser.open("file://" + tmp.name)

            self.current_fig = fig
            self.export_btn.configure(state="normal")
            self.status_var.set("Plot created and opened in browser")
        except Exception as e:
            messagebox.showerror("Plot error", f"Failed to generate plot:\n{str(e)}")
            self.status_var.set("Plot generation failed")

    def _create_plot(self, selected_channels):
        """Assemble a Plotly subplot figure according to options and selection"""
        df = self.data.copy()
        # filter by time window first
        tr = self._parse_time_window()
        if tr:
            df = df[(df['Time'] >= tr[0]) & (df['Time'] <= tr[1])]
        time = df['Time']

        # categorize selected channels
        eeg_sel = [c for c in selected_channels if c in self.EEG_CHANNELS and c in df.columns]
        ecg_sel = [c for c in selected_channels if c in self.ECG_CHANNELS and c in df.columns]
        cm_sel = [c for c in selected_channels if c in self.REFERENCE_CHANNELS and c in df.columns]

        # also treat any other selected columns as "other" and plot in separate subplot
        others = [c for c in selected_channels if c not in (eeg_sel + ecg_sel + cm_sel) and c in df.columns]

        # Decide how many subplots:
        # We'll include EEG subplot if eeg_sel non-empty
        # ECG subplot if ecg_sel non-empty
        # CM subplot if cm_sel non-empty and user chose to show it
        # Others subplot if others exist
        subplots = []
        if eeg_sel:
            subplots.append("EEG")
        if ecg_sel:
            subplots.append("ECG")
        if cm_sel and self.show_cm.get():
            subplots.append("CM")
        if others:
            subplots.append("OTHER")

        if not subplots:
            return None

        fig = make_subplots(rows=len(subplots), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=[("EEG Channels (μV)" if s == "EEG" else
                                             "ECG Channels (mV)" if s == "ECG" else
                                             "Reference (CM)" if s == "CM" else
                                             "Other Channels") for s in subplots])

        row = 1

        # CM baseline for subtraction
        cm_series = None
        if cm_sel:
            # we take the first CM available
            cm_col = cm_sel[0]
            cm_series = pd.to_numeric(df[cm_col], errors='coerce')
            # if subtract option unchecked, we'll still plot CM in CM subplot if shown
        # --- EEG traces ---
        if "EEG" in subplots:
            for ch in eeg_sel:
                y = pd.to_numeric(df[ch], errors='coerce')
                # subtract CM if requested and CM present
                if self.subtract_cm.get() and (cm_series is not None):
                    y = y - cm_series
                # normalize if requested
                unit_label = "μV"
                if self.normalize_eeg.get():
                    # z-score
                    mean = y.mean()
                    std = y.std()
                    if std == 0 or math.isnan(std):
                        y_plot = y - mean  # fallback
                        unit_label = "(demeaned)"
                    else:
                        y_plot = (y - mean) / std
                        unit_label = "normalized"
                else:
                    y_plot = y

                fig.add_trace(
                    go.Scatter(x=time, y=y_plot, name=f"{ch} ({unit_label})",
                               hovertemplate=f"<b>{ch}</b><br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.2f}} {unit_label}<extra></extra>",
                               line=dict(width=1)),
                    row=row, col=1
                )
            # y-axis title
            fig.update_yaxes(title_text="μV (or normalized)", row=row, col=1)
            row += 1

        # --- ECG traces ---
        if "ECG" in subplots:
            for ch in ecg_sel:
                y = pd.to_numeric(df[ch], errors='coerce')
                # unit conversion choice:
                if self.unit_choice.get() == "native":
                    # show ECG in mV
                    y_plot = y / 1000.0
                    unit_label = "mV"
                else:
                    # show ECG in μV (no conversion)
                    y_plot = y
                    unit_label = "μV"
                fig.add_trace(
                    go.Scatter(x=time, y=y_plot, name=f"{ch} ({unit_label})",
                               hovertemplate=f"<b>{ch}</b><br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.3f}} {unit_label}<extra></extra>",
                               line=dict(width=1.5)),
                    row=row, col=1
                )
            ytitle = "mV" if self.unit_choice.get() == "native" else "μV"
            fig.update_yaxes(title_text=f"Amplitude ({ytitle})", row=row, col=1)
            row += 1

        # --- CM trace ---
        if "CM" in subplots:
            cm_col = cm_sel[0]
            cm_y = pd.to_numeric(df[cm_col], errors='coerce')
            fig.add_trace(
                go.Scatter(x=time, y=cm_y, name=f"{cm_col} (CM)", hovertemplate=f"<b>{cm_col}</b><br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.2f}}<extra></extra>",
                           line=dict(width=1, dash="dash")),
                row=row, col=1
            )
            fig.update_yaxes(title_text="CM (raw)", row=row, col=1)
            row += 1

        # --- OTHER ---
        if "OTHER" in subplots:
            for ch in others:
                y = pd.to_numeric(df[ch], errors='coerce')
                fig.add_trace(
                    go.Scatter(x=time, y=y, name=f"{ch}", hovertemplate=f"<b>{ch}</b><br>Time: %{{x:.3f}} s<br>Amplitude: %{{y:.2f}}<extra></extra>",
                               line=dict(width=1)),
                    row=row, col=1
                )
            fig.update_yaxes(title_text="Value", row=row, col=1)
            row += 1

        # Layout polish
        fig.update_layout(
            title={
                'text': f"QUASAR EEG/ECG — {os.path.basename(self.csv_path)}",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=220 * len(subplots) + 120,
            margin=dict(l=60, r=140, t=90, b=80),
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )

        # x-axis: range slider on bottom
        fig.update_xaxes(title_text="Time (s)", rangeslider=dict(visible=True), row=len(subplots), col=1)

        return fig

    # -------------------------
    # Export
    # -------------------------
    def export_plot(self):
        if self.current_fig is None:
            messagebox.showwarning("No plot", "Generate a plot first.")
            return

        # Ask for file save base name and type
        path = filedialog.asksaveasfilename(title="Export Plot", defaultextension=".html",
                                            filetypes=[("HTML file", "*.html"), ("PNG image", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            if path.lower().endswith(".html"):
                self.current_fig.write_html(path, include_plotlyjs="cdn")
                messagebox.showinfo("Exported", f"Plot exported to:\n{path}")
            elif path.lower().endswith(".png"):
                try:
                    # attempt png via kaleido
                    self.current_fig.write_image(path)
                    messagebox.showinfo("Exported", f"PNG exported to:\n{path}")
                except Exception as e:
                    # fallback: export HTML to temp and open for manual saving
                    messagebox.showwarning("PNG export failed", f"PNG export failed (kaleido may not be installed):\n{e}\n\nExporting HTML instead.")
                    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
                    tmp.close()
                    self.current_fig.write_html(tmp.name, include_plotlyjs="cdn")
                    webbrowser.open("file://" + tmp.name)
            else:
                # unknown extension — default to HTML
                self.current_fig.write_html(path + ".html", include_plotlyjs="cdn")
                messagebox.showinfo("Exported", f"Plot exported to:\n{path}.html")
        except Exception as e:
            messagebox.showerror("Export error", f"Export failed:\n{e}")

# -------------------------
# Entry
# -------------------------
def main():
    root = tk.Tk()
    app = QuasarGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
