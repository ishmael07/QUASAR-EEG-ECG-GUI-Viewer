# Ishmael's QUASAR EEG/ECG Data Viewer

**An interactive GUI application for visualizing EEG and ECG signals from CSV files.**

This tool loads raw EEG/ECG data and displays multiple channels in organized, interactive plots with comprehensive filtering and export options.

---

## Key Features

- **📂 Smart File Loading** – Automatically handles CSV files with metadata lines (ignores `#` comments)
- **📊 Multi-Channel Display** – View EEG, ECG, and reference channels in separate organized subplots
- **🎛️ Advanced Controls**
  - Select specific channels to plot using multi-selection interface
  - Choose unit display: native (EEG in μV, ECG in mV) or all in μV
  - Normalize EEG channels using z-score standardization
  - Subtract reference (CM) signal from EEG channels before plotting
  - Set custom time windows for focused analysis
- **🔍 Interactive Navigation** – Pan, zoom, and scroll through signals with Plotly's range slider
- **💾 Flexible Export** – Save plots as HTML (always) or PNG (if kaleido installed)
- **ℹ️ Data Summary** – Real-time info panel showing sampling rate, duration, channel ranges, and metadata

---

## Technical Details

### Channel Classification
- **EEG Channels**: Fz, Cz, P3, C3, F3, F4, C4, P4, Fp1, Fp2, T3, T4, T5, T6, O1, O2, F7, F8, A1, A2, Pz
- **ECG Channels**: X1:LEOG, X2:REOG
- **Reference**: CM (Common Mode)
- **Auto-Ignored**: Trigger, Time_Offset, ADC_Status, ADC_Sequence, Event, Comments, and X3: patterns

### Plot Organization
- EEG channels grouped in top subplot (stacked traces in μV)
- ECG channels in separate subplot (converted to mV for readability)
- CM reference plotted separately if selected
- Shared x-axis with unified hover and range controls

### Data Processing
- Background file loading prevents GUI freezing
- Robust CSV parsing handles various formats
- Automatic numeric conversion with error handling
- Time window filtering for focused analysis

---

## AI Assistance

During development of **Ishmael's EEG/ECG Viewer**, AI tools were used to streamline design and implementation:

- **GUI Layout Suggestions** – AI helped structure the Tkinter interface for a clear, user-friendly layout with organized frames, buttons, and control panels.
- **Plotly Configuration** – AI provided guidance on setting up interactive subplots, handling multiple EEG/ECG channels, adding range sliders, and customizing hover templates.
- **Code Optimization** – Recommendations from AI assisted in threading file I/O, improving responsiveness, and managing channel scaling for EEG (μV) vs ECG (mV).

This assistance ensured a more maintainable, usable, and professional application while preserving full control over the code logic and design decisions.

---

## Installation & Usage

### Requirements
```bash
pip install pandas plotly tkinter
```

### Quick Start 
``` bash 
git clone https://github.com/<your-github-username>/QUASAR-EEG-ECG-GUI-Viewer.git
cd quasar-eeg-ecg-gui-viewer
python quasar_viewer_enhanced.py
``` 

## Usage Guide

1. **Load Data**  
   Click "Browse..." and select your CSV file.

2. **Select Channels**  
   Use the channel list to choose which signals to display.

3. **Configure Processing**  
   - Set unit display preferences  
   - Enable normalization or reference subtraction as needed  
   - Define time window for focused analysis (optional)

4. **Generate Visualization**  
   Click "Generate Plot" to create the interactive plot.

5. **Export Results**  
   Save your plot as HTML or PNG using the "Export..." button.

---

## Architecture Notes

Built with Python's **Tkinter** for the GUI framework and **Plotly** for interactive visualization. The application handles the different amplitude scales of EEG (microvolts) and ECG (millivolts) by organizing them into logical subplot groups.

### Key Design Features
- Threaded file I/O for responsive interface  
- Automatic signal type detection and routing  
- Flexible unit conversion system  
- Graceful degradation when optional packages are unavailable  

---

<sub>Built with ❤️ by Ishmael | For QUASAR's Coding Screener </sub>
