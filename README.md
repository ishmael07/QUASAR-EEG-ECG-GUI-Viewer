# Ishmael's EEG/ECG Viewer

**Built for QUASAR's Coding Screener to visualize EEG and ECG data interactively in a single GUI.**  
This application allows you to load raw EEG/ECG CSV files and explore multiple channels in one window with scrollable, zoomable, and pannable plots.

---

## Features

- 📂 Load EEG/ECG CSV files directly from the GUI.  
- 📊 Display multiple channels simultaneously using interactive Plotly plots.  
- 🔄 Scroll, zoom, and pan through time-series signals.  
- ✅ Select which EEG, ECG, or reference channels to display.  
- ⚡ Normalize EEG signals for easier comparison.  
- 💾 Export the current plot as an HTML file.  
- ℹ️ Info panel displaying metadata, sampling rate, duration, and channel ranges.  

---

## How I Created It

I built the GUI using **Tkinter** and embedded **Plotly** charts for interactive plotting. Here’s my approach:

1. **CSV Parsing**  
   - Ignored lines starting with `#` as metadata.  
   - Loaded the rest into a Pandas DataFrame, using the first non-comment row as headers.  

2. **Channel Handling**  
   - Categorized channels into EEG, ECG, and Reference (CM).  
   - Ignored irrelevant columns (e.g., Trigger, ADC_Status).  

3. **Scaling Decisions**  
   - EEG signals (μV) are small and plotted normally.  
   - ECG signals (X1:LEOG, X2:REOG) are converted to mV for readability.  
   - Reference signals (CM) are large and plotted separately to avoid scale overlap.  

4. **Interactive Plot**  
   - Used Plotly to allow scroll, zoom, and pan.  
   - Added a range slider for easier navigation.  
   - Hover templates display channel, time, and amplitude.  

5. **GUI Layout**  
   - Designed a single-window layout with file selector, options panel, info panel, and plot embedded together.  

---

## AI Assistance

I used AI to:

- Help organize the **GUI layout** for clarity and usability.  
- Configure **Plotly plots**, subplots, and hover behavior efficiently.  
- Suggest **code structure** to keep it clean and maintainable.  

---

## How to Run

### 1. Clone the repository
``` bash
git clone https://github.com/<your-username>/quasar-eeg-ecg-viewer.git
cd quasar-eeg-ecg-viewer 
```

2. Install dependencies
```bash
pip install -r requirements.txt
``` 

3. Run in your preferred IDE!
