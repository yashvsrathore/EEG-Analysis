import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="EEG Analysis and Federated Learning", layout="wide")

# Load and preprocess EEG data
def load_and_preprocess(file_path, l_freq=1, h_freq=100):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.filter(l_freq, h_freq)
    return raw

# Calculate Power Spectral Density (PSD)
def calculate_psd(raw, fmin=1, fmax=100):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    psds, freqs = mne.time_frequency.psd_array_welch(
        data, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 4), n_overlap=int(sfreq * 2), verbose=False, sfreq=sfreq
    )
    return psds, freqs

# Extract band power
def extract_band_power(psds, freqs, band_range):
    freq_res = freqs[1] - freqs[0]
    band_idx = np.logical_and(freqs >= band_range[0], freqs <= band_range[1])
    band_power = np.sum(psds[:, band_idx], axis=1) * freq_res
    return band_power * 1e12  # Convert to pico-watts

# Process EEG data and extract features
def process_eeg_data(file_paths):
    features = []
    for file_path in file_paths:
        raw = load_and_preprocess(file_path)
        psds, freqs = calculate_psd(raw)
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 100)
        }
        band_power = {band: extract_band_power(psds, freqs, freq_range) for band, freq_range in bands.items()}
        features.append(band_power)
    return features

# Aggregate updates from multiple clients (simulated)
def aggregate_client_updates(client_updates):
    aggregated_update = np.mean(client_updates, axis=0)
    return aggregated_update

# Load data
st.title("EEG Analysis and Federated Learning")

# Upload EEG data files
rest_file = st.file_uploader("Upload Rest State EDF File", type=["edf"])
task_file = st.file_uploader("Upload Task State EDF File", type=["edf"])

if rest_file and task_file:
    st.write("*Processing Rest and Task State Data...*")
    rest_raw = load_and_preprocess(rest_file)
    task_raw = load_and_preprocess(task_file)
    
    # Display data description
    st.write("### Data Information")
    st.write(rest_raw.info)
    st.write(task_raw.info)
    
    # Display raw data plots
    st.write("### Raw Data Visualization")
    st.write("Rest State")
    st.pyplot(rest_raw.plot(n_channels=20, show=False))
    st.write("Task State")
    st.pyplot(task_raw.plot(n_channels=20, show=False))

    # Calculate and display PSD
    rest_psds, rest_freqs = calculate_psd(rest_raw)
    task_psds, task_freqs = calculate_psd(task_raw)
    
    # Display PSD comparison
    st.write("### Power Spectral Density (PSD) Comparison")
    plt.figure(figsize=(12, 6))
    plt.semilogy(rest_freqs, rest_psds.mean(axis=0), label='Rest')
    plt.semilogy(task_freqs, task_psds.mean(axis=0), label='Task')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV²/Hz)')
    plt.title('PSD Comparison: Rest vs Task')
    plt.legend()
    st.pyplot(plt)

    # Extract band power
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 100)
    }

    rest_band_power = {band: extract_band_power(rest_psds, rest_freqs, freq_range) 
                       for band, freq_range in bands.items()}
    task_band_power = {band: extract_band_power(task_psds, task_freqs, freq_range) 
                       for band, freq_range in bands.items()}

    # Band comparison visualization
    st.write("### Band Power Comparison")
    for band in bands.keys():
        plt.figure(figsize=(10, 6))
        plt.boxplot([rest_band_power[band], task_band_power[band]], labels=['Rest', 'Task'])
        plt.title(f"{band} Band Power Comparison")
        plt.ylabel("Power Units - (pW)")
        st.pyplot(plt)

    # Summary of findings
    st.write("### Summary of Findings")
    for band in bands.keys():
        rest_mean = np.mean(rest_band_power[band])
        task_mean = np.mean(task_band_power[band])
        percent_change = ((task_mean - rest_mean) / rest_mean) * 100
        
        st.write(f"*{band} Band:*")
        st.write(f"  Rest mean power: {rest_mean:.2f} pW")
        st.write(f"  Task mean power: {task_mean:.2f} pW")
        st.write(f"  Percent change: {percent_change:.2f}%")
        st.write("")

# Federated Learning Section
st.write("## Federated Learning Simulation")
if rest_file and task_file:
    # Example of simulated client-side processing
    client_features_1 = process_eeg_data([rest_file, task_file])
    client_features_2 = process_eeg_data([rest_file, task_file])
    client_features_3 = process_eeg_data([rest_file, task_file])
    
    client_updates = [client_features_1, client_features_2, client_features_3]
    global_update = aggregate_client_updates(client_updates)
    
    st.write("### Aggregated Update from Clients")
    st.write(global_update)

st.write("Upload EEG files to visualize the data and simulate the federated learning process.")