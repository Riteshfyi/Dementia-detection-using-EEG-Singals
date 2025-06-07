import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from glob import glob
import cv2
from tftb.processing import WignerVilleDistribution
from scipy.signal import hilbert, windows

epoch = 4
display_segment = 5

def Makesegment(df, freq):
    eeg_signal = df.iloc[:, 9].values
    sampling_rate = freq
    segment_duration = epoch
    segment_samples = int(segment_duration * sampling_rate)

    segments = []
    num_segments = len(eeg_signal) // segment_samples

    for i in range(num_segments):
        segment = eeg_signal[i * segment_samples : (i + 1) * segment_samples]
        segments.append(segment)

    segments = np.array(segments)
    print(f"Total Segments: {segments.shape[0]}, Segment Shape: {segments.shape[1]}")
    return segments

def compute_spwvd(segment, sampling_rate, window_size=31):
    analytic_segment = hilbert(segment)
    wvd = WignerVilleDistribution(analytic_segment)
    tfr, times, freqs = wvd.run()

    kaiser_window = windows.kaiser(window_size, beta=8)
    smoothed_tfr = np.apply_along_axis(lambda x: np.convolve(x, kaiser_window, mode='same'), axis=1, arr=tfr)
    return smoothed_tfr, times, freqs

folder_path = "./derivative _set_files/"
file_list = glob(os.path.join(folder_path, "*.set"))

for file in file_list:
    try:
        print(f"Processing file: {file}")
        new_folder = file[43:51]
        folder_path = "./output/spwvd_plotso2" + new_folder
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        raw = mne.io.read_raw_eeglab(file, preload=True)
        sfreq = raw.info['sfreq']
        dt = 1 / sfreq

        eeg_data = raw.get_data()
        ch_names = raw.ch_names
        df = pd.DataFrame(eeg_data.T, columns=ch_names)

        segments = Makesegment(df, sfreq)
        seg_no = 0

        for segment in segments:
            spwvd_tfd, times, freqs = compute_spwvd(segment, sfreq)

            n_freq_bins = spwvd_tfd.shape[0]
            freqs = np.fft.fftfreq(n_freq_bins, d=1/sfreq)
            pos_mask = freqs >= 0
            spwvd_tfd = spwvd_tfd[pos_mask, :]
            freqs     = freqs[pos_mask]

            max_freq = 60
            freq_mask = freqs <= max_freq
            spwvd_tfd = spwvd_tfd[freq_mask, :]
            freqs     = freqs[freq_mask]

            spwvd = np.abs(spwvd_tfd)
            spwvd = (spwvd - spwvd.min()) / (spwvd.max() - spwvd.min())

            N   = spwvd.shape[1]
            ts  = np.linspace(0, epoch, N)

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(
             spwvd,
             aspect='auto',
             origin='lower',
             extent=(ts[0], ts[-1], freqs[0], freqs[-1]),
             interpolation='nearest',
             cmap='jet'
            )
            ax.axis("off")

            save_path = os.path.join(folder_path, f"seg{seg_no}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close(fig)

            seg_no += 1

    except Exception as e:
        print(f"Error processing {file}: {e}")
