import pandas as pd 
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

excel_file_path = "prob01_data.xlsx"
csv_file_path = "prob01_data.csv"
csv_df = pd.read_csv(csv_file_path, header=None)

original_data = csv_df.to_numpy()
audio_data = original_data[:, 1:]

fs = 44100
n_samples = len(audio_data)
bandwidth = 100

spikes = [2027, 4080, 6150, 8172, 10225, 12279, 14317, 16387, 18374, 20462]

def convert_to_csv():
    """
    Convert excel files to CSV files to be viewed on VS Code
    """
    excel_df = pd.read_excel(excel_file_path, header=None)
    excel_df.to_csv(csv_file_path, index=False, header=None)
    print("Converted to CSV successfully.")

def convert_to_wav(data):
    """
    Convert data into WAV sound file format 
    """
    max_val = np.max(np.abs(data)) #need to normalize the data, since WAV files expect values from -1 to 1.
    if max_val > 0:
        data_normalized = data / max_val
    else:
        data_normalized = data

    fs = 44100
    wav.write('clean_prob1_sound.wav', fs, data_normalized.astype(np.float32))
    print("WAV sound file saved.")

def find_spikes():
    """
    To find the high frequencies of the beep 
    """
    fft_domain = np.fft.fft(audio_data, axis=0)
    freqs = np.fft.fftfreq(n_samples, d=1/fs)
    magnitude = np.abs(fft_domain)

    plt.subplot(2, 1, 1)
    plt.plot(freqs[:n_samples//2], magnitude[:n_samples//2, 0]) #plot only positive freqs
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    ax = plt.gca()
    ax.format_coord = lambda x, y: f'x={x:.2f}, y={y:.2f}'
    plt.show()

    return fft_domain, freqs

def apply_mask(fft_domain, freqs):
    """
    Apply a mask consisting of 0s and 1s to remove the unwanted beep frequencies
    """
    mask = np.ones_like(fft_domain)
    for spike in spikes:
        indices_to_remove = np.where((np.abs(freqs) > spike - bandwidth) & (np.abs(freqs) < spike + bandwidth))
        mask[indices_to_remove] = 0
    
    clean_freq = fft_domain * mask
    clean_time = np.fft.ifft(clean_freq).real

    return clean_time

if __name__ == "__main__":
    cleaned_channels = []
    for i in range(audio_data.shape[1]):
        channel_data = audio_data[:, i]
        fft_domain = np.fft.fft(channel_data)
        freqs = np.fft.fftfreq(n_samples, d=1/fs)
        cleaned_col = apply_mask(fft_domain, freqs)
        cleaned_channels.append(cleaned_col)
        
    final_clean_audio = np.column_stack(cleaned_channels)
    
    convert_to_wav(final_clean_audio)