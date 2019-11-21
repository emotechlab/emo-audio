import argparse
from librosa import spectrum
import numpy.random as npr
import numpy as np
import os

def get_signal_length():
    # Assuming 16KHz sample rate this would be 10ms -> 1s audio sample
    return npr.randint(160, 16000)

def get_stft_params():
    nfft = npr.randint(1, 1024) * 2
    win_length = npr.randint(4, nfft)
    hop_length = npr.randint(1, win_length/4)
    power = npr.random() * 1.5 + 0.5
    return np.array([nfft, win_length, hop_length, power], dtype='float32')

def generate_audio():
    return npr.rand(get_signal_length())

def generate_spectrum_data(filename):
    audio = generate_audio()
    params = get_stft_params()

    mag_spectra = spectrum._spectrogram(y=audio, n_fft = int(params[0]),
            win_length=int(params[1]), hop_length=int(params[2]), 
            power=params[3], window='hann', center=True, pad_mode='reflect')[0]
    
    np.savez(filename, audio=audio, params=params, magnitude=mag_spectra)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate some test data")
    parser.add_argument('-s', '--samples', type=int, help='number of samples to generate', required=True)
    parser.add_argument('-f', '--folder', type=str, help='output_folder', required=True)

    args = parser.parse_args()
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    for i in range(args.samples):
        generate_spectrum_data(os.path.join(args.folder, f'data_{i}.npz'))

