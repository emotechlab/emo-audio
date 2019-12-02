import argparse
import librosa
from librosa import spectrum
import numpy.random as npr
import numpy as np
import os

def get_signal_length():
    # Assuming 16KHz sample rate this would be 10ms -> 1s audio sample
    return npr.randint(5012, 16000)

# Rust and python potential use different FFT solvers so restricting
# the parameters to try and minimise the areas it can go off while
# still being representative
def get_stft_params():
    nfft = npr.choice([512, 1024, 2048])
    win_length = npr.choice([nfft/8, nfft/4, nfft/2, nfft])
    current = win_length / 4
    choices = [current]
    while current > 0:
        current = current * 0.5
        choices.append(current)
    hop_length = npr.choice(choices)
    power = npr.choice([1.0, 2.0])
    return np.array([nfft, win_length, hop_length, power], dtype='float32')

def generate_audio():
    return npr.rand(get_signal_length())

def generate_spectrum_data(filename):
    audio = generate_audio()
    params = get_stft_params()
    stft = librosa.stft(y=audio, n_fft=int(params[0]),
            win_length=int(params[1]), hop_length=int(params[2]),
            window='hann', center=True, pad_mode='reflect')
    stft, _ = librosa.magphase(stft)
    if stft.ndim is 1:
        stft = np.expand_dims(stft, axis=0)

    mag_spectra = spectrum._spectrogram(y=audio, n_fft = int(params[0]),
            win_length=int(params[1]), hop_length=int(params[2]), 
            power=params[3], window='hann', center=True, pad_mode='reflect')[0]
    if mag_spectra.ndim is 1:
        mag_spectra = np.expand_dims(mag_spectra, axis=1)
    
    np.savez(filename, audio=audio, params=params, stft=stft, magnitude=mag_spectra)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate some test data")
    parser.add_argument('-s', '--samples', type=int, help='number of samples to generate', required=True)
    parser.add_argument('-f', '--folder', type=str, help='output_folder', required=True)

    args = parser.parse_args()
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    for i in range(args.samples):
        generate_spectrum_data(os.path.join(args.folder, f'data_{i}.npz'))

