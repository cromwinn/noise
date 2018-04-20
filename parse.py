import numpy as np
import scipy.io.wavfile as wav
import librosa
import os


def load_training_sample(filename, block_size=2048):

    data, sr = librosa.load(filename)
    x_blocks = dice_np_audio_into_blocks(data, block_size)
    X = time_blocks_to_fft_blocks(x_blocks)
    return X

def time_blocks_to_fft_blocks(blocks_time_domain):
    fft_blocks = []
    for block in blocks_time_domain:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(new_block)
    return fft_blocks

def fft_blocks_to_time_blocks(blocks_freq_domain):
    time_blocks = []
    for block in blocks_freq_domain:
        num_elems = round(block.shape[0]/2)
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks


def dice_np_audio_into_blocks(sample_np, block_size):
    block_lists = []
    total_frames = len(sample_np)
    frame_count = 0
    while frame_count < total_frames:
        block = sample_np[frame_count:frame_count+block_size]
        if len(block) < block_size:
            padding = np.zeros((block_size - len(block),))
            block = np.concatenate((block, padding))
        block_lists.append(block)
        frame_count += block_size
    return block_lists

def convert_wav_file_to_nptensor(filename, block_size, out_file):
    chunks_X = []
    X = load_training_sample(filename, block_size)




if __name__ == '__main__':

    data, sr = librosa.load('test_slice.wav')
    # sr, data = wav.read('test_slice.wav')
    block_lists = dice_np_audio_into_blocks(data, 2048)
    fft_blocks = time_blocks_to_fft_blocks(block_lists)
    time_blocks = fft_blocks_to_time_blocks(fft_blocks)
    song = np.concatenate(time_blocks)
    song = song.astype('float32')
    librosa.output.write_wav('sandbox_test.wav', song, sr)
    # wav.write('sandbox_test.wav', sr, song)
