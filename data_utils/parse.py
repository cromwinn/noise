import numpy as np
import scipy.io.wavfile as wav
import librosa
import os


def load_training_sample(filename, block_size=2048):

    data, sr = librosa.load(filename)
    x_blocks = dice_np_audio_into_blocks(data, block_size)
    y_blocks = x_blocks[1:]
    y_blocks.append(np.zeros(block_size)) #Add special end block composed of all zeros
    X = time_blocks_to_fft_blocks(x_blocks)
    Y = time_blocks_to_fft_blocks(y_blocks)
    return X, Y

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

def convert_wav_file_to_nptensor(directory, block_size, max_seq_len, out_file):
    files = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            files.append(directory+file)
    chunks_X = []
    chunks_Y = []
    num_files = len(files)
    for file_idx in range(num_files):
        file = files[file_idx]
        print("Processing: {}/{}".format(file_idx+1, num_files))
        print("Filename: {}".format(file))
        X, Y = load_training_sample(file, block_size)
        cur_seq = 0
        total_seq = len(X)
        while cur_seq + max_seq_len < total_seq:
            chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
            cur_seq += max_seq_len
    num_examples = len(chunks_X)
    num_dims_out = block_size * 2
    out_shape = (num_examples, max_seq_len, num_dims_out)
    x_data = np.zeros(out_shape)
    y_data = np.zeros(out_shape)
    for n in range(num_examples):
    	for i in range(max_seq_len):
            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_Y[n][i]
    	print('Saved example {}/{}'.format((n+1),num_examples))
    print('Flushing to disk...')
    mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
    std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
    std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
    x_data[:][:] -= mean_x #Mean 0
    x_data[:][:] /= std_x #Variance 1
    y_data[:][:] -= mean_x #Mean 0
    y_data[:][:] /= std_x #Variance 1

    np.save(out_file+'_mean', mean_x)
    np.save(out_file+'_var', std_x)
    np.save(out_file+'_x', x_data)
    np.save(out_file+'_y', y_data)
    print("Done!")

def save_generated_example(filename, generated_sequence, sample_frequency=44100):
    time_blocks = fft_blocks_to_time_blocks(generated_sequence)
    song = np.concatenate(time_blocks)
    song = song.astype('float32')
    librosa.output.write_wav(filename, song, sample_frequency)
    return


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
