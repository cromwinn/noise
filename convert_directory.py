from data_utils.parse import *


input_directory = './datasets/bach_slices/'
output_filename = './datasets/bachNP'

freq = 22050
clip_len = 5
block_size = round(freq / 4)
max_seq_len = int(round((freq * clip_len) / block_size))
convert_wav_file_to_nptensor(input_directory, block_size, max_seq_len, output_filename)
