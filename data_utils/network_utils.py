from keras.models import Sequential
from keras.layers import TimeDistributed, Dense
from keras.layers.recurrent import LSTM, GRU

def create_lstm_network(num_timesteps, num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributed(Dense(1024), input_shape=(num_timesteps, num_frequency_dimensions)))
    for cur_unit in range(num_recurrent_units):
        model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
    model.add(TimeDistributed(Dense(num_frequency_dimensions)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
