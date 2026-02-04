from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

def build_lstm_autoencoder(input_dim):
    print("[*] Building LSTM Architecture...")
    inputs = Input(shape=(1, input_dim))
    
    # Encoder
    encoded = LSTM(16, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(8, activation='relu', return_sequences=False)(encoded)
    
    # Decoder
    decoded = RepeatVector(1)(encoded)
    decoded = LSTM(8, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    
    # Output
    decoded = TimeDistributed(Dense(input_dim))(decoded)
    
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model