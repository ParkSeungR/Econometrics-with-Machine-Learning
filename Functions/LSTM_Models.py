import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten
from keras.layers import ConvLSTM2D, BatchNormalization
import tensorflow as tf

def create_lstm_model(X_shape, output_seq_len, lstm_type='vanilla', units=50, 
                     dropout_rate=0.2, layers=2, cnn_filters=64, cnn_kernel_size=3, 
                     bidirectional=False, return_sequences_final=False):
    """
    Create various types of LSTM models.
    
    Parameters:
    -----------
    X_shape : tuple
        Input shape of data: (samples, time_steps, features)
    output_seq_len : int
        Length of output sequences (for multi-step prediction)
    lstm_type : str
        Type of LSTM model: 'vanilla', 'stacked', 'bidirectional', 'cnn_lstm', 'convlstm'
    units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate between LSTM layers
    layers : int
        Number of LSTM layers for stacked LSTM
    cnn_filters : int
        Number of filters for CNN layers
    cnn_kernel_size : int
        Kernel size for CNN layers
    bidirectional : bool
        Whether to use bidirectional LSTM
    return_sequences_final : bool
        Whether the final LSTM layer should return sequences
        
    Returns:
    --------
    model : keras.models.Sequential
        Compiled LSTM model
    """
    model = Sequential()
    
    if lstm_type == 'vanilla':
        # Simple LSTM model with a single LSTM layer
        model.add(LSTM(units, activation='relu', input_shape=(X_shape[1], X_shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_seq_len))
    
    elif lstm_type == 'stacked':
        # Stacked LSTM with multiple LSTM layers
        for i in range(layers):
            return_sequences = True if i < layers - 1 or return_sequences_final else False
            if i == 0:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences, 
                             input_shape=(X_shape[1], X_shape[2])))
            else:
                model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        if return_sequences_final:
            model.add(TimeDistributed(Dense(1)))
        else:
            model.add(Dense(output_seq_len))
    
    elif lstm_type == 'bidirectional':
        # Bidirectional LSTM
        for i in range(layers):
            return_sequences = True if i < layers - 1 or return_sequences_final else False
            if i == 0:
                model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=return_sequences), 
                                     input_shape=(X_shape[1], X_shape[2])))
            else:
                model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=return_sequences)))
            model.add(Dropout(dropout_rate))
        
        if return_sequences_final:
            model.add(TimeDistributed(Dense(1)))
        else:
            model.add(Dense(output_seq_len))
    
    elif lstm_type == 'cnn_lstm':
        # CNN-LSTM: CNN layers followed by LSTM layers
        model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', 
                       input_shape=(X_shape[1], X_shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate/2))
        
        # Add more CNN layers if needed
        model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate/2))
        
        # Add LSTM layers
        for i in range(layers-1):
            model.add(LSTM(units, activation='relu', return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        model.add(LSTM(units, activation='relu', return_sequences=return_sequences_final))
        model.add(Dropout(dropout_rate))
        
        if return_sequences_final:
            model.add(TimeDistributed(Dense(1)))
        else:
            model.add(Dense(output_seq_len))
    
    elif lstm_type == 'convlstm':
        # ConvLSTM: For spatio-temporal data
        # For ConvLSTM, input shape should be (samples, time_steps, rows, cols, features)
        # We need to reshape the input data for ConvLSTM
        # Assuming input is univariate or multivariate time series, we'll reshape it
        
        # Note: This assumes X_shape[1] (time_steps) is divisible by spatial_dim
        spatial_dim = int(np.sqrt(X_shape[1]))
        remainder = X_shape[1] % spatial_dim
        
        if remainder != 0:
            spatial_dim = X_shape[1]  # Use 1D spatial dimension if not divisible
            model.add(ConvLSTM2D(filters=cnn_filters, kernel_size=(1, cnn_kernel_size),
                             activation='relu', 
                             input_shape=(1, spatial_dim, X_shape[2], 1),
                             return_sequences=return_sequences_final))
        else:
            model.add(ConvLSTM2D(filters=cnn_filters, kernel_size=(cnn_kernel_size, cnn_kernel_size),
                             activation='relu', 
                             input_shape=(1, spatial_dim, spatial_dim, X_shape[2]),
                             return_sequences=return_sequences_final))
            
        model.add(BatchNormalization())
        
        if return_sequences_final:
            model.add(TimeDistributed(Flatten()))
            model.add(TimeDistributed(Dense(output_seq_len)))
        else:
            model.add(Flatten())
            model.add(Dense(output_seq_len))
    
    else:
        raise ValueError(f"Unknown LSTM type: {lstm_type}. Available types: 'vanilla', 'stacked', 'bidirectional', 'cnn_lstm', 'convlstm'")
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model


def prepare_convlstm_data(X, spatial_dim=None):
    """
    Reshape data for ConvLSTM input.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data with shape (samples, time_steps, features)
    spatial_dim : int or None
        Spatial dimension to reshape to. If None, will try to find a square root.
        
    Returns:
    --------
    X_reshaped : numpy.ndarray
        Reshaped data with shape (samples, 1, rows, cols, features)
    """
    samples, time_steps, features = X.shape
    
    if spatial_dim is None:
        # Try to find a square spatial dimension
        spatial_dim = int(np.sqrt(time_steps))
        if spatial_dim * spatial_dim != time_steps:
            # If not a perfect square, use 1D spatial dimension
            spatial_dim = time_steps
            X_reshaped = X.reshape(samples, 1, spatial_dim, features, 1)
            return X_reshaped
    
    # Reshape to 2D spatial dimensions
    X_reshaped = X.reshape(samples, 1, spatial_dim, spatial_dim, features)
    return X_reshaped


def train_evaluate_lstm(X_train, y_train, X_test, y_test, lstm_type='vanilla', batch_size=32, epochs=100,
                       validation_split=0.2, patience=10, **lstm_params):
    """
    Train and evaluate an LSTM model.
    
    Parameters:
    -----------
    X_train, y_train : numpy.ndarray
        Training data
    X_test, y_test : numpy.ndarray
        Testing data
    lstm_type : str
        Type of LSTM model
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs
    validation_split : float
        Proportion of training data to use for validation
    patience : int
        Patience for early stopping
    lstm_params : dict
        Additional parameters for create_lstm_model
        
    Returns:
    --------
    model : keras.models.Sequential
        Trained LSTM model
    history : keras.callbacks.History
        Training history
    predictions : numpy.ndarray
        Predictions on test data
    """
    from keras.callbacks import EarlyStopping
    
    # Special preprocessing for ConvLSTM
    if lstm_type == 'convlstm':
        X_train = prepare_convlstm_data(X_train)
        X_test = prepare_convlstm_data(X_test)
    
    # Get output sequence length
    if len(y_train.shape) > 2:
        output_seq_len = y_train.shape[1]
    else:
        output_seq_len = 1
    
    # Create the model
    model = create_lstm_model(X_train.shape, output_seq_len, lstm_type=lstm_type, **lstm_params)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MSE: {mse:.4f}')
    
    return model, history, predictions