import numpy as np
import pandas as pd

def create_sequence_data(data, input_seq_len, output_seq_len=1, target_columns=None):
    """
    Create sequence data for LSTM from time series data.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Time series data. For multivariate, shape should be (samples, features)
    input_seq_len : int
        Length of input sequences
    output_seq_len : int
        Length of output sequences (for multi-step prediction)
    target_columns : list or None
        Indices or column names of target variables. If None, all columns are used as targets.
        
    Returns:
    --------
    X : numpy.ndarray
        Input sequences with shape (samples, input_seq_len, features)
    y : numpy.ndarray
        Target sequences with shape (samples, output_seq_len, n_targets)
    """
    # Convert to numpy array if DataFrame
    if hasattr(data, 'values'):
        if target_columns is not None:
            # Save column names/indices for targets
            if isinstance(target_columns[0], str):
                target_columns = [list(data.columns).index(col) for col in target_columns]
        data = data.values
    
    # Default: use all features as targets
    if target_columns is None:
        target_columns = list(range(data.shape[1])) if len(data.shape) > 1 else [0]
    
    n_features = 1 if len(data.shape) == 1 else data.shape[1]
    
    # Reshape if univariate
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    X, y = [], []
    
    # Create sequences
    for i in range(len(data) - input_seq_len - output_seq_len + 1):
        # Input sequence
        X.append(data[i:(i + input_seq_len)])
        
        # Output sequence (target)
        target_seq = data[(i + input_seq_len):(i + input_seq_len + output_seq_len), target_columns]
        y.append(target_seq)
    
    return np.array(X), np.array(y)

def split_train_test(X, y, test_ratio=0.2, shuffle=False, random_state=None):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input sequences
    y : numpy.ndarray
        Target sequences
    test_ratio : float
        Proportion of the data to include in the test split
    shuffle : bool
        Whether to shuffle the data before splitting
    random_state : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Training and testing splits
    """
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    split_idx = int(X.shape[0] * (1 - test_ratio))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train=None, y_test=None, scaler=None):
    """
    Scale the data using the provided scaler.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training input sequences
    X_test : numpy.ndarray
        Testing input sequences
    y_train : numpy.ndarray or None
        Training target sequences
    y_test : numpy.ndarray or None
        Testing target sequences
    scaler : object or None
        Scaler object with fit_transform and transform methods
        If None, returns the original data
        
    Returns:
    --------
    Scaled versions of the input arrays
    """
    if scaler is None:
        if y_train is not None and y_test is not None:
            return X_train, X_test, y_train, y_test
        return X_train, X_test
    
    # Get original shapes
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    
    # Reshape to 2D for scaling
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    
    # Fit and transform on training data
    X_train_scaled = scaler.fit_transform(X_train_2d)
    # Transform test data
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to original shape
    X_train_scaled = X_train_scaled.reshape(X_train_shape)
    X_test_scaled = X_test_scaled.reshape(X_test_shape)
    
    # Scale y if provided
    if y_train is not None and y_test is not None:
        y_train_shape = y_train.shape
        y_test_shape = y_test.shape
        
        # Create a new instance of the same scaler type
        y_scaler = type(scaler)()
        
        # Reshape to 2D for scaling
        y_train_2d = y_train.reshape(-1, y_train.shape[-1])
        y_test_2d = y_test.reshape(-1, y_test.shape[-1])
        
        # Fit and transform
        y_train_scaled = y_scaler.fit_transform(y_train_2d)
        y_test_scaled = y_scaler.transform(y_test_2d)
        
        # Reshape back
        y_train_scaled = y_train_scaled.reshape(y_train_shape)
        y_test_scaled = y_test_scaled.reshape(y_test_shape)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler
    
    return X_train_scaled, X_test_scaled

def inverse_transform_predictions(y_pred, y_scaler, original_shape=None):
    """
    Inverse transform scaled predictions.
    
    Parameters:
    -----------
    y_pred : numpy.ndarray
        Scaled predictions
    y_scaler : object
        Scaler used to scale the targets
    original_shape : tuple or None
        Original shape to reshape to after inverse transform
        
    Returns:
    --------
    numpy.ndarray
        Inverse transformed predictions
    """
    # Store original shape if provided
    if original_shape is None:
        original_shape = y_pred.shape
    
    # Reshape to 2D for inverse transform
    y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
    
    # Inverse transform
    y_pred_inverse = y_scaler.inverse_transform(y_pred_2d)
    
    # Reshape back to original shape
    return y_pred_inverse.reshape(original_shape)