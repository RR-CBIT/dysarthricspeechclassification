import librosa
import numpy as np

def extract_features(file_path, max_pad_len=1024, n_mfcc=128):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Ensure MFCCs are padded or truncated to the required length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    # Now reshape the MFCCs to fit the model's expected input shape of (16, 8, 1)
    # We're assuming you want to reshape 128 MFCCs into (16, 8, 1)
    mfcc = mfcc.T  # Transpose to shape: (time_steps, features)

    # The shape after transposition should be (max_pad_len, n_mfcc)
    # We need to reshape it into (16, 8, 1)
    # Let's reshape it into (16, 8, 1)
    
    reshaped_mfcc = mfcc[:16, :8]  # Use first 16 time steps and 8 features for each step
    reshaped_mfcc = np.expand_dims(reshaped_mfcc, axis=-1)  # Add the channel dimension (1)

    # Reshape to (1, 16, 8, 1) for a batch size of 1
    reshaped_mfcc = np.expand_dims(reshaped_mfcc, axis=0)
    
    # Print to debug the shapes
    print(f"Original MFCC shape: {mfcc.shape}")
    print(f"Reshaped MFCC shape: {reshaped_mfcc.shape}")

    return reshaped_mfcc