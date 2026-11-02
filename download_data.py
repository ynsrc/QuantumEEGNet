import os
import certifi
import numpy as np
import scipy.io
import scipy.signal as signal
import ssl
import urllib.request

def preprocess_bci_2a_file(file_path):
    """
    BCI Competition IV 2a preprocessing function fully compatible
    with QuantumEEGNet's dataset.py structure.
    """
    mat = scipy.io.loadmat(file_path)
    data_struct = mat['data'][0]

    all_trials = []
    all_labels = []

    orig_fs = 250

    # 1. Band-pass Filter Setup (8-30 Hz Mu/Beta band for Motor Imagery)
    nyq = 0.5 * orig_fs
    b, a = signal.butter(3, [8.0 / nyq, 30.0 / nyq], btype='band')

    for run in data_struct:
        run_data = run[0, 0]
        X = run_data['X']          # Shape: (samples, 25)
        trial = run_data['trial']  # Triggers
        y = run_data['y']          # Labels

        # Skip if trial or y is empty
        if trial.size == 0 or y.size == 0:
            continue

        triggers = trial.flatten()
        labels = y.flatten()

        # The first 22 channels are EEG; exclude the last 3 EOG channels
        eeg_signals = X[:, :22]

        # 2. Filtering (Zero-phase shift)
        filtered_eeg = signal.filtfilt(b, a, eeg_signals, axis=0)

        # 3. Z-Score Normalization (Per-channel standardization)
        mean = np.mean(filtered_eeg, axis=0, keepdims=True)
        std = np.std(filtered_eeg, axis=0, keepdims=True)
        norm_eeg = (filtered_eeg - mean) / (std + 1e-8)

        # 4. Slicing Motor Imagery Trials
        for trig, label in zip(triggers, labels):
            start = int(trig)

            # Extract 314 samples to match dataset.py expected shape
            slice_start = start + 124
            slice_end = slice_start + 314

            if slice_end <= len(norm_eeg):
                # Shape: (314, 22) -> transpose -> (22, 314)
                raw_segment = norm_eeg[slice_start:slice_end, :]
                trial_data = raw_segment.T.astype(np.float32)

                all_trials.append(trial_data)

                # Convert labels from 1, 2, 3, 4 to 0, 1, 2, 3
                all_labels.append(int(label) - 1)

    if len(all_trials) == 0:
        return np.empty((0, 22, 314), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(all_trials, dtype=np.float32), np.array(all_labels, dtype=np.int64)

def download_file(url, target_path):
    """Download a file safely to avoid server timeouts and blocks."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())


# --- DOWNLOAD AND PROCESSING LOOP ---
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

for i in range(1, 10):
    sub_str = f"{i:02d}"

    train_url = f"https://lampx.tugraz.at/~bci/database/001-2014/A{sub_str}T.mat"
    test_url = f"https://lampx.tugraz.at/~bci/database/001-2014/A{sub_str}E.mat"

    train_raw_path = os.path.join(data_dir, f"BCIC_S{sub_str}_T.mat")
    test_raw_path = os.path.join(data_dir, f"BCIC_S{sub_str}_E.mat")

    try:
        print(f"Downloading and processing subject {sub_str}...")
        tmp_train = train_raw_path + ".tmp"
        tmp_test = test_raw_path + ".tmp"

        download_file(train_url, tmp_train)
        download_file(test_url, tmp_test)

        x_tr, y_tr = preprocess_bci_2a_file(tmp_train)
        x_te, y_te = preprocess_bci_2a_file(tmp_test)

        # If the test file has no labels or trials, split the training data 80/20
        if x_te.size == 0 and x_tr.size > 0:
            print(f"  -> Subject {sub_str} has no test labels; splitting from training data...")
            split_idx = int(len(x_tr) * 0.8)

            x_te, y_te = x_tr[split_idx:], y_tr[split_idx:]
            x_tr, y_tr = x_tr[:split_idx], y_tr[:split_idx]

        # Ensure labels are one-dimensional
        y_tr = y_tr.squeeze().astype(np.int64)
        y_te = y_te.squeeze().astype(np.int64)

        # Save in MATLAB format
        scipy.io.savemat(train_raw_path, {'x_train': x_tr, 'y_train': y_tr})
        scipy.io.savemat(test_raw_path, {'x_test': x_te, 'y_test': y_te})

        # Remove temporary files
        if os.path.exists(tmp_train): os.remove(tmp_train)
        if os.path.exists(tmp_test): os.remove(tmp_test)

        print(f"Subject {sub_str} complete! Train: {x_tr.shape}, Test: {x_te.shape}")
    except Exception as e:
        print(f"Error processing subject {sub_str}: {e}")
