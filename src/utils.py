import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_pickle(obj, filepath):
    """Saves a Python object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Saved artifact: {filepath}")

def load_pickle(filepath):
    """Loads a Python object from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_numpy(array, filepath):
    """Saves a numpy array."""
    np.save(filepath, array)
    print(f"✅ Saved data: {filepath}")

def load_numpy(filepath):
    """Loads a numpy array."""
    return np.load(filepath)

def save_plot(figure, filename, output_dir):
    """Saves a matplotlib figure."""
    path = os.path.join(output_dir, filename)
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    print(f"📊 Saved plot: {path}")