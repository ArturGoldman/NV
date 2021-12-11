import io

import matplotlib.pyplot as plt


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(5, 20))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def plot_attention_to_buf(attention_tensor, name=None):
    plt.figure(figsize=(20, 20))
    plt.imshow(attention_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
