import gradio as gr
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your Models class
from Development.models import RCNN 
from Development.data_loading import load_data_for_inference

# Load model
device = 'cpu'
model = RCNN(input_size=1500).float().to(device)

# Load the trained weights
model.load_state_dict(torch.load("model_weightsRCNN12dB.pt", map_location=torch.device(device)))
model.eval()

def process_uploaded_file(uploaded_file):
    """Handles the uploaded H5 file and returns the denoised ECG signal, plot, and denoised vector file."""
    
    # Load noisy ECG signal
    noisy_signal = load_data_for_inference(uploaded_file)
    
    # Throw an error if not valid signal.
    if noisy_signal is None:
        return "Error: Invalid file or format."
    
    # Ensure the signal length is 1500 if it’s larger than that
    if noisy_signal.shape[1] > 1500:  # Works for 2D array with shape (1, 15000)
        noisy_signal = noisy_signal[:, :1500]  # Slice the second dimension to get the first 1500 samples
    
    # Convert to Tensor
    noisy_signal_tensor = torch.tensor(noisy_signal).float().unsqueeze(0).unsqueeze(0)

    # Run through the model
    with torch.no_grad():
        denoised_signal = model(noisy_signal_tensor)
        denoised_signal_np = denoised_signal.squeeze().numpy()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(noisy_signal.squeeze().numpy(), label="Noisy Signal", color='r')
    plt.plot(denoised_signal_np, label="Denoised Signal", color='b')
    plt.title("Noisy vs Denoised ECG Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode image as base64 string for Gradio
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # Create a temporary file for the denoised signal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    with h5py.File(temp_file.name, 'w') as f:
        f.create_dataset("denoised_signal", data=denoised_signal_np)
    
    # Return the plot and the temporary denoised signal file
    return plot_base64, temp_file.name

# Create Gradio interface
iface = gr.Interface(
    fn=process_uploaded_file,
    inputs=gr.File(type="filepath"),  # File upload input
    outputs=["image", "file"],  # Show plot as image, and return the denoised file
    title="ECG Electrode Motion Denoising with Deep Learning",
    description="Upload a noisy ECG signal in HDF5 format, and receive a denoised version.",
)

# Launch Gradio app
iface.launch()
