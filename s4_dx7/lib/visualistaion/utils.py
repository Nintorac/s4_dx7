import io
from pathlib import Path
import soundfile as sf
import base64
from io import BytesIO

def fig_to_png_data_uri(fig):
    """
    Converts a given Matplotlib figure to a base64 encoded string of the PNG image,
    suitable for embedding in an HTML document as a data URI.

    Args:
    fig (matplotlib.figure.Figure): The Matplotlib figure to convert.

    Returns:
    str: A base64 encoded string of the figure's PNG image, with the data URI prefix.
    """
    # Create a BytesIO buffer to save the image
    buf = BytesIO()
    # Save the figure to the buffer in PNG format
    fig.savefig(buf, format='png')
    # Ensure the buffer's position is at the start
    buf.seek(0)
    # Convert the PNG image in the buffer to a base64 encoded string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    # Construct the full data URI scheme
    data_uri = f'data:image/png;base64,{image_base64}'
    
    return data_uri


def float_to_ogg_data_uri(x, sample_rate):
    """
    Converts audio data from float values to ogg format.
    
    Args:
        x (np.array): Array of float values representing the audio data.
        sample_rate (int): The sample rate of the audio data.
        
    Returns:
        str: The data URI representing the ogg audio file.
    """
    # Create an in-memory buffer
    output_buffer = io.BytesIO()
    # Convert float values to audio file in the buffer
    sf.write(output_buffer, x, sample_rate, format='OGG')

    # Seek to the beginning of the buffer
    output_buffer.seek(0)

    # Read the audio data from the buffer
    audio_data = output_buffer.read()

    # Encode the audio data to base64
    encoded_data = base64.b64encode(audio_data).decode()

    # Generate data URI
    data_uri = "data:audio/ogg;base64," + encoded_data

    # Return the data URI
    return data_uri

def float_to_ogg_file(x, sample_rate, f):
    """
    Converts audio data from float values to ogg format.
    
    Args:
        x (np.array): Array of float values representing the audio data.
        sample_rate (int): The sample rate of the audio data.
        
    Returns:
        str: The data URI representing the ogg audio file.
    """
    # Convert float values to audio file in the buffer
    sf.write(f, x, sample_rate, format='OGG')
