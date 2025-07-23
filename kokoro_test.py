import subprocess
import sys
import os

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Install kokoro and soundfile if not already installed
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kokoro==0.7.16', 'soundfile'])

# Install espeak-ng (Linux only, will fail silently on non-Linux)
try:
    subprocess.run(['apt-get', '-qq', '-y', 'install', 'espeak-ng'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    pass  # Ignore errors on non-Linux systems

from kokoro import KPipeline
import soundfile as sf
import torch

# Only import IPython.display.Audio if running in a notebook
try:
    from IPython.display import display, Audio
    def play_audio(audio, rate, autoplay):
        display(Audio(data=audio, rate=rate, autoplay=autoplay))
except ImportError:
    def play_audio(audio, rate, autoplay):
        print(f"Audio playback not available outside notebook. {len(audio)} samples at {rate} Hz.")

pipeline = KPipeline(lang_code='b')
text = '''
Biashara Plus is specifically designed for Forever Living Business Owners (FBOs) to help them access and view prices of Forever Living products. The app provides real-time pricing, product descriptions, and images, making it easier for FBOs to make informed decisions about their purchases.
In essence, Biashara Plus serves as a vital tool that connects FBOs directly to the Forever Living product line, enhancing their business operations and sales efforts. If you have more questions or need further clarification about Biashara Plus or its relationship with Forever Living, feel free to ask! 😊🌟
'''
generator = pipeline(text, voice='af_heart')

# Create an 'audio_output' directory if it doesn't exist
output_dir = os.path.join(current_dir, 'audio_output')
os.makedirs(output_dir, exist_ok=True)

for i, (gs, ps, audio) in enumerate(generator):
    print(f"\nSegment {i}:")
    print(f"Text: {gs}")
    print(f"Phonemes: {ps}")
    
    # Save the audio file with a descriptive name
    filename = f'kokoro_segment_{i}.wav'
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio, 24000)
    print(f"Saved audio to: {filepath}")
    
    # Play audio if in notebook environment
    play_audio(audio, 24000, autoplay=(i==0)) 