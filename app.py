import os
import tempfile
import subprocess
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Allowed file extensions
ALLOWED_AUDIO_VIDEO_EXTENSIONS = {'mp3', 'mp4', 'wav'}

# Streamlit App
st.title("Wav2Lip")
st.write("Upload a video file and an audio file for processing.")

# File Upload
video_file = st.file_uploader("Upload Video File (mp4):", type=['mp3', 'mp4', 'wav', 'jpg', 'jpeg', 'png'])
audio_file = st.file_uploader("Upload Audio File (mp3, wav):", type=['mp3', 'mp4', 'wav'])

# Path to checkpoint
checkpoint_path = "checkpoints/wav2lip_gan.pth"

# Initialize session state variables
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None  # Tracks the processing state
if "processed_file_path" not in st.session_state:
    st.session_state.processed_file_path = None  # Tracks the processed file path

# Process files on button click
if st.button("Process Files"):
    if video_file is None or audio_file is None:
        st.error("Please upload both video and audio files!")
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            # Save uploaded files temporarily
            video_path = os.path.join(tempdir, video_file.name)
            audio_path = os.path.join(tempdir, audio_file.name)
            output_path = "./results/result_voice.mp4"

            with open(video_path, "wb") as f:
                f.write(video_file.read())
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())

            # Run the inference script
            try:
                st.session_state.processing_status = "Processing files... This may take a moment."
                st.info(st.session_state.processing_status)

                result = subprocess.run(
                    ['python', 'inference.py', '--checkpoint_path', checkpoint_path, '--face', video_path, '--audio', audio_path],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    st.session_state.processing_status = "Processing complete!"
                    st.session_state.processed_file_path = output_path
                    st.success(st.session_state.processing_status)
                else:
                    st.session_state.processing_status = None
                    st.error(f"Error during processing:\n{result.stderr}")
            except Exception as e:
                st.session_state.processing_status = None
                st.error(f"An error occurred: {e}")

# Provide the download button if the file was successfully processed
if st.session_state.processed_file_path and os.path.exists(st.session_state.processed_file_path):
    st.success("Download your processed file below:")
    with open(st.session_state.processed_file_path, "rb") as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name="result_voice.mp4",
            mime="video/mp4"
        )