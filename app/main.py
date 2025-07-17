import streamlit as st
import tempfile
import os
import sys
import shutil
import subprocess

# Tambahkan root path agar bisa akses 'models/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unattended_tracker import detect_unattended

def convert_to_h264(input_path, output_path):
    """Gunakan FFmpeg untuk convert video ke H.264 agar bisa diputar di browser"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

st.set_page_config(layout="wide")
st.title("🎒 Unattended Object Detection App")
st.markdown("Upload video, lalu tekan **Jalankan Deteksi** untuk mendeteksi objek unattended.")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    input_path = tfile.name

    if st.button("🚀 Jalankan Deteksi", type="primary"):
        raw_output = os.path.join("results", "raw_output.mp4")
        final_output = os.path.join("results", "detected_output.mp4")
        os.makedirs("results", exist_ok=True)

        with st.spinner("🔍 Deteksi sedang berlangsung..."):
            detect_unattended(input_path, raw_output)
            convert_to_h264(raw_output, final_output)

        if os.path.exists(final_output):
            st.success("✅ Deteksi selesai!")

            st.subheader("📌 Hasil Deteksi (klik Play):")
            with open(final_output, "rb") as f:
                video_bytes = f.read()
                st.video(video_bytes)  # Ini bisa autoplay jika browser support

            st.download_button(
                "⬇ Download Video Hasil",
                data=video_bytes,
                file_name="hasil_deteksi.mp4",
                mime="video/mp4"
            )
        else:
            st.error("❌ Gagal menghasilkan video hasil deteksi.")