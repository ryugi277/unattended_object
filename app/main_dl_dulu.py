import streamlit as st
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unattended_tracker import detect_unattended

st.set_page_config(layout="wide")
st.title("Unattended Object Detection App")

st.markdown("Upload video, jalankan deteksi, lalu lihat hasilnya langsung di browser.")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Simpan sementara file video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    temp_video_path = tfile.name

    st.subheader("Video Asli")
    st.video(temp_video_path)

    if st.button("Jalankan Deteksi"):
        output_path = os.path.join("results", "detected_output.mp4")
        os.makedirs("results", exist_ok=True)

        with st.spinner("Sedang menjalankan deteksi, harap tunggu..."):
            detect_unattended(temp_video_path, output_path)

        if os.path.exists(output_path):
            st.success("Deteksi selesai!")

            st.subheader("Hasil Deteksi:")
            st.video(output_path)

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="Download Video Hasil",
                data=video_bytes,
                file_name="detected_output.mp4",
                mime="video/mp4"
            )
        else:
            st.error("Gagal membuat output video.")