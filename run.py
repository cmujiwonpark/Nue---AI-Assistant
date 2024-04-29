import cv2
import wave
import pyaudio
import numpy as np
import streamlit as st
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import av
from queue import deque
import pydub
import base64

client = OpenAI(api_key='[INSERT KEY HERE]')
lock = threading.Lock()

frames_deque_lock = threading.Lock()
frames_deque: deque = deque([])

for file in os.listdir('stt'):
    os.remove(f'stt/{file}')

async def queued_audio_frames_callback(frames) -> av.AudioFrame:
    with frames_deque_lock:
        frames_deque.extend(frames)

    # Return empty frames to be silent.
    new_frames = []
    for frame in frames:
        input_array = frame.to_ndarray()
        new_frame = av.AudioFrame.from_ndarray(
            np.zeros(input_array.shape, dtype=input_array.dtype),
            layout=frame.layout.name,
        )
        new_frame.sample_rate = frame.sample_rate
        new_frames.append(new_frame)

    return new_frames

def transcribe_audio(filename):
    print("Transcribing!!!")
    idx = filename.split("-")[-1].split(".")[0]
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        ]
    )
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input= completion.choices[0].message.content
    )
    response.stream_to_file(f'stt/response-{idx}.mp3')
    return transcript, completion.choices[0].message.content, response

def stream_data(s):
    for word in s.split():
        yield word + " "
        time.sleep(0.05)
    


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
        # sleep for duration of audio
        audio = pydub.AudioSegment.from_file(file_path)
        time.sleep(audio.duration_seconds)

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.length = len(os.listdir('nue'))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        x, y = img.shape[1], img.shape[0]
        iron_man = cv2.imread(f"nue/nue_{self.count % self.length}.jpg", cv2.IMREAD_COLOR)
        x_iron, y_iron = iron_man.shape[1], iron_man.shape[0]
        iron_man = cv2.resize(iron_man, (int(y * x_iron / y_iron), y))
        concatenated_image = np.concatenate([img, iron_man], axis=1)
        img = cv2.resize(concatenated_image, (x, y))

        self.count += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")
# Initialize isLoggedIn in the session state
if 'isLoggedIn' not in st.session_state:
    st.session_state.isLoggedIn = False
    st.session_state.username = ""

if not st.session_state.isLoggedIn:
    username = st.text_input('Username', 'Your Username')
    st.session_state.username = username
    password = st.text_input('Password', 'Your Password', type='password')

    # Add a button to simulate a login action
    if st.button('Login'):
        if password == "1234":
            st.session_state.isLoggedIn = True

else:
    st.success('Logged in successfully!')
    # Add the content you want to display after logging in
    st.write(f"Welcome {st.session_state.username}!")
    webrtc_ctx = webrtc_streamer(key="example", 
                                    video_processor_factory=VideoProcessor,
                                    queued_audio_frames_callback=queued_audio_frames_callback,
                                    media_stream_constraints={"video": True, "audio": True}, 
                                )    
    prev_transcript = None
    prev_completion = None
    idx = 0
    response_idx = 0
    with st.container(height=400):
        while webrtc_ctx.state.playing:
            sound_chunk = pydub.AudioSegment.empty()

            if len(frames_deque) < 150:
                time.sleep(0.1)
                continue

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)


            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk.export(f"stt/output-{idx}.wav", format="wav")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(transcribe_audio, f"stt/output-{idx}.wav")
                    transcript, completion, response = future.result()
            
            if transcript == prev_transcript or completion == prev_completion:
                continue

            prev_transcript = transcript
            prev_completion = completion
            
            with st.chat_message("User"):
                st.write_stream(stream_data(transcript))
            with st.chat_message("ai"):
                st.write_stream(stream_data(completion))
            idx += 1

            if os.path.exists(f"stt/response-{response_idx}.mp3"):
                autoplay_audio(f"stt/response-{response_idx}.mp3")
                response_idx += 1
                with frames_deque_lock:
                    frames_deque: deque = deque([])


