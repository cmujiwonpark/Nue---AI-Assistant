from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Load environment variables from .env file
load_dotenv()
client = OpenAI(api_key='')
# audio_file = open("stt_test_2.mp3", "rb")

# speech to text
# call API, generate transcript object
# https://platform.openai.com/docs/api-reference/audio/createTranscription
transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="text",
)

print(transcript)

# text to GPT response
# call API, generate completion object
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a fictional character that is trying to help the user."},
    {"role": "user", "content": transcript}
  ]
)


speech_file_path = Path(__file__).parent / "gpt_output.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input= completion.choices[0].message.content
)

response.stream_to_file(speech_file_path)