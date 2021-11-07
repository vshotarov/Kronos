from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import os
import tempfile
import pyaudio
import wave

load_dotenv()

WATSON_API_KEY = os.getenv("WATSON_API_KEY")
WATSON_URL = os.getenv("WATSON_URL")

authenticator = IAMAuthenticator(WATSON_API_KEY)
text_to_speech = TextToSpeechV1(
    authenticator=authenticator
)

text_to_speech.set_service_url(WATSON_URL)

p = pyaudio.PyAudio()

def synthesize(text):
    tmp_file = tempfile.mkstemp(".wav")[1]

    with open(tmp_file, "wb") as af:
        af.write(text_to_speech.synthesize(
            text, voice="en-GB_JamesV3Voice",
            accept="audio/wav").get_result().content)

    wf = wave.open(tmp_file)
    stream = p.open(
        format=pyaudio.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.close()
    wf.close()

    os.remove(tmp_file)
