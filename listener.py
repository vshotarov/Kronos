import argparse
import pyaudio
import torch
import torchaudio
import wave

from wake_word_detection.model import WWDModel as WakeWordDetectionModel
from wake_word_detection.data import Preprocessor as WakeWordPreprocessor
from speech_recognition.model import STTModel as SpeechRecognitionModel
from speech_recognition.data import Preprocessor as SpeechRecognitionPreprocessor,\
    LanaguageModelDecoder
from intent_and_slot_inference.main import getIntentAndSlotModels, inferIntentAndSlots


def _load_model(model_class, state_dict_path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model

class Listener(object):
    def __init__(self, wake_word_model_state_path, speech_recognition_model_state_path,
            intent_inference_model_state_path, slot_filling_model_state_path,
            language_model_path, wake_notification_wav_path, intent_handler,
            synthesize_func=None):
        self.intent_handler = intent_handler
        self.synthesize_func = synthesize_func

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,channels=1,
            rate=8000,input=True)

        # Load models
        self.wake_word_detection_model = _load_model(
            WakeWordDetectionModel, wake_word_model_state_path)
        self.speech_recognition_model = _load_model(
            SpeechRecognitionModel, speech_recognition_model_state_path)
        self.intent_and_slot_models = getIntentAndSlotModels(
            intent_inference_model_state_path, slot_filling_model_state_path)

        self.wake_word_preprocessor = WakeWordPreprocessor()
        self.speech_recognition_preprocessor = SpeechRecognitionPreprocessor()
        self.language_model_decoder = LanaguageModelDecoder(language_model_path)
        self._initial_hidden, self._initial_c0 = [
            x for x in self.speech_recognition_model.get_initial_hidden(1)]

        # While listening, we are passing each buffer through a
        # wake word detection neural net. If that detects the wake
        # word then we start recording chunks that will be given
        # to the speech recognition neural net
        self.recording_speech = False
        self.speech_buffer = torch.tensor([])
        self.wake_buffer = torch.tensor([])

        # Store the length of the wake notification sound file in seconds
        wake_notification_wavfile = wave.open(wake_notification_wav_path)
        self.wake_notification_wave_duration =\
            wake_notification_wavfile.getnframes() / wake_notification_wavfile.getframerate()
        wake_notification_wavfile.close()
        self.wake_notification_wav_path = wake_notification_wav_path

    def play_wake_notification(self):
        wake_notification_wavfile = wave.open(self.wake_notification_wav_path)

        # Create an output pyaudio stream to play the notification sound
        stream = self.pyaudio.open(
            format=pyaudio.get_format_from_width(wake_notification_wavfile.getsampwidth()),
            channels=wake_notification_wavfile.getnchannels(),
            rate=wake_notification_wavfile.getframerate(),
            output=True)

        data = wake_notification_wavfile.readframes(1024)
        while data:
            stream.write(data)
            data = wake_notification_wavfile.readframes(1024)

        stream.close()
        wake_notification_wavfile.close()

    def start(self):
        self.recording_speech = False
        self.speech_buffer = torch.tensor([])
        self.wake_buffer = torch.tensor([])

        while True:
            try:
                current_chunk = bytearray(self.stream.read(4000))

                with torch.no_grad():
                    self.process_chunk(current_chunk)

            except KeyboardInterrupt as e:
                print("KeyboardInterrupt detected, exiting listener..")
                self.cleanup()
                break

    def process_chunk(self, chunk):
        torch_chunk = torch.frombuffer(chunk, dtype=torch.float32)

        self.wake_buffer = torch.cat((self.wake_buffer, torch_chunk))

        # Check if we have 1.5 seconds of recording
        # 8000 is sample rate
        # 1.5 is a second and a half
        if len(self.wake_buffer) < 8000 * 1.5:
            return

        # Normalize gain
        normalized_wake_buffer, _ = torchaudio.sox_effects.apply_effects_tensor(
            self.wake_buffer.unsqueeze(0), 8000, [["gain", "-n"]])

        # Wake word
        # The wake word model has been trained on samples that are 3 seconds long
        # so we can train it to ignore a lot of background chatter. Both the
        # wake word and the stop word, though, take far less than that to say
        # so we're essentially running the wake word detection on a recording of
        # 1.5 seconds, but we pad it with zeros to 3 seconds
        padded_wake_buffer = torch.cat(
            (normalized_wake_buffer[0], torch.zeros(12000))).unsqueeze(0)
        wake_word_action = ["wake","stop","pass"][
            self.wake_word_detection_model.classify(
                self.wake_word_preprocessor(padded_wake_buffer)).item()]

        # Clear the old samples from the buffer, so we always keep it to the last
        # second
        self.wake_buffer = self.wake_buffer[-8000 * 1:]

        if wake_word_action == "wake":
            if not self.recording_speech:
                # If we've detected the wake word and we're currently idle
                # (not recording speech) then switch to recording speech mode
                self.play_wake_notification()
                self.recording_speech = True
                self.speech_buffer = torch.tensor([])
                return
        elif wake_word_action == "stop":
            if self.recording_speech:
                # If we've detected the stop word and we're currently recording
                # speech, then stop, which switches us back to idle mode
                self.recording_speech = False
                return

        # Speech recognition
        # We keep a separate buffer for the speech recognition, so we can
        # easily impose different rules about their sizes. The process
        # is very similar to the wake word where we:
        # - append the current recorded chunk to the buffer
        # - normalize the gain across the whole buffer (in a copy, so we
        #   avoid normalizing the same chunk multiple times)
        # - get the raw speech recognition outputs from the model
        # - pass them through a ctc beam search
        # - pass the top 5 results from the ctc search through the intents
        #   and slots models
        # - pick the one result that has the highest intent classification
        #   confidence
        if self.recording_speech:
            self.speech_buffer = torch.cat((self.speech_buffer, torch_chunk))

            # Stop recording speech after 4 seconds
            # NOTE: This is what needs to be replaced with voice activity detection
            if self.speech_buffer.shape[0] >= 8000 * 4:
                self.recording_speech = False

                # Strip the first few frames, as otherwise we pick up some of the
                # wake word and also the wake notification
                self.speech_buffer = self.speech_buffer[
                    int((self.wake_notification_wave_duration) * 8000):]

                # Normalize gain
                normalized_speech_buffer, _ = torchaudio.sox_effects.apply_effects_tensor(
                    self.speech_buffer.unsqueeze(0), 8000, [["gain", "-n"]])

                #normalized_speech_buffer = torch.cat((normalized_speech_buffer[0], torch.zeros(12000))).unsqueeze(0)

                raw_speech_recognition_output = self.speech_recognition_model.recognize(
                    self.speech_recognition_preprocessor(normalized_speech_buffer),
                    self._initial_hidden, self._initial_c0)

                top_language_model_results = self.language_model_decoder.decode(
                    raw_speech_recognition_output, num_top_results=5)

                # Intents and slots
                # NOTE: Make it so we prefer having slots and we prefer
                # not having the PAD slot
                highest_confidence = 0
                most_confident_intent = -1
                most_confident_result = None
                most_confident_slots = []
                for result in top_language_model_results:
                    confidence, intent, slots = inferIntentAndSlots(
                        result, self.intent_and_slot_models)

                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        most_confident_intent = intent
                        most_confident_result = result
                        most_confident_slots = slots

                print(most_confident_result, "||", most_confident_intent,
                    "||", highest_confidence, "||", most_confident_slots)

                self.process_intent(most_confident_intent, most_confident_slots)
                #torchaudio.save("tmp.wav", normalized_speech_buffer, 8000)

                #import string
                #arg_maxes = torch.argmax(raw_speech_recognition_output[0], dim=1)
                #LABELS = {letter:i+2 for i, letter in enumerate(string.ascii_lowercase)}
                #LABELS[' '] = 1
                #LABELS['_'] = 0
                #LABEL_INDICES = {v:k for k,v in LABELS.items()}
                #no_ctc_output = ''.join([LABEL_INDICES[int(x)] for x in arg_maxes if x != 0])
                #print(no_ctc_output)

    def process_intent(self, intent, slots):
        handler = getattr(self.intent_handler, intent.replace(".","_"), None)
        if not handler:
            print("ERROR in Kronos.process_intent: "
                  "The intent %s doesn't have a handler." % intent)
            return

        response = handler(slots)

        if response:
            if self.synthesize_func:
                self.synthesize_func(response)
            else:
                print("Kronos:", response)

    def cleanup(self):
        pass

if __name__ == "__main__":
    class DummyIntentHandler():
        def __getattr__(self, attr):
            return lambda x: print(x)

    Listener("data/wake_word_model_state.torch","data/speech_model_state.torch",
           "data/intent_model_state.torch","data/slot_model_state.torch",
           "data/language_model.arpa","data/wake.wav",DummyIntentHandler()).start()
