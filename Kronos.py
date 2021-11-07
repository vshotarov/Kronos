import argparse
from listener import Listener
from intent_handler import IntentHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos virtual assistant")
    parser.add_argument("trained_wake_word_model_path",
        help="path to the saved state of the trained wake word detection model")
    parser.add_argument("trained_speech_recognition_model_path",
        help="path to the saved state of the trained speech recognition model")
    parser.add_argument("trained_intent_inference_model_path",
        help="path to the saved state of the trained intent inference model")
    parser.add_argument("trained_slot_filling_model_path",
        help="path to the saved state of the trained slot filling model")
    parser.add_argument("language_model_path",
        help="path to the language model")
    parser.add_argument("wake_notification_wav_path",
        help="path to the wave file to play upon waking up")
    parser.add_argument("-uvs", "--use_voice_synthesis", action="store_true",
        help="whether or not to use the voice_synthesis.py to synthesize responses")

    parsed_args = parser.parse_args()

    synthesize_func = None
    if parsed_args.use_voice_synthesis:
        import voice_synthesizer
        synthesize_func = voice_synthesizer.synthesize

    Listener(parsed_args.trained_wake_word_model_path,
        parsed_args.trained_speech_recognition_model_path,
        parsed_args.trained_intent_inference_model_path,
        parsed_args.trained_slot_filling_model_path,
        parsed_args.language_model_path,
        parsed_args.wake_notification_wav_path,
        IntentHandler(), synthesize_func).start()
