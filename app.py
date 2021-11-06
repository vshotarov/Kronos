import argparse
from Kronos import Kronos

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

    parsed_args = parser.parse_args()

    Kronos(parsed_args.trained_wake_word_model_path,
        parsed_args.trained_speech_recognition_model_path,
        parsed_args.trained_intent_inference_model_path,
        parsed_args.trained_slot_filling_model_path,
        parsed_args.language_model_path,
        parsed_args.wake_notification_wav_path).start()
