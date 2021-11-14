#Kronos
A small personal virtual assistant, built and trained from scratch with the exception of using a pretrained BERT as the backbone of intent inference and slot filling. Additionally, IBM's Watson is optionally used to give voice to the responses.

## Table of Contents
<ol>
<li><a href="#about">About</a></li>
<li><a href="#overview">Overview</a></li>
<li><a href="#requirements">Requirements</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#language-model">Language model</a></li>
<li><a href="#technologies-used">Technologies used</a></li>
</ol>

## About
The idea of personal virtual assistant à la HAL, Jarvis and my personal favourite - Marvin has excited me for a long time, but the thought that my best bet is using an always listening and transmitting product by the tech giants to achieve that has been putting me off. So I wanted to build something small for myself, that could serve me as a proof of concept when I chat to my friends about how feasible it would be to each have our own completely offline version of an AI assistant.

Of course, Kronos in it's current form is a very pale imitation of it's FAANG forebears, but I hope the concept of a personal virtual assistant that you can safely say you **own** (*as in, all of it's data belongs entirely to you and no one else*) is as appealing to you as it is to me and Kronos inspires you to build your own.

## Overview
Kronos consists of three standalone modules that are combined together to form the basis of a virtual assistant. Those modules are included in this repo as submodules and they are:

- [Wake word detection](https://github.com/vshotarov/Kronos_wake-word-detection/)
- [Speech recognition](https://github.com/vshotarov/Kronos_speech-recognition/)
- [Intent and slot inference](https://github.com/vshotarov/Kronos_intent-and-slot-inference/)

The brief overview of the project is as follows:

On starting the app, the trained models for each of the above mentioned tasks are loaded and a stream to the mic is opened, so we start listening for inputs. Every half a second, the last second and a half of recording is ran through the wake word detection model to check if the wake word command has been uttered. If so, then we start recording for four seconds\* in order to capture the voice command. Once the four seconds are captured we pass them to the speech recognition module to get a text output, which is then passed to the intent and slot inference module to try and make sense of the command. 

If an intent has been recognized, it's then passed to the last piece of the puzzle - `intent_handler.py` - to perform the necessary actions. After that, we go back to listening for a wake word.

The `intent_handler.py` is where all the actions that we want the assistant to be able to perform are defined. E.g. checking what the time is, what the weather will be like, turning the lights off, etc. Those are heavily dependent on your specific use case and that's why it's been separated from the main system to be more easily customized.

\* I decided to use a hardcoded timespan for the voice command recording, as I thought for a proof of concept I don't need a voice activity detector, although, it's certainly in the specs for future versions.

For a more descriptive overview of each of the modules, please refer to the READMEs in their own repositories.

## Requirements
All the python requirements are stored in the `requirements.txt`, so you can install them using

```
pip install -r requirements.txt
```

with the caveat, that I use the [ctcdecode](https://github.com/parlance/ctcdecode) module, which is not part of pypi, so it has to be installed manually, but it's really easy following [the instructions](https://github.com/parlance/ctcdecode#installation).

## Usage
If you have all the trained modules, running Kronos is as simple as calling `python Kronos.py` with the relevant arguments.

```
usage: Kronos.py [-h] [-uvs]
                 trained_wake_word_model_path
                 trained_speech_recognition_model_path
                 trained_joint_intent_and_slot_model_path language_model_path
                 wake_notification_wav_path

Kronos virtual assistant

positional arguments:
  trained_wake_word_model_path
                        path to the saved state of the trained wake word
                        detection model
  trained_speech_recognition_model_path
                        path to the saved state of the trained speech
                        recognition model
  trained_joint_intent_and_slot_model_path
                        path to the saved state of the trained joined intent
                        inference and slot filling model
  language_model_path   path to the language model
  wake_notification_wav_path
                        path to the wave file to play upon waking up

optional arguments:
  -h, --help            show this help message and exit
  -uvs, --use_voice_synthesis
                        whether or not to use the voice_synthesis.py to
                        synthesize responses
```

Here's the exact command I use to run it

```
python Kronos.py /
	data/wake_word_model_state.torch /
	data/speech_model_state.torch /
	data/joint_intent_and_slot_model_state.torch /
	data/language_model.arpa /
	data/wake.wav /
	-uvs
```

As you can see, I am passing the saved state_dicts of all three trained modules (for how to train them refer to their own repos), as well as a language model ([here's how I generate it](#language-model)) and a `.wav` file to play any time the wake word has been recognized.

## Language Model
I've used [Kenneth Heafield](https://kheafield.com/)'s language model inference system for building a `.arpa` file that can be queried with the beam search, in order to give me the most likely interpretations of my voice commands.

To build the `.arpa` file, what I did was:

- compile kenlm following the instructions [here](https://github.com/kpu/kenlm#compiling)
- create a corpus file containing all of my speech samples (*I am still experimenting with the idea of using only the voice commands I would like to be understood as a corpus rather than ALL the samples I have recorded*), that has each sentence on a new line, similar to this
```
what will the weather be like tomorrow
whats the time in paris
set a timer for five minutes
set a timer for thirty seconds
...
```
- use the built `lmplz` binary to built the `.arpa` file like so:

```
bin/lmplz -o 5 --discount_fallback < corpus > language_model.arpa
```


## Technologies Used
I am lucky to be able to stand on the shoulder of giants and here's where I give a brief mention to some of them.

Here are the main python libraries I rely on:

- [PyTorch](https://pytorch.org/)
- PyAudio
- [ctcdecode](https://github.com/parlance/ctcdecode) 
- [transformers](https://github.com/huggingface/transformers)
- [kenlm](https://github.com/kpu/kenlm)

And here are the main machine learning algorithms I've relied on:

- [DeepSpeech](https://arxiv.org/abs/1512.02595) - Both my wake word and speech recognition modules are heavily based on it. They are basically much smaller and simplified versions of it, which I can get away with, since the whole idea of my personal virtual assistant is that it's *personal* and is only trained on my voice.
- [DistilBERT](https://arxiv.org/abs/1910.01108) - My joint intent and slot filling model is built as two extra heads on top of the [DistilBERT model provided by huggingface/transformers](https://huggingface.co/transformers/model_doc/distilbert.html), heavily based on [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909v1)

