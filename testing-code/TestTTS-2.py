#from scipy import spatial  # for calculating vector similarities for search
from moviepy.editor import *
#import spacy
#from pyparsing import List
#import tiktoken  # for counting tokens
#import pandas as pd  # for storing text and embeddings data
#import openai  # for calling the OpenAI API
#import ast  # for converting embeddings saved as strings back to arrays
#import operator
#from nltk.sentiment import SentimentIntensityAnalyzer
#import nltk
#import pandas as pd
#from nudenet import NudeClassifier
#from os import listdir
#from ultralytics import YOLO
import os
import json
import argparse
import time
import moviepy.editor as mp
#from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import concatenate_audioclips, AudioFileClip

from CaptionComponent import Caption
from CaptionComponent import Actor
from StringUtils import StringUtils

import whisper_timestamped as whisper
from TTS.api import TTS
#import googletrans
#from googletrans import Translator
import openai
import shutil

openai.api_key = os.environ["OPENAI_API_KEY"]

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()

if not os.path.exists(args.filename):
    os.mkdir(args.filename)

input = args.filename + '.mp4'
output = args.filename + '.wav'

#translator = Translator(service_urls=['translate.googleapis.com'])
# you will speak
#from_lang = 'zh-CN'
from_lang_word = 'Chinese'
#to_lang = 'en'
to_lang_word = 'English'

tts_lang = 'eng'  # hak
  
# Writing to whisper_json_object.json
regroupList_json_file = os.path.join(args.filename, "regroupList.json")
 
#load from disk,  just for testing,  file may be modified by UI
regroupList = json.load(open(regroupList_json_file, ))

finalcount =  len(regroupList) 

maxHintCap = Caption()
captions = []
hintclips = AudioFileClip(os.path.join(
    args.filename, "oaudio-" + str(1) + ".wav"))

resultaudios = []
for i in range(finalcount):
    agroup = regroupList[i]
    aclips = AudioFileClip(os.path.join(
        args.filename, "oaudio-" + str(agroup["id"]) + ".wav"))
    merged_audio = concatenate_audioclips([aclips, hintclips])
    # Save the merged audio to a new file
    hintfile = os.path.join(args.filename, "hintcopy.wav")
    if os.path.isfile(hintfile):
        os.remove(hintfile)
    merged_audio.write_audiofile(hintfile)
    print(agroup["translated"])
   # text_translated = translator.translate(agroup["text"],
   #                                                  src= from_lang,
   #                                                  dest= to_lang)
    # tts languages
    # hak   Chinese, Hakka
    # nan   Chinese, Min Nan
    # eng   English
    # TTS with on the fly voice conversion
    api = TTS("tts_models/" + tts_lang + "/fairseq/vits", gpu=False)

    resultaudiofile = os.path.join(args.filename, "ttsaudio-" +
                                   str(agroup["id"]) + ".wav")
    api.tts_with_vc_to_file(
        agroup["translated"],
        speaker_wav=os.path.join(args.filename, "hintcopy.wav"),
        file_path=resultaudiofile
    )

    resultaudio = AudioFileClip(resultaudiofile)

    resultaudio = resultaudio.set_start(agroup["start"])
    print(" group start = " + str(agroup["start"]))
    resultaudios.append(resultaudio)

    """
    api.tts_to_file(text=agroup["translated"], file_path=os.path.join(args.filename, "ttsaudio-" +
                                                                      str(agroup["id"] ) + ".wav"),
                emotion="Happy", speed=1.5)
    """

videoclip = mp.VideoFileClip(input)
videoclipfinal = videoclip.set_audio(CompositeAudioClip(resultaudios))
videoclipfinal.write_videofile(os.path.join(args.filename, "finalvideo.mp4"))
