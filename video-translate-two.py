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
import json5

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str) 
parser.add_argument('--from_lang', type=str, nargs='?',  default='Chinese')
parser.add_argument('--to_lang', type=str,  nargs='?',  default='English')
#parser.add_argument('--tts_lang', type=str,  nargs='?',  default='en')
parser.add_argument('--whisper_lang', type=str,  nargs='?',  default='zh')
parser.add_argument('--sample_voice', type=str,  nargs='?',  default='')
parser.add_argument('--skip_whisper', action=argparse.BooleanOptionalAction)
parser.add_argument('--tts_provider', type=str,  nargs='?',  default='ms')
parser.add_argument('--skip_adjust_speed',
                    action=argparse.BooleanOptionalAction)
# tts_provider  support ms (microsoft), coqui( coqui-ai )

#language whisper vits_tts
#chinese  zh  nan
#English  en    eng   ( en  because use your_tts)
#French   fr    fra   ( fr-fr because use your_tts
#Portuguese     por   (  pt-br because use your_tts )
#german   "de": "deu"
#"spanish" "es":  spa,
#"korean"  "ko": , kor
#"japanese" "ja": , jpn

args = parser.parse_args()

#from_lang = 'zh-CN'
from_lang_word = args.from_lang #'Chinese'
#to_lang = 'en'
to_lang_word =  args.to_lang #'English'

#tts_lang = args.tts_lang #'eng'  # nan

whisper_lang = args.whisper_lang  # zh  or en
sample_voice = args.sample_voice
 

regroupList = [] 
regroupList_json_file = os.path.join(args.filename, "regroupList0.json")  
#load from disk,  just for testing,  file may be modified by UI
regroupList = json.load(open(regroupList_json_file, ))

regroupstr_withorder_file = os.path.join(
    args.filename, "regroupstr_withorder.txt")
with open(regroupstr_withorder_file) as file:
    regroupstr_withorder = file.read()

prompt = """ 
Please translate the following {numlines}  sentences from {from_lang_word} into {to_lang_word} . Each sentence stands alone; please do not merge adjacent sentences.
### 
{regroupstr_withorder}

"""

prompt_json = """
###
This is a sample of  returned JSON format:
{"data":[{ "id" : 0, "original" : "",  "translated": ""} ,{ "id" : 1, "original" : "",  "translated": ""},{ "id" : 2, "original" : "",  "translated": ""},{ "id" : 3, "original" : "",  "translated": ""} ]}

"""

prompt_example = ""
if from_lang_word == "Chinese":
    prompt_example = """
###
This is an example:
{"data":[{ "id" : 0, "original" : "我们引入了更强大的这个动画开源库",  "translated": "we have introduced a more powerful open-source animation library"}  ]}
"""
else:
    prompt_example = """
###
This is an example:
{"data":[{ "id" : 0, "original" : "we have introduced a more powerful open-source animation library",  "translated": "我们引入了一个更强大的动画开源库"}  ]}
"""

pcontent = prompt.format(numlines=str(len(regroupList)),
                         from_lang_word=from_lang_word, to_lang_word=to_lang_word, regroupstr_withorder=regroupstr_withorder) + "\n" + prompt_json + "\n" + prompt_example

pcontent += "\n###\n please make sure you have all sentences translated"
print( pcontent )

 

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "system", "content": "You are an language expert that only speaks JSON. Do not write normal text."},
        {"role": "user", "content": pcontent}
    ],
    #  max_tokens=150,
    #n=1,
    #stop=None,
    temperature=0.1,
)
translation = response.choices[0].message.content.strip()
print(" translation result: ")
print(translation)
translations =  list(map(lambda x: x['translated'], json5.loads(translation)["data"]))
#translations = translation.split("\n")
print(" translation result2: ")
print(translations)

print("translations size = " + str(len(translations)))
print("regroupList size = " + str(len(regroupList)))
if len(translations) != len(regroupList):
    print("ERROR!! translation and regroup does not match!! ")

finalcount = min(len(translations), len(regroupList))

for i in range(finalcount):
    regroupList[i]["translated"] = StringUtils.stripLeadingNumberOrder(
        translations[i]) 
 
project = {
    "name": args.filename,
   	"fromLang": args.from_lang,
    "toLang": args.to_lang,
    "useVoiceCloning": args.sample_voice != "",
 
    "whisperLang": args.whisper_lang,
    "sampleVoice": args.sample_voice,
    "captions": regroupList,
    "speakers": []
}

#save to disk for UI use
project_json = json.dumps(project, indent=4)
# Writing to whisper_json_object.json
project_json_file = os.path.join(args.filename, "project.json")
with open(project_json_file, "w") as outfile:
    outfile.write(project_json)
  