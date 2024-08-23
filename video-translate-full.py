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

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str) 
parser.add_argument('--from_lang', type=str, nargs='?',  default='Chinese')
parser.add_argument('--to_lang', type=str,  nargs='?',  default='English')
parser.add_argument('--tts_lang', type=str,  nargs='?',  default='eng')
parser.add_argument('--whisper_lang', type=str,  nargs='?',  default='zh')


args = parser.parse_args()

#from_lang = 'zh-CN'
from_lang_word = args.from_lang #'Chinese'
#to_lang = 'en'
to_lang_word =  args.to_lang #'English'

tts_lang = args.tts_lang #'eng'  # nan

whisper_lang = args.whisper_lang  # zh  or en

if not os.path.exists(args.filename):
    os.mkdir(args.filename)

input = args.filename + '.mp4'
output = args.filename + '.wav'

# Insert Local Video File Path
clip = mp.VideoFileClip(input) 
# Insert Local Audio File Path
output = os.path.join(args.filename, output)
clip.audio.write_audiofile(output, codec='pcm_s16le')
 

audio = whisper.load_audio(output)
#tiny base small, medium, large large-v2 
model = whisper.load_model("medium", device="cpu")

#result = whisper.transcribe(model, audio,  vad=True, beam_size=5, best_of=5, temperature=(
#    0.0, 0.2, 0.4, 0.6, 0.8, 1.0), language="zh")

whisper_result = whisper.transcribe(
    model, audio,  vad=True, beam_size=5, best_of=5, language=whisper_lang)
print(json.dumps(whisper_result, indent=2, ensure_ascii=False))

whisper_result_json = json.dumps(whisper_result, indent=4)
# Writing to whisper_json_object.json
with open(os.path.join(args.filename, "whisper_result.json"), "w") as outfile:
    outfile.write(whisper_result_json)

actor = Actor()
maxHintCap = Caption()
captions = []
audioclips = AudioFileClip(output)
for segment in whisper_result["segments"]:
    cap = Caption()
    cap.order = segment["id"]
    cap.start = segment["start"]
    cap.end = segment["end"]
    cap.text = segment["text"]
    captions.append(cap)
    sub_clip = audioclips.subclip(cap.start, cap.end)
    sub_clip.write_audiofile(os.path.join(args.filename, "oaudio-" + str(cap.order) + ".wav")  )
    if cap.end - cap.start >= maxHintCap.end - maxHintCap.start:
        maxHintCap = cap


whisperfile = open(os.path.join(args.filename, "whisper_result.json"),)

whisper_result = json.load(whisperfile)

#do translate
captionlist = ""
count = 1
segments = whisper_result["segments"]
numsegments = len(segments)
for segment in segments:
   # captionlist += "#" + str(count) + ". " + segment["text"] + "\n"
    captionlist += segment["text"] + "\n"
    print(segment["text"])
    count += 1

promptRegroup = """
in the following content, some sentences may be splitted into several lines.
please reformat the  content, such that one whole sentence should only occupy one line
 
{captionlist}
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that understand language."},
        {"role": "user", "content": promptRegroup.format(
            captionlist=captionlist)}
    ],
    #  max_tokens=150,
    #n=1,
    #stop=None,
    temperature=1,
)
regroupstr = response.choices[0].message.content.strip()
print(regroupstr)
regroupstrlist = regroupstr.split("\n")

curpos = 0
numgroups = len(regroupstrlist)

print("numgroups = {numgroups} numsegments = {numsegments}".format(
    numgroups=numgroups, numsegments=numsegments))

regroupList = []
regroupstr_withorder = ""
for i in range(numgroups):
    agroupstr = regroupstrlist[i]
    regroupstr_withorder += str(i) + ". " + agroupstr + "\n"
    nextpos = -1
    if i == numgroups - 1:
        nextpos = numsegments
    else:
        nextgroupstr = regroupstrlist[i+1]
        for j in range(curpos, numsegments):
            if nextgroupstr.startswith(segments[j]["text"]):
                nextpos = j
                break
        if nextpos == -1:
            nextpos = numsegments
    if curpos >= numsegments:
        print("ERROR : curpos = " + str(curpos))
        break
    seg = segments[curpos]
    agroup = {"id": seg["id"], "start": seg["start"],
              "end": segments[nextpos-1]["end"], 
              "startpos" : curpos, "endpos" : nextpos-1,
              "text": agroupstr}
    regroupList.append(agroup)

    curpos = nextpos
""" merge audio for group ....
    if curpos + 1 < nextpos:
        clip = AudioFileClip(os.path.join(
            args.filename, "oaudio-" + str(seg["id"]) + ".mp3"))
        clips = [clip]
        for k in range(curpos+1, nextpos):
            agroup["end"] = segments[k]["end"] 
            aclip = AudioFileClip(os.path.join(
                args.filename, "oaudio-" + str(segments[k]["id"]) + ".mp3"))
            clips.append( aclip )
        
        merged_audio = concatenate_audioclips(clips)
        merged_audio.write_audiofile(os.path.join(args.filename, "temp001.mp3"))
        # copy src to dst
        shutil.copyfile(os.path.join(args.filename, "temp001.mp3"), os.path.join(
            args.filename, "oaudio-" + str(seg["id"]) + ".mp3"))
"""


prompt = """
please translate the following {numlines} {from_lang_word} sentences into {numlines} {to_lang_word} sentences. 
please translate each sentence one line at a time without combining any two lines. 
you should return {numlines} lines of translated sentences and only translated content will be returned.
  
{regroupstr_withorder}

"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that translates text."},
        {"role": "user", "content": prompt.format(numlines=str(len(regroupList)),
                                                  from_lang_word=from_lang_word, to_lang_word=to_lang_word, regroupstr_withorder=regroupstr_withorder)}
    ],
    #  max_tokens=150,
    #n=1,
    #stop=None,
    temperature=1,
)
translation = response.choices[0].message.content.strip()
print(translation)
translations = translation.split("\n")

print("translations size = " + str(len(translations)))
print("regroupList size = " + str(len(regroupList)))
if len(translations) != len(regroupList):
    print("ERROR!! translation and regroup does not match!! ")

finalcount = min(len(translations), len(regroupList))

for i in range(finalcount):
    regroupList[i]["translated"] = StringUtils.stripLeadingNumberOrder(
        translations[i]) 
 

#save to disk for UI use
regroupList_json = json.dumps(regroupList, indent=4)
# Writing to whisper_json_object.json
regroupList_json_file = os.path.join(args.filename, "regroupList.json")
with open(regroupList_json_file, "w") as outfile:
    outfile.write(regroupList_json)

""" """
#load from disk,  just for testing,  file may be modified by UI
regroupList = json.load(open(regroupList_json_file, ))


finalcount = len(regroupList)

maxHintCap = Caption()
captions = []


resultaudios = []
for i in range(finalcount):
    agroup = regroupList[i]
    """ 
    aclips = AudioFileClip(os.path.join(
        args.filename, "oaudio-" + str(agroup["id"]) + ".wav"))
    merged_audio = concatenate_audioclips([aclips, hintclips])
    # Save the merged audio to a new file
    hintfile = os.path.join(args.filename, "hintcopy.wav")
    if os.path.isfile(hintfile):
        os.remove(hintfile)
    merged_audio.write_audiofile(hintfile)
    """
    translated = agroup.get("translated", "")
    if translated == "":
        print("ERROR translated is empty for " + agroup["text"])
    print(translated)
     
    hintclips = []
    for i in range(agroup["startpos"], agroup["endpos"]+1):
        hintclips.append(AudioFileClip(os.path.join(
            args.filename, "oaudio-" + str(i+1) + ".wav")))

    merged_audio = concatenate_audioclips(hintclips)
    # Save the merged audio to a new file
    hintfile = os.path.join(args.filename, "hintcopy.wav")
    if os.path.isfile(hintfile):
        os.remove(hintfile)
    merged_audio.write_audiofile(hintfile)

 
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
         translated ,
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
