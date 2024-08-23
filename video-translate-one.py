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
import shutil
import moviepy.editor as mp
#from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import concatenate_audioclips, AudioFileClip

from CaptionComponent import Caption
from CaptionComponent import Actor
from StringUtils import StringUtils

import whisper_timestamped as whisper
from difflib import SequenceMatcher

from TTS.api import TTS
#import googletrans
#from googletrans import Translator
import openai
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str) 
parser.add_argument('--from_lang', type=str, nargs='?',  default='Chinese')
parser.add_argument('--to_lang', type=str,  nargs='?',  default='English')
#parser.add_argument('--tts_lang', type=str,  nargs='?',  default='en')
parser.add_argument('--whisper_lang', type=str,  nargs='?',  default='zh')
parser.add_argument('--sample_voice', type=str,  nargs='?',  default='')
parser.add_argument('--skip_whisper', action=argparse.BooleanOptionalAction) 
parser.add_argument('--no_regroup', action=argparse.BooleanOptionalAction)
parser.add_argument('--skip_chatgpt_regroup', action=argparse.BooleanOptionalAction)
parser.add_argument('--tts_provider', type=str,  nargs='?',  default='ms')
parser.add_argument('--skip_adjust_speed', action=argparse.BooleanOptionalAction)
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

if not os.path.exists(args.filename):
    os.mkdir(args.filename)

frameimagesdir = os.path.join(args.filename, "frameimgs")
if not os.path.exists(frameimagesdir):
    os.mkdir(frameimagesdir)

#move sample_voice into project dir
if sample_voice:
    sample_voice_dir = os.path.join(args.filename, sample_voice)
    if not os.path.exists(sample_voice_dir):
        shutil.copyfile(sample_voice, sample_voice_dir) 

input = args.filename + '.mp4'
output = args.filename + '.wav'


# Insert Local Video File Path
clip = mp.VideoFileClip(input) 
# Insert Local Audio File Path
output = os.path.join(args.filename, output)
clip.audio.write_audiofile(output, codec='pcm_s16le')

#used to skip whisper code -- start
if not args.skip_whisper:
    audio = whisper.load_audio(output)
    #tiny base small, medium, large large-v2 
    model = whisper.load_model("small", device="cpu")

    #result = whisper.transcribe(model, audio,  vad=True, beam_size=5, best_of=5, temperature=(
    #    0.0, 0.2, 0.4, 0.6, 0.8, 1.0), language="zh")

    whisper_result = whisper.transcribe(
        model, audio,  vad=True, beam_size=5, best_of=5, language=whisper_lang)
    print(json.dumps(whisper_result, indent=2, ensure_ascii=False))

    whisper_result_json = json.dumps(whisper_result, indent=4)
    # Writing to whisper_json_object.json
    with open(os.path.join(args.filename, "whisper_result.json"), "w") as outfile:
        outfile.write(whisper_result_json)

#used to skip whisper code -- end

whisperfile = open(os.path.join(args.filename, "whisper_result.json"),) 
whisper_result = json.load(whisperfile)
  

#audioclips = AudioFileClip(output)
for segment in whisper_result["segments"]:
    cap = Caption()
    cap.order = segment["id"]
    cap.start = segment["start"]
    cap.end = segment["end"]
    cap.text = segment["text"]
    frametime = cap.start+1 if cap.start + 1 <= cap.end else cap.start+0.2
    clip.save_frame(os.path.join(
        frameimagesdir, str(cap.order) + ".png"), t=frametime)

#audioclips.close()

""" 
words: [  {
          "text": "Do",
          "start": 3.01,
          "end": 3.15,
          "confidence": 0.742
        }, 
"""
def getWordByPos(pos):
    curpos = 0
    for segment in whisper_result["segments"]:
        words = segment["words"]
        if curpos + len(words) < pos + 1:
            curpos += len(words)
            continue
        return words[pos-curpos]
    return None


def countMismatch(source, target):
    mismatch = 0
    for i in range(len(source)):
       if source[i] != target[i]:
           mismatch += 1
    return mismatch


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#do translate
captionlist = ""
count = 0
segments = whisper_result["segments"]
numsegments = len(segments)

print("text from whisper_result:" + str(numsegments))

for segment in segments:
   # captionlist += "#" + str(count) + ". " + segment["text"] + "\n"
    captionlist += segment["text"] + "\n"
    print(segment["text"])
    count += 1

print("done with ---- text from whisper_result:")

example = """
I eat
breakfast at home
It is a nice day today

should be:

I eat breakfast at home
It is a nice day today
"""

if whisper_lang == 'zh':
    example = """
我在家
里吃了早饭
今天的天气很好

should be:

我在家里吃了早饭
今天的天气很好
"""

promptRegroup = """
In the following content, some sentences have been broken into multiple lines. 
Please merge the sentences that are clearly broken into multiple lines into a single line. 
In the final returned content, each short sentence should occupy its own line, with each line not exceeding 50 words. 
Please avoid forming overly long compound sentences or paragraphs.

for example: 

{example}


### 
{captionlist}

###
Please don't insert/delete/change any word.
Please don't return a single paragraph.
"""

if args.no_regroup:
    count = 1
    regroupstr = ""
    for segment in segments:
       regroupstr +=  segment["text"] + "\n"
       count += 1
    with open(os.path.join(args.filename, "regroupstr_fromchatgpt.txt"), "w") as outfile:
        outfile.write(regroupstr)

elif not args.skip_chatgpt_regroup:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that understand language."},
            {"role": "user", "content": promptRegroup.format(
                captionlist=captionlist, example=example)}
        ],
        #  max_tokens=150,
        #n=1,
        #stop=None,
        temperature=0.1,
    )
    regroupstr = response.choices[0].message.content.strip()
    print("text from regroupstr:")
    print(regroupstr)

    with open(os.path.join(args.filename, "regroupstr_fromchatgpt.txt"), "w") as outfile:
        outfile.write(regroupstr) 

 
with open(os.path.join(args.filename, "regroupstr_fromchatgpt.txt"),) as file:
    regroupstr = file.read()
 
regroupstrlist = regroupstr.split("\n")
if len(regroupstrlist) == 1:
    regroupstrlist = regroupstr.split("。")
if len(regroupstrlist) == 1:
    regroupstrlist = regroupstr.split(".")


print("numgroups = {numgroups} numsegments = {numsegments}".format(
    numgroups=len(regroupstrlist), numsegments=numsegments))

regroupstrlist = list(filter(lambda x: len(x) > 0, regroupstrlist))

numgroups = len(regroupstrlist)

print("numgroups = {numgroups} numsegments = {numsegments}".format(
    numgroups=numgroups, numsegments=numsegments))

def filterTokens(input):
    return input.replace(
        "，", "").replace(",", "").replace(".", "").replace("。", "").replace(" ", "").replace("?", "").replace("？", "").replace(":", "").replace("：", "").replace("！", "").replace("!", "")

regroupList = []
regroupstr_withorder = ""

curpos_in_segments = 0
whisper_text = whisper_result["text"].strip()
if whisper_lang == 'zh':
    curpos_in_one_segment = 0
    for i in range(numgroups):
        agroupstr = regroupstrlist[i]
        regroupstr_withorder += str(i) + ". " + agroupstr + "\n"

        agroupstr_s = filterTokens(agroupstr )
        print("\ni = " + str(i) + " " + agroupstr_s)
        agroupstr_acc = ""
        #mismatch_count = 0
        for j in range(curpos_in_segments, numsegments):
            asegment = segments[j]
            asegment_words = asegment["words"]
            found = -1
            start_pos_in_one_segment = curpos_in_one_segment if j == curpos_in_segments else 0
            for k in range(start_pos_in_one_segment, len(asegment_words)):
                aword = asegment["words"][k]
                aword_w = filterTokens( aword["text"] )
                agroupstr_acc += aword_w
                if len(agroupstr_s) == len(agroupstr_acc):
                    mismatch_count = countMismatch(agroupstr_s, agroupstr_acc)
                    mis_rate = mismatch_count * 1.0 / len(agroupstr_s)
                    print( "miscount = " + str(mismatch_count) + " mis-rate = " +
                           str(mis_rate) + " acc_str = " + agroupstr_acc)
                    if mis_rate < 0.1 :
                        found = k
                        break
              
            #print("found=" + str(found) + " " + agroupstr_acc)
            #break
            if found == -1:
                len0 = len(agroupstr_s)
                len1 = len(agroupstr_acc)
                if len0 > 10 and len1 > 10 and abs(len0-len1)/len0 < 0.2 and agroupstr_s[-5:] == agroupstr_acc[-5:] and similar(agroupstr_s, agroupstr_acc) >=0.8:
                    found = len(asegment_words)-1

                                    

            if found >= 0:
                seg_s = segments[curpos_in_segments]
                seg_e = asegment
                agroup = {"id": i, "start": seg_s["words"][curpos_in_one_segment]["start"],
                          "end": seg_e["words"][found]["end"], 
                          "text": agroupstr}
                regroupList.append(agroup)

                if found == len(asegment_words)-1:
                    curpos_in_segments = j+1
                    curpos_in_one_segment = 0
                else:
                    curpos_in_segments = j 
                    curpos_in_one_segment  = found + 1 
                break  
            #else:
            #    curpos_in_segments += 1
else:
    for i in range(numgroups):
        agroupstr = regroupstrlist[i].strip()
        regroupstr_withorder += str(i) + ". " + agroupstr + "\n"
        try:
            pos = whisper_text.index(agroupstr, curpos_in_segments)
            tokencount = len(whisper_result["text"][0:pos].strip().split(" "))
            start = getWordByPos(tokencount)
            grouptokencount = len( agroupstr.split(" "))
            end = getWordByPos(tokencount + grouptokencount-1)
            print( start["text"] + " " + end["text"] )
            agroup = {"id": i, "start": start["start"], 
                    "speakerId": -1,
                    "end": end["end"], 
                    "text": agroupstr}
            regroupList.append(agroup)

            curpos_in_segments += len(agroupstr)
        except Exception as e: 
            print("ERROR!! group not found for " + str(e)  ) 
            print("ERROR!! group not found for " + agroupstr) 
    

#save to disk for UI use
regroupList_json = json.dumps(regroupList, indent=4)
# Writing to whisper_json_object.json
regroupList_json_file = os.path.join(args.filename, "regroupList0.json")
with open(regroupList_json_file, "w") as outfile:
    outfile.write(regroupList_json)

regroupstr_withorder_file = os.path.join(
    args.filename, "regroupstr_withorder.txt")
with open(regroupstr_withorder_file, "w") as outfile:
    outfile.write(regroupstr_withorder)

 