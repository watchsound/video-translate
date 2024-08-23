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
#from gtts import gTTS
import soundfile as sf 
#import pyrubberband as pyrb
#from pydub import AudioSegment

import librosa

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str) 
parser.add_argument('--from_lang', type=str, nargs='?',  default='Chinese')
parser.add_argument('--to_lang', type=str,  nargs='?',  default='English')
#parser.add_argument('--tts_lang', type=str,  nargs='?',  default='en')
parser.add_argument('--whisper_lang', type=str,  nargs='?',  default='zh')
parser.add_argument('--sample_voice', type=str,  nargs='?',  default='')
parser.add_argument('--speaker_sex', type=str,  nargs='?',  default='male') #or female
parser.add_argument('--skip_whisper', action=argparse.BooleanOptionalAction)
parser.add_argument('--tts_provider', type=str,  nargs='?',  default='ms')
parser.add_argument('--adjust_speed',
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
tts_lang = "eng" 
if to_lang_word == "Chinese":
    tts_lang = "nan"
elif to_lang_word == "French":
    tts_lang = "fra"
elif to_lang_word == "Portuguese":
    tts_lang = "por"

whisper_lang = args.whisper_lang  # zh  or en
sample_voice = args.sample_voice
speaker_sex = args.speaker_sex

tts_provider = args.tts_provider
 
input = args.filename + '.mp4'  
 
"""
project = {
    "name": args.filename,
   	"fromLang": args.from_lang,
    "toLang": args.to_lang, 
    "whisperLang": args.whisper_lang,
    "sampleVoice": args.sample_voice,
    "captions": regroupList,
    "speaker" : 
    "speakers": []
}
"""  
 
project_json_file = os.path.join(args.filename, "project.json") 
#load from disk,  just for testing,  file may be modified by UI
project = json.load(open(project_json_file, ))
 
if project.get("fromLang"):
    from_lang_word = project.get("fromLang")
if project.get("toLang") :
    to_lang_word = project.get("toLang")
#if project.get("ttsLang"):
#    tts_lang = project.get("ttsLang")
if project.get("whisperLang"):
    whisper_lang = project.get("whisperLang") 
if project.get("sampleVoice"):
    sample_voice = project.get("sampleVoice")
 
speaker_sex = args.speaker_sex
speakers = project.get("speakers") if project.get(
    "speakers") else []
 

default_speaker = project.get("speaker", None)
regroupList = project.get("captions")  
useVoiceCloning = project.get("useVoiceCloning", False)
finalcount = len(regroupList)

captions = [] 
resultaudios = []
your_tts = None
vits_tts = None
base_tts = None

def getSpeaker(sid):
    for speaker in speakers:
        if speaker["id"] == sid:
            return speaker
    return None 
 
def isBlankSpeaker(speaker):
    if speaker is None:
        return True
    vt = speaker.get("voiceTTS", None)
    audiofile = speaker.get("audiofile", None)
    return (vt is None or vt == '') and (audiofile is None or audiofile == '')
     
"""
if run face-detection and not modify code in GUI, we need to pass face-detection result into project.
(if run GUI already, this task is done by GUI)
"""
if useVoiceCloning is False and  len(speakers) > 0:
    if to_lang_word == 'Chinese':
        females = ["zh-CN-XiaoxiaoNeural",  "zh-CN-XiaoyiNeural",  "zh-CN-XiaochenNeural",
                   "zh-CN-XiaohanNeural",  "zh-CN-XiaomengNeural", "zh-CN-XiaomoNeural", "zh-CN-XiaoxuanNeural",
                   "zh-CN-XiaoqiuNeural",
                   "zh-CN-XiaoruiNeural", "zh-CN-XiaoyanNeural", "zh-CN-XiaoshuangNeural",  'zh-CN-YunxiaNeural', ]

        males = ['zh-CN-YunfengNeural', 'zh-CN-YunjianNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural',
                 'zh-CN-YunyeNeural', 'zh-CN-YunzeNeural',   'zh-CN-YunhaoNeural']
    elif to_lang_word == 'Japan':
        females = [
            "ja-JP-NanamiNeural",
            "ja-JP-AoiNeural",
            "ja-JP-MayuNeural",
        ]
        males = [
            "ja-JP-KeitaNeural",
            "ja-JP-DaichiNeural",
            "ja-JP-NaokiNeural",
        ]
    elif to_lang_word == 'Franch':
        females = [
            "fr-FR-DeniseNeural",
            "fr-FR-BrigitteNeural",
            "fr-FR-CelesteNeural",
            "fr-FR-CoralieNeural",
            "fr-FR-EloiseNeural",
            "fr-FR-JacquelineNeural",
            "fr-FR-JosephineNeural",
            "fr-FR-YvetteNeural",
        ]
        males = [
            "fr-FR-HenriNeural",
            "fr-FR-AlainNeural",
            "fr-FR-ClaudeNeural",
            "fr-FR-JeromeNeural",
            "fr-FR-MauriceNeural",
            "fr-FR-YvesNeural",
        ]
    elif to_lang_word == 'English':
        females = [
            "en-US-JennyNeural",
            "en-US-AriaNeural",
            "en-US-AmberNeural",
            "en-US-AnaNeural",
            "en-US-AshleyNeural",
            "en-US-CoraNeural",
            "en-US-ElizabethNeural",
            "en-US-JaneNeural",
            "en-US-MichelleNeural",
            "en-US-MonicaNeural",
            "en-US-NancyNeural",
            "en-US-SaraNeural",
            "en-US-AIGenerateNeural",
            "en-US-BlueNeural",
            "en-US-JennyMultilingualVNeural",
        ]
        males = [
            "en-US-GuyNeural",
            "en-US-DavisNeural",
            "en-US-BrandonNeural",
            "en-US-ChristopherNeural",
            "en-US-EricNeural",
            "en-US-JacobNeural",
            "en-US-JasonNeural",
            "en-US-RogerNeural",
            "en-US-SteffanNeural",
            "en-US-TonyNeural",
            "en-US-AIGenerateNeural",
            "en-US-BlueNeural",
            "en-US-RyanMultilingualNeural",
        ]
    femalecount, malecount = 0,0
    newspeakers = []
    for aspeaker in speakers:
        if isBlankSpeaker(aspeaker) is False:
            newspeakers.append( aspeaker )
            continue
        voicename = ""
        if aspeaker["gender"] == "Male":
            voicename = males[malecount]
            malecount += 1
        else:
            voicename = females[femalecount]
            femalecount += 1
        newspeakers.append({
            "id": aspeaker["id"],
            "name": voicename,
            "icon":  str(aspeaker["id"]) + ".png",
            "gender":  aspeaker["gender"] == "Male",
            "langCode": tts_lang,
            "fromFaceUI":  True,
            "voiceTTS": {
                "source": "ms",
                "voiceName": voicename
            },
            "note": ""
          })
    speakers = newspeakers
""" 
    for i in range(finalcount):
        agroup = regroupList[i]
        aspeaker = agroup.get("speakerId", None)
        if aspeaker is not None:
            agroup['speaker'] = aspeaker['id']
            agroup['emotion'] = aspeaker['emotion']
"""

"""
check default speaker
"""
if isBlankSpeaker(default_speaker):
    if useVoiceCloning: 
        if sample_voice != '':
            default_speaker = {
                "id": 0,
                "name": "default",
                "gender":  "",
                "langCode": tts_lang,
                "audiofile":  sample_voice,
                "note": ""
            }
        elif len(speakers) > 0:
            default_speaker = speakers[0]
    else:
        if len(speakers) > 0:
            default_speaker = speakers[0] 
        else:
            if to_lang_word == 'Chinese':
                voicename =  "zh-CN-XiaoxiaoNeural" 
            elif to_lang_word == 'Japan':
                voicename = "ja-JP-NanamiNeural" 
            elif to_lang_word == 'Franch':
                voicename = "fr-FR-DeniseNeural" 
            elif to_lang_word == 'English':
                voicename = "en-US-JennyNeural" 
            default_speaker = {
                "id": 0,
                "name": "default",
                "gender":  "",
                "langCode": tts_lang,
                "voiceTTS": {
                    "source": "ms",
                    "voiceName": voicename
                },
            }


def speak_ms_tts_to_file(content, lang, voice, emotion, styledegree, speedRate, speech_synthesizer):
    if abs(speedRate -1) > 0.05:
        text = """
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang}">
            <voice name="{voice}">
                <prosody rate="{speedRate}">
                    {content}
                </prosody>
            </voice>
        </speak>
        """
        speech_synthesis_result = speech_synthesizer.speak_ssml_async(
            ssml=text.format(lang=lang, voice=voice,  speedRate=speedRate, content=content)).get()

    else:
        text = """
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
            xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{lang}">
            <voice name="{voice}"> 
                <mstts:express-as    style="{emotion}" styledegree="{styledegree}">
                {content}
                </mstts:express-as> 
            </voice>
        </speak>
        """ 
        speech_synthesis_result = speech_synthesizer.speak_ssml_async(
            ssml=text.format(lang=lang, voice=voice, emotion=emotion, styledegree=styledegree, content=content)).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("SynthesizingAudioCompleted result")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(
                    cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


if project.get("useVoiceCloning", False) is False: #tts_provider == 'ms':
    import azure.cognitiveservices.speech as speechsdk
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('COGNITIVE_SERVICE_KEY'), region="westus")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    

    emotion_mapping_to_ms = {
        "angery": "angery",
        "fear": "fearful",
        "neutral": "calm",
        "sad": "sad",
        "disgust": "disgruntled",
        "happy": "cheerful",
        "surprice": "cheerful"
    }
    for i in range(finalcount):
        agroup = regroupList[i]
        translated = agroup.get("translated", "")
        resultaudiofile = os.path.join(args.filename, "ttsaudio-" +
                                       str(i) + ".wav")

        file_config = speechsdk.audio.AudioOutputConfig(
            filename=resultaudiofile) 
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=file_config)
        
        aspeaker = getSpeaker(agroup.get("speakerId", None))
        if aspeaker is None:
            aspeaker = default_speaker
        
        voice  = aspeaker.get("voiceTTS")["voiceName"]
         
        emotion = agroup.get("emotionTTS", '')
        if emotion == '':
            emotion = agroup.get("emotion", "calm")
            emotion = emotion_mapping_to_ms.get(emotion, emotion)
        styledegree = agroup.get("styleDegree", 1) 
        speedRate = agroup.get("speedRate", 1)
        ## FIXME -- emotion does not work well, it affect voice quality
        #emotion = "calm"
        print(" {translated}, ||||| voice : {voice},  emotion: {emotion} ".format(
            translated=translated, voice=voice, emotion=emotion))
        speech_config.speech_synthesis_voice_name = voice # 'zh-CN-XiaomoNeural'
        lang = voice[0:5]
        speak_ms_tts_to_file(translated, lang,  voice, emotion, styledegree, speedRate, speech_synthesizer)

        resultaudio = AudioFileClip(resultaudiofile)

        resultaudio = resultaudio.set_start(agroup["start"])
        print(" group start = " + str(agroup["start"]))
        resultaudios.append(resultaudio)
else:
    for i in range(finalcount):
        agroup = regroupList[i]
        translated = agroup.get("translated", "")
        if translated == "":
            print("ERROR translated is empty for " + agroup["text"])
        print(translated)

        speakerid = agroup.get("speakerId", None)
        speaker = getSpeaker(
            speakerid) if speakerid is not None else default_speaker
        
        if speaker is not None and speaker.audiofile:
            hintfile = os.path.join(args.filename, speaker.audiofile)   
        else:
            hintfile = sample_voice

        resultaudiofile = os.path.join(args.filename, "ttsaudio-" +
                                        str(i) + ".wav")
        """
        if os.path.exists(resultaudiofile):
            resultaudio = AudioFileClip(resultaudiofile)
            resultaudio = resultaudio.set_start(agroup["start"])
            print(" group start = " + str(agroup["start"]))
            resultaudios.append(resultaudio)
            continue
        """
        if to_lang_word == "Chinese":
            #use google tts (free but not good)
            #tts = gTTS(translated, lang='zh-CN')
            #tts.save(resultaudiofile)

            if base_tts is None:
                tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST")
            tts.tts_to_file(text=translated, file_path=resultaudiofile)
        elif not hintfile:
            print("use base_tts.................")
            if base_tts is None:
                model_name = TTS.list_models()[0] 
                base_tts = TTS(model_name)

            speakerrole = base_tts.speakers[0]
            sex = speaker_sex
            if speaker is not None:
                sex = "male" if speaker["male"] else "female"

            for s in base_tts.speakers:
                print("a speaker is = " + s)
                if s.startswith(sex):
                    speakerrole = s
                    break;

            try:
                base_tts.tts_to_file(
                    text=translated, speaker=speakerrole, language=tts_lang, file_path=resultaudiofile)
            except Exception as error:
                print("base_tts Error for " + translated, error)
                continue
        elif to_lang_word in ["English", "French", "Portuguese"]:
            """
            # Example voice cloning with YourTTS in English, French and Portuguese 
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
            tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
            tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
            tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
            """ 
            print("use your_tts.................")
            if your_tts is None:
                your_tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts",
                    progress_bar=False, gpu=False)
        
            try:
                your_tts.tts_to_file(
                    translated,
                    speaker_wav=hintfile,
                    language=tts_lang,
                    file_path=resultaudiofile
                )
            except Exception as error: 
                print("you_tts Error for " + translated, error)
                continue
        else:
            # text_translated = translator.translate(agroup["text"],
            #                                                  src= from_lang,
            #                                                  dest= to_lang)
            # tts languages
            # hak   Chinese, Hakka
            # nan   Chinese, Min Nan
            # eng   English
            # TTS with on the fly voice conversion
            print("use vits_tts.................")
            if vits_tts is None:
                vits_tts = TTS("tts_models/" + tts_lang + "/fairseq/vits", gpu=False)
    
            try:
                vits_tts.tts_with_vc_to_file(
                    translated,
                    speaker_wav=hintfile,
                    file_path=resultaudiofile
                )
            except Exception as error:
                print("vits_tts Error for " + translated, error)
                continue

        resultaudio = AudioFileClip(resultaudiofile)

        resultaudio = resultaudio.set_start(agroup["start"])
        print(" group start = " + str(agroup["start"]))
        resultaudios.append(resultaudio)

#adjust speed /duration for each audio clips
adjustedaudios = []
if  not args.adjust_speed:
    adjustedaudios = resultaudios
else:
    for index in range(0, len(resultaudios)):
        aclip = resultaudios[index]
        if index == len(resultaudios) - 1:
            adjustedaudios.append(aclip)
            break
        nextclip = resultaudios[index+1]

        numframes = 0
        for frame in aclip.iter_frames():
            numframes += 1
        audio_duration = numframes * 1.0 / 44100
        audio_start = regroupList[index]["start"]
        next_start = regroupList[index+1]["start"] 
        print( " {index} : {audio_start} : {audio_duration} -> {next_start}".format(
            index=index, audio_start=audio_start, audio_duration=audio_duration, next_start=next_start
        )   )
        if audio_duration + audio_start <= next_start:
            adjustedaudios.append(aclip)
            continue
        print( "adjust duration for " + str(index) )
        #adjduration = max(1, next_start - audio_start - 0.1)
        adjduration =  next_start - audio_start   
        afactor = audio_duration / adjduration 
    
        resultaudiofile = os.path.join(args.filename, "ttsaudio-" +
                                    str(index) + ".wav")
        adjustedfile = os.path.join(
            args.filename,  "adjust-audio-" + str(index) + "-hint.wav")
        
        # pyrb can not installed correct for version conflict
        y, sr =   sf.read(resultaudiofile)
        y_stretch = librosa.effects.time_stretch(y,  rate=afactor)
    #  y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=afactor)
        sf.write(adjustedfile, y_stretch, sr, format='wav')

        """ 
        # have no effect ...
        audio = AudioSegment.from_file(resultaudiofile, format="wav")  # wav
        audio.speedup(playback_speed=afactor)   
        audio.export(adjustedfile, format="wav")
        """

        """
        # use moviepy directly without additional libs.....having pitch problem
        aclip = aclip.fl_time(lambda t: afactor*t, apply_to=['mask', 'audio'])
        # aclip = aclip.set_start(audio_start)
        # aclip = aclip.set_duration(adjduration)  
        # aclip = aclip.set_duration(adjduration) 
        ##
        aclip = aclip.set_duration(adjduration) 
        aclip.write_audiofile(adjustedfile, fps=int(44100 * afactor))
        """

        aclip = AudioFileClip(adjustedfile)
        aclip = aclip.set_start(audio_start)
        adjustedaudios.append(aclip)
        

videoclip = mp.VideoFileClip(input)
videoclipfinal = videoclip.set_audio(CompositeAudioClip(adjustedaudios))
videoclipfinal.write_videofile(os.path.join(args.filename, "finalvideo.mp4"))
