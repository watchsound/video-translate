import azure.cognitiveservices.speech as speechsdk
import os

speech_config = speechsdk.SpeechConfig(
    subscription=os.environ.get('COGNITIVE_SERVICE_KEY'), region="westus")
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
file_config = speechsdk.audio.AudioOutputConfig(filename="./output-tts.wav")


speech_config.speech_synthesis_voice_name = 'zh-CN-XiaomoNeural'

speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, audio_config=file_config)

def speakIt(voice, emotion):
    print(" ======= start with voice " + voice )
    text = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
        <voice name="zh-CN-{voice}">
            女儿看见父亲走了进来，问道：
            <mstts:express-as    style="{emotion}">
                “您来的挺快的，怎么过来的？”
            </mstts:express-as>
            父亲放下手提包，说：
            <mstts:express-as   style="fear">
                “刚打车过来的，路上还挺顺畅。”
            </mstts:express-as>
        </voice>
    </speak>
    """


    #speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    speech_synthesizer.speak_ssml_async(ssml=text).get()
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml=text.format(voice=voice, emotion=emotion)).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("SynthesizingAudioCompleted result")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(
                    cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


females = ["XiaohanNeural",  "XiaomengNeural", "XiaomoNeural", "XiaoxiaoNeural", "XiaoxuanNeural",
           "XiaoyiNeural", "XiaoruiNeural", "XiaozhenNeural", 'YunxiaNeural',
           "XiaoshuangNeural"]

males = ['YunfengNeural', 'YunjianNeural', 'YunjianNeural3', 'YunjianNeural4',
           'YunxiNeural', 'YunyangNeural',
         'YunyeNeural', 'YunzeNeural', 'YunhaoNeural2', 'YunhaoNeural']

males = ['YunxiaNeural']

for voice in males:
    speakIt(voice, "angery")