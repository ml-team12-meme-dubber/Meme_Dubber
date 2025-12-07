#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meme Dubber - A web application for generating teacher's audio from meme images 
Uses Google Gemini API for text extraction and gTTS/ChatTTS for audio generation,
Then use RVC for voice conversion.
"""

import os
import json
import sys
from io import BytesIO
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from inferrvc import RVC, load_torchaudio
import soundfile as sf
import torch
import fairseq.data.dictionary


# Load environment variables from .env file, which should contain GOOGLE_API_KEY
load_dotenv()

# Import Google Gemini SDK
from google import genai
from google.genai import types

# Get API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file")


def extract_text_from_meme(image):
    """
    Extract text and language from meme image using Google Gemini API
    Args:
        image: PIL Image object
    Returns:
        tuple: (meme_text, lang_code)
    """
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # System instruction for Gemini
        system_instruction = """
        You are an expert meme analyst.
        """

        # Call Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=img_byte_arr,
                    mime_type='image/png'
                ),
                """
                You are an expert meme analyst. Your task is to analyze the provided image.
                1.  First, determine if the image contains clear, readable text (e.g., captions, dialogue).
                2.  If it DOES contain text: Extract the text verbatim.
                3.  If it does NOT contain text (or the text is unreadable): Create a short, funny, meme-style dialogue that fits the scene, characters, and mood.
                4.  Identify the primary language of the extracted or generated text. Use standard language codes (e.g., 'en' for English, 'zh-tw' for Traditional Chinese, 'ja' for Japanese, 'es' for Spanish).
                5.  Return your response as a single JSON object with two keys: "language_code" and "text". Do not add any other explanatory text or formatting.

                Example for an English meme:
                {
                    "language_code": "en",
                    "text": "This is the text from the meme."
                }

                Example for a Japanese meme without text:
                {
                    "language_code": "ja",
                    "text": "面白いセリフを生成しました。"
                }
                """
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024 # 思考的預算，單位為token
                ),
                temperature=0.7, # 隨機性
                response_mime_type="application/json" # 指定回應格式為JSON
            )
        )

        # Parse response
        print(f"Gemini response: {response.text}")  # Add debug info
        result = json.loads(response.text)
        meme_text = result.get('text', '')
        lang_code = result.get('language_code', 'en')
        
        # 確保返回有效的文本
        if not meme_text or meme_text.strip() == "":
            print("Warning: Empty text extracted from meme")
            return None, "en"

        return meme_text, lang_code

    except Exception as e:
        print(f"Error extracting text from meme: {e}")
        import traceback
        traceback.print_exc()
        return None, "en"  


def generate_audio_gtts(text, lang_code):
    """
    Generate audio using Google Text-to-Speech (gTTS)
    Args:
        text: Text to convert to speech
        lang_code: Language code (ISO 639-1)
    Returns:
        str: Path to generated audio file
    """
    try:
        from gtts import gTTS

        # Generate speech
        tts = gTTS(text=text, lang=lang_code)

        # Create full audio file path
        audio_filename = "meme_audio_gtts.mp3"
        audio_file = os.path.join(os.getcwd(), audio_filename)
        
        # Debug information
        print(f"Current working directory: {os.getcwd()}")
        print(f"Audio file path: {audio_file}")
        print(f"Is directory: {os.path.isdir(audio_file)}")
        
        # Save audio file
        tts.save(audio_file)
        
        # Verify file was created
        if os.path.exists(audio_file):
            print(f"✓ Audio file created successfully: {audio_file}")
        else:
            print(f"✗ Failed to create audio file")
            return None

        return audio_file

    except Exception as e:
        print(f"Error generating audio with gTTS: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_audio_chattts(text, lang_code):
    """
    Generate audio using ChatTTS
    Args:
        text: Text to convert to speech
        lang_code: Language code (ISO 639-1)
    Returns:
        str: Path to generated audio file
    """
    try:
        import ChatTTS
        import soundfile as sf

        # Load ChatTTS model
        chat = ChatTTS.Chat()
        chat.load(compile=False)

        # 採樣隨機說話的人聲 
        # rand_spk = chat.sample_random_speaker()  
        # print("Randomly sampled speaker embedding:!!!!!!!")
        # print(rand_spk)
        fix_spk = "蘁淰敥欀桃湤緕筝憭熕卽褥誼椝簬嶁甤櫼諰砃櫐狶盠誸凄勧嘪爟呭君冲砗侤莆譎諶呔桊叏虆呴覠欻毟草彳扌畩偄炑薧昬殥碫暢扱瘲乺爵坺詍曈幙偄積疭庂梻峋勩耭筶栣嗤礥硘箢賹桉某朣見拕媗絇兙堾垜姮莗哻吮卨苈蘰綧啾趍呎豪砼脯竈券熢萨沽烯蕂聍俌厰虝渷冟嵌衧榆溗懆抃婼僇创罞膇桳胎蘛朋圄珈兹欨愰擞眗翤嗞蓴楼斘瘥縷聟跬諮疁梹甀灂崂皕庬砌呓胦責臽譟淛塛尩硇擌茞慒盐塎垖訁侅檅刞恴俴繼累囮桥媳噵忀寷刄勓崶衝禢易仿孆譅芛织儫毻煺蕤正梞尭涆孶訑堍蛯恀樝类拿簟嶂讵佲焩噦聏满纯悶籁冉挪啠豳襏儱籑嶤琮渘柄噺珡橇菾慗娣澡樲燾臶癤燲畽洇菉液砖嘁椑塅諼襶檣巛俹潭唄灀沖砧烷灜娳噵侃墊肂学猒粇愇互澩紓笖误展或斛渘賙艛湛煝羶勲氚燙穽晌仍跐趚嫀瓶籖嫊匵獘垝嘣艈莰唏詠琺睲嵩濫扫窵櫷眡粎恶囨礝讙玄暡廞藖算恦勝沺膼盠侤慭負耧曛淡傶嬸葟质懼趉椻敯認秨删偂杉咦覼穡紵浣烤嫱洎篳庄妡恶庱巗簡牻硢攡撩叐猽芣敦汘袃嫽觧氩莪泧桿禤烒圵翛汄蕏嵏瞟垛伮蟘菎嫮拱磷涀玐廊臬悏幣炞攁攣抈脣薬莇沨做劙伊枘攕緸蕾彇蘙犎剚簟瞂堘蓵徑犁玉虌憩蛥欯蠞牦峧掾壞绥糘搣瓻募叁焳莄揿嬡屙昈翞戍癉級带蠠妜寍啎璥濒偬崯竌礌溧卟澚聟姛汪垤歞樓澋苮畱滊谈眸篬橰依歉纟讯嗜礃澿俓欱嚟冝襓偈伈丛谒懕誷汍彋坱瘧坉祐趶篾嚾譍勻諏裡千灲柛跪悴寉牨绳嶵觷劾毄湊滚椳嶍弣愸皻甉素涭摾惎猻懸嫰嫮您脯掭坒玮堸瑮菷曄列懔戯諺秶羿敘覄畢缺涵貸垌丱磪掱観蘅葛犓睰戄蚶衘蒏苚箼毝岑毺擞総嘻跳劤繯赳寴薰缥弁儈癕敳儅化佛叿蒩櫗疨皏禡譳蚾斋趒泏瘒卖礪家炪扑嵴吓橶曱貅揿攄螗柏蝼漣殟貼是勜皒趤嫇瓒淒是牖系倶徙粂眘櫙燣洡坞夅叶乡惇祛模渑澈橝细溉挖築侞瞊擬篗衬捍帬榠椦詂稂殔睄哠胉腳戺庅煍畹卐禟硥珜梣蚂儝竱屇婅琞統痫儷嶓杲庹绢塦綫聥觕喟跟裤蜐禓呐旺蜲蘚檶熖喖氀垪礕蒡桳皁劆墄砲竽紘巚諾荛吐犢叒伮豱荟圪嬛糖灊溲巍匰们挳沽渝杸繙狂蒞仪勻桉趘湔楬猎蜉熃藯纻渮裟椉袉垷拥呓冼琞玜残伍忚詛歗艗股蟭乮寃翤塞仈奮譡坋窤槚攭涔贩凟曽狢畳蛠石稸嵂諉沥盼嶕菕艦噰玭更氺掸撕糪唅巁曰滃襞砋苄胃誔幏罱筺栃痼旗箠埽縘侘聅嬁罦甒趉眲崖爼泦蝎宛桱槀荽胫裢袉世妔穞稢跗眥甯侖撩劚硶砙坨宠一"

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=fix_spk,  # 使用固定的說話人嵌入
            # spk_emb=rand_spk,  # 使用採樣的說話人嵌入
            temperature=.00001,
            top_P=0.7,
            top_K=20,
        )

        # Generate speech
        wavs = chat.infer([text], params_infer_code=params_infer_code)

        # Chattts有時候音量太小聲！ 
        # 自動將最大音量縮放到 1.0
        wav = wavs[0]
        import numpy as np
        current_peak = np.max(np.abs(wav)) if wav.size else 0.0
        if current_peak > 0:
            wav = wav / current_peak  # 縮放最大音量到 1.0

        # Create full audio file path
        audio_filename = "meme_audio_chattts.wav"
        audio_file = os.path.join(os.getcwd(), audio_filename)
        
        # Debug information
        print(f"Current working directory: {os.getcwd()}")
        print(f"Audio file path: {audio_file}")
        
        # Save audio file using soundfile
        sf.write(audio_file, wavs[0], 24000)
        
        # Verify file was created
        if os.path.exists(audio_file):
            print(f"✓ Audio file created successfully: {audio_file}")
        else:
            print(f"✗ Failed to create audio file")
            return None

        return audio_file

    except Exception as e:
        print(f"Error generating audio with ChatTTS: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_meme(image, tts_engine):
    """
    Main function to process meme image and generate audio

    Args:
        image: PIL Image object from Gradio
        tts_engine: TTS engine to use ("gTTS" or "ChatTTS")

    Returns:
        tuple: (extracted_text, audio_file_path)
    """
    if image is None:
        return "**Error:** Please upload an image first.", None

    # Extract text from meme
    meme_text, lang_code = extract_text_from_meme(image)

    # 檢查文本提取是否成功
    if not meme_text or meme_text is None:
        return "**Error:** No text found in the image. Please try another image.", None

    # Generate audio based on selected TTS engine
    try:
        if tts_engine == "gTTS":
            audio_file = generate_audio_gtts(meme_text, lang_code)
        else:  # ChatTTS
            audio_file = generate_audio_chattts(meme_text, lang_code)
        
        # 檢查音檔是否生成成功
        if audio_file is None:
            result_text = f"**Extracted Text:** {meme_text}\n\n**Language:** {lang_code}\n\n**Error:** Failed to generate audio"
            return result_text, None
        
        result_text = f"**Extracted Text:** {meme_text}\n\n**Language:** {lang_code}"
        return result_text, audio_file
        
    except Exception as e:
        error_text = f"**Error generating audio:** {str(e)}\n\n**Extracted Text:** {meme_text}\n\n**Language:** {lang_code}"
        print(f"Error in process_meme: {e}")
        import traceback
        traceback.print_exc()
        return error_text, None

def rvc_convert(input_audio_path, f0_key):
    """
    Convert audio to teacher voice using RVC
    Args:
        input_audio_path: Path to meme_audio_xxx.wav
        f0_key: int, pitch shift (-24 ~ 24)
    Returns:
        str: Path to converted audio (meme_audio_teacher.wav)
    """

    print("=== RVC Inference Start ===")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "Teacher_infer.pth")
    index_path = os.path.join(base_dir, "index", "Teacher_infer.index")

    output_path = os.path.join(os.getcwd(), "meme_audio_teacher.wav")

    # fairseq fix
    try:
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
    except:
        pass

    # Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load RVC model
    model = RVC(model_path, index=index_path)
    print("Loaded RVC:", model.name)

    # Load input audio
    audio, sr = load_torchaudio(input_audio_path)

    # Run conversion
    converted = model(
        audio,
        f0_up_key=int(f0_key),     # ★ 加入音高控制
        output_device="cpu",
        output_volume=RVC.MATCH_ORIGINAL,
        index_rate=0.5
    )

    if isinstance(converted, torch.Tensor):
        converted = converted.detach().cpu().numpy()

    # Save output
    sf.write(output_path, converted, 44100)
    print("Saved:", output_path)

    return output_path


def create_gradio_interface():
    """
    Create and configure Gradio web interface

    Returns:
        gr.Blocks: Gradio interface
    """
    with gr.Blocks(title="Meme Dubber", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 😜 Meme Dubber

            Upload a meme image and generate audio dubbing using AI!

            **How it works:**
            1. Upload a meme image
            2. Select your preferred TTS engine (gTTS or ChatTTS)
            3. Click "Generate Audio Dub"
            4. Listen to the AI-generated voiceover!
            """
        )

        with gr.Row():
            with gr.Column():
                # Input components
                image_input = gr.Image(
                    label="Upload Meme Image",
                    type="pil",
                    sources=["upload", "clipboard"]
                )

                tts_selector = gr.Radio(
                    choices=["gTTS", "ChatTTS"],
                    value="gTTS",
                    label="TTS Engine",
                    info="gTTS: Fast and simple | ChatTTS: More natural sounding"
                )

                pitch_slider = gr.Slider(
                                    minimum=-24,
                                    maximum=24,
                                    value=-5,
                                    step=1,
                                    label="Pitch Shift (F0 Key)",
                                    info="Adjust voice pitch (-24 ~ 24)"
                                )

                generate_btn = gr.Button("🎬 Generate Audio Dub", variant="primary", size="lg")
                rvc_btn = gr.Button("🎤 Convert to Teacher Voice", variant="secondary")

            with gr.Column():
                # Output components
                text_output = gr.Markdown(label="Extracted Text")
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
                rvc_audio_output = gr.Audio(label="Teacher Voice Audio", type="filepath")

        # Set up event handler
        generate_btn.click(
            fn=process_meme,
            inputs=[image_input, tts_selector],
            outputs=[text_output, audio_output]
        )

        def run_rvc(audio_file, f0_key):
            if audio_file is None:
                return None
            return rvc_convert(audio_file, f0_key)
        
        rvc_btn.click(
            fn=run_rvc,
            inputs=[audio_output, pitch_slider],
            outputs=[rvc_audio_output]
        )

        gr.Markdown(
            """
            ---
            ### Notes:
            - **gTTS**: Google Text-to-Speech - Fast, cloud-based, supports many languages
            - **ChatTTS**: More natural sounding but requires more processing power
            - Supported languages: English, Chinese, Japanese, Spanish, and more!
            """
        )

    return demo


def main():
    """
    Main function to launch the application
    """
    print("Starting Meme Dubber...")
    print(f"API Key configured: {'✓' if GOOGLE_API_KEY else '✗'}")

    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
