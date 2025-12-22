# Meme Dubber

**Meme Dubber** is an AI-powered project that extracts text from meme images and generates audio dubbing using **Google Gemini API** and **text-to-speech (TTS) engines**.  

It can either extract **existing text** from the meme or generate **meme-style dialogue** using Gemini's vision capabilities. The text is then converted into speech via one of two TTS engines:  

* **gTTS** – a fast, cloud-based Google Text-to-Speech engine
* **ChatTTS** – a more natural-sounding, locally processed voice synthesis engine

The generated audio is then transformed into a custom voice using a model trained via **RVC WebUI** — an open-source interface for training voice models with **Retrieval-Based Voice Conversion (RVC)**.

This voice model (`.pth`) converts the TTS-generated audio into a specific target voice, enabling highly personalized and expressive meme dubbing.  

The project includes a user-friendly Gradio web interface, allowing users to upload meme images, preview and download audio, choose between TTS engines, and apply custom voice models via a simple web interface.

(再改)

---

## Prerequisites

- Python 3.8 or higher (3.10 recommended)
- Anaconda or Miniconda installed (recommended for environment management)
- Google API key for Gemini API ([Get one here](https://aistudio.google.com/app/apikey))
- Internet connection (required for gTTS and Gemini API)
- At least 2GB of free disk space (for ChatTTS models)

---

## Installation
Installing `fairseq` on Windows requires compiling C++ code.   
If you do not run the terminal as an Administrator, the installation may fail.  
Therefore, it is **strongly recommended** to run the terminal as ***Administrator*** throughout the entire process.

### 1. Create Conda Environment and Install Python:
```bash
conda create -n Meme_env python=3.10
conda activate Meme_env
```

### 2. Install PyTorch:
Install PyTorch according to whether you use **CPU** or **GPU**.

* CPU version:
```bash
pip install torch==2.9.0+cpu torchaudio==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

* **GPU version (CUDA 11.8 only — required for RVC)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Why CUDA 11.8?**
> RVC depends on multiple third-party libraries (e.g., PyTorch, faiss, fairseq),
> and **CUDA 11.8 is the only version that has been verified to work reliably**
> with the complete RVC inference pipeline.

### 3. Install Microsoft Visual C++ Build Tools (C++ Compiler):  
Installing `fairseq` requires compiling C++ extensions, so Windows users must install a C++ compiler.

Follow these steps:
1. Go to https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Download and run the installer
3. Check the "C++ build tools" workload
4. Also check "Windows 10 SDK" or "Windows 11 SDK"
5. Install and restart your computer if required

> macOS and Linux usually come with a C++ compiler preinstalled, so this step is generally not needed on those systems.

### 4. Clone the Repository and Install Dependencies:
```bash
git clone https://github.com/ml-team12-meme-dubber/Meme_Dubber Meme_Dubber
cd Meme_Dubber
pip install -r requirements.txt
```
This will install: (再改)
* Common packages such as librosa, soundfile, scikit-learn, faiss-cpu
* fairseq from GitHub (requires C++ compilation)
* The inferrvc package (in .whl format)

### 5. Downgrade pip and Install Compatible Package Versions (to Avoid Dependency Conflicts):
To avoid dependency conflicts between `gradio`, `gradio_client`, `websockets`, and `google-genai`, it is recommended to downgrade pip and install specific versions of these packages:
```bash
# Downgrade pip to improve compatibility
python -m pip install pip==23.2.1

# Install specific compatible versions
pip install gradio==4.19.2
pip install gradio_client==0.10.1
pip install websockets==15.0.1
pip install google-genai==1.52.0
```

### 6. Modify `inferrvc` Source Code:
(1) **Enable CPU Inference Mode**
To run RVC inference **without** an NVIDIA GPU: 
* Open the following file:  
```bash
<your Conda environment path>\Lib\site-packages\inferrvc\pipeline.py
```

* Locate the following line (approximately line 31):
```python
bh, ah = torch.from_numpy(bh).to(_gpu, non_blocking=True), torch.from_numpy(ah).to(_gpu, non_blocking=True)
```

* Replace it with:
```py
device = "cuda" if torch.cuda.is_available() else "cpu"
bh, ah = torch.from_numpy(bh).to(device), torch.from_numpy(ah).to(device)
```

(2) **Ensure GPU Mode Works Properly**
`torchaudio`'s resample module does not support FP16 tensors and may raise an error like:
```bash
RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

To fix this, force the model to use FP32 (float32) instead:  
* Open the file:  
```bash
<your Conda environment path>\Lib\site-packages\inferrvc\configs\config.py
```

* Find all lines containing:
```py
self.is_half = True
```

* Replace them with:
```py
self.is_half = False
```
This ensures RVC uses FP32 inference on both CPU and GPU.

### 7. Configure Environment Variables:
* Copy the example environment file and set your Google API key:
```bash
# Linux/macOS
cp .env.example .env
# Windows
copy .env.example .env
```

* Edit `.env` and add:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

---

## Usage

**Important:** Make sure your conda environment is activated before running:
```bash
conda activate Meme_env
```

### 1. Placing Your RVC Models
> **Note:** This project is intended for **educational and personal use only**.  
> We do **not** provide our personal voice model due to **privacy concerns**.  
> Please download other publicly available models from [Hugging Face (example)](https://huggingface.co/models?other=rvc&utm_source) for testing purposes.

```perl
Meme_Dubber/
├── asset/                    # ChatTTS model files (auto-downloaded)
│   ├── Decoder.safetensors
│   ├── DVAE.safetensors
│   ├── Embed.safetensors
│   ├── Vocos.safetensors
│   ├── gpt/
│   └── tokenizer/
├── index/                    # RVC index files (.index) (NOT included in repo)
│   └── .index
├── model/                    # RVC model files (.pth) (NOT included in repo)
│   └── .pth
├── .env                      # API keys (user-created, not tracked by git)
├── .env.example              # Template for environment variables
├── meme_dubber.py            # Main application script
├── meme_audio_gtts.mp3       # Output from gTTS
├── meme_audio_chattts.wav    # Output from ChatTTS
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```


### 2. Start the application:
```bash
python meme_dubber.py
```

### 3. Open your browser and navigate to:
   ```
   http://127.0.0.1:7860
   ```

### 4. Generate audio:
- Upload a meme image (or paste from clipboard)
- Select your preferred TTS engine:
   - **gTTS**: Faster, requires internet
   - **ChatTTS**: Better quality, works offline (after initial setup)
- Click "Generate Audio Dub"
- View the extracted/generated text
- Listen to and download the generated audio!


### 5. Adjust `f0_up_key` (Pitch Control)
`f0_up_key` controls the pitch shift (in semitones) applied during RVC voice conversion.

* Basic Meaning

| Value | Effect |
| --- | --- |
| `0` | No pitch change |
| `> 0` | Raises pitch (brighter voice) |
| `< 0` | Lowers pitch (deeper voice) |


* Suggested Values by Scenario:  

| Scenario | Suggested `f0_up_key` |
| --- | --- |
| Male → Female | `+5 ~ +12` |
| Female → Male | `-5 ~ -12` |
| Male → Higher-pitched Male | `+2 ~ +5` |
| Female → Lower-pitched Female | `-2 ~ -5` |
| Pitch already matches target | `0` |
---

(以下再改)

## Troubleshooting

### "GOOGLE_API_KEY not found" error
- Make sure you created a `.env` file in the project root
- Verify your API key is correctly set in the `.env` file: `GOOGLE_API_KEY=your_key_here`
- Check there are no extra spaces or quotes around the key
- Restart the application after creating/modifying `.env`

### "No text found in the image" error
- The image may not contain readable text
- Try a different meme image with clearer text
- The AI will attempt to generate meme-style dialogue if no text is detected

### ChatTTS model download issues
- ChatTTS downloads models (~2GB) on first use to `asset/` directory
- Ensure you have stable internet connection
- Make sure you have at least 2GB free disk space
- Check write permissions in the project directory
- If download fails, delete `asset/` folder and try again

### ChatTTS fails to load or crashes
- ChatTTS requires more system resources and dependencies
- Ensure PyTorch is properly installed: `python -c "import torch; print(torch.__version__)"`
- Try using gTTS as a faster alternative
- Check if you have enough RAM (recommended: 4GB+)
- Verify all dependencies are installed in the conda environment

### Network errors with gTTS
- gTTS requires internet connection to Google's TTS service
- Check your network connectivity
- If behind a proxy, configure proxy settings
- Consider using ChatTTS for offline processing (after initial model download)

### "Audio file failed to create" error
- Check disk space availability
- Verify write permissions in the project directory
- Try the alternative TTS engine
- Check terminal output for detailed error messages

### Gradio interface won't open
- Make sure port 7860 is not in use: `lsof -i :7860`
- Try accessing via http://localhost:7860 instead of 127.0.0.1
- Check firewall settings
- Look for error messages in terminal output

---

## Development

This project was originally based on a Google Colab notebook, and has been converted into a standalone local web application with the following key enhancements:

### Key Technologies

* Google Gemini 2.5 Flash — advanced multimodal AI for image and text understanding

* Thinking Mode — extended token budget (1024 tokens) improves text extraction accuracy

* Gradio — web UI framework for fast prototyping, using the “Soft” theme

* PyTorch — powers the neural network behind ChatTTS

* Transformers — Hugging Face library used for model loading and inference

---

## How It Works
### Text Extraction Pipeline
1. **Image Upload**
   User uploads a meme image via the Gradio interface

2. **AI Analysis**
   Google Gemini 2.5 Flash analyzes the image using multimodal thinking mode

3. **Smart Detection**
   If the image contains clear text → extract it directly  
   If not → generate a meme-style dialogue based on image content

4. **Language Detection**
   Automatically detects language (e.g., en, zh-tw, ja, es, etc.)

5. **JSON Output**
   Returns structured data including extracted/generated text and language code

### Audio Generation Pipeline
1. **TTS Selection**
   User selects either gTTS or ChatTTS as the speech engine

2. **Audio Synthesis**
   * gTTS → sends text to Google’s cloud TTS API → receives MP3
   * ChatTTS → runs local PyTorch model → generates 24 kHz WAV

3. **Audio Output**
   Final audio is saved locally and played in-browser

4. **Download**
   Users can preview and download the audio file directly from the web interface

---

## TTS Engine Comparison

| Feature | gTTS | ChatTTS |
|---------|------|---------|
| Speed | Fast | Slower (first run requires model download) |
| Quality | Good | Excellent, more natural |
| Languages | Many (100+) | Focus on Chinese/English |
| Requirements | Lightweight (~10MB) | Requires more resources (~2GB models) |
| Network | Cloud-based (requires internet) | Local processing (offline after setup) |
| Output Format | MP3 | WAV (24kHz) |
| First-time Setup | Ready to use | Downloads models (~2GB) |

---

## Reference Repositories
Below are the open-source repositories referenced by our project:

- https://github.com/YY-BOY/ml
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI 
- https://github.com/CircuitCM/RVC-inference  

