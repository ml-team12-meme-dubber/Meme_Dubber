# Meme Dubber

**Meme Dubber** is an AI-powered system that generates speech from meme content and transforms it into a target voice using a model trained with Retrieval-Based Voice Conversion (RVC).

---
## Features

* **Text extraction & dialogue generation**  
  Extracts existing text from meme images or generates meme-style dialogue using the **Google Gemini API**.

* **Multiple TTS backends**  
  Converts text into speech using either **gTTS** or **ChatTTS**.

* **Custom voice conversion (RVC)**  
  Transforms synthesized speech into a target voice using **RVC models** trained via **RVC WebUI**.

* **Web-based user interface**  
  Provides a **Gradio web interface** for uploading images, selecting TTS engines, and downloading dubbed audio.

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

* **CPU version**:
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

This will install the following key dependencies:

- **fairseq (installed from GitHub, requires C++ compilation)**
    
    Used to extract **content features (semantic representations)** from input audio, which serve as the intermediate representation in the RVC pipeline.
    
- **faiss-cpu**
    
    Performs efficient **K-nearest neighbor (KNN) search** on the feature index, enabling fast similarity matching during voice conversion inference.
    
- **inferrvc (.whl package)**
    
    The core RVC inference package that integrates feature extraction, index lookup, and neural voice conversion into a unified pipeline.
    
- **Common audio & ML libraries**
    
    Including `librosa`, `soundfile`, and `scikit-learn`, which are used for audio processing, feature handling, and utility functions.


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
(1) **Enable CPU Inference Mode Works Properly**
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
* Copy the example environment file:

```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

* Edit `.env` and set your Google API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```
Replace `your_actual_api_key_here` with the Google API key you obtained.


### 8. Install ffmpeg (Windows Only)

> **Note (Windows users):**  
> ffmpeg is required for audio processing.  
> Please make sure `ffmpeg` is installed and available in your system `PATH`.


1. Download ffmpeg from the official website:  
   https://ffmpeg.org/download.html

2. Extract the archive and add the `bin/` directory to your system `PATH`.

3. Verify installation:
```bash
ffmpeg -version
```

---

## Usage

Make sure your conda environment is activated before running:

```bash
conda activate Meme_env
```

### 1. Placing Your RVC Models
> **Note:** This project is intended for **educational and personal use only**.  
> For demonstration purposes, the RVC model and index files are provided in the **Release** section.  
> Due to **privacy concerns**, these files will be **removed after the project evaluation is completed**.  
> If you would like to reproduce this project in the future, please download publicly available RVC models from  
> [Hugging Face (example)](https://huggingface.co/models?other=rvc&utm_source).


Please download the following two files from the  
[**Releases page**](https://github.com/ml-team12-meme-dubber/Meme_Dubber/releases/tag/RVC_model%26index):

- `Teacher_infer.pth` — RVC model weights  
- `Teacher_infer.index` — RVC voice feature index

After downloading, place the files into the corresponding directories as shown below:

```text
Meme_Dubber/
├── asset/                    # ChatTTS model files (auto-downloaded after execution)
│   ├── Decoder.safetensors
│   ├── DVAE.safetensors
│   ├── Embed.safetensors
│   ├── Vocos.safetensors
│   ├── gpt/
│   └── tokenizer/
├── index/                    # RVC index files (.index)
│   └── Teacher_infer.index
├── model/                    # RVC model files (.pth)
│   └── Teacher_infer.pth
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

#### Basic Meaning

| Value | Effect |
| --- | --- |
| `0` | No pitch change |
| `> 0` | Raises pitch (brighter voice) |
| `< 0` | Lowers pitch (deeper voice) |


#### Suggested Values (Based on This Model)
The optimal value of `f0_up_key` depends on the fundamental pitch characteristics of the input TTS audio.

| Input Source | Suggested `f0_up_key` | Rationale |
| --- | --- | --- |
| **gTTS** | `-14` | gTTS output tends to have a higher fundamental pitch, requiring a larger downward shift to match the target voice. |
| **ChatTTS** | `-5` | ChatTTS produces audio with a pitch closer to natural speech, so only a moderate adjustment is needed. |

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

