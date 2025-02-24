from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline, Wav2Vec2Processor
import torchaudio
import io
import re
import torch

app = FastAPI(title="STT API")

# Determine available device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor with explicit device mapping
model = pipeline(
    "automatic-speech-recognition",
    model="StefanStefan/Wav2Vec-100-CSR-9M",
    framework="pt",
    device=device  # Add this line
).model.to(device)  # Explicit device placement

processor = Wav2Vec2Processor.from_pretrained("StefanStefan/Wav2Vec-100-CSR-9M")


def clean_transcription(text: str) -> str:
    return re.sub(r'<s>|</s>|<unk>', '', text).strip().lower()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ["audio/wav", "audio/mpeg", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        # Load and process audio
        audio_bytes = await file.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process inputs and move to same device as model
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160000
        ).to(device)  # Add this line to move inputs to device

        # Generate predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)

        # Move predictions back to CPU for decoding
        transcription = processor.batch_decode(
            pred_ids.cpu().numpy(),  # Move to CPU before decoding
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            filter_illegal_characters=True,
            illegal_characters_re=re.compile(r'[^A-Z\' ]')
        )[0]

        return {"text": clean_transcription(transcription)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")