# the 1st experiment for Audio encoder -- Chinese-HuBERT-Large
import torch
import transformers
import soundfile as sf
from transformers import (
    SequenceFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from pathlib import Path


if __name__ == '__main__':
    model_path = Path(__file__).resolve().absolute().parent.parent / 'pretrained_weights' / 'chinese-hubert-large'
    wav_path=Path(__file__).resolve().absolute().parent.parent / 'datasets' / 'audio_eg' / 'sample-15s.wav'
    device = 'cuda:7'

    sampling_rate = 16000
    # base class: transformers.SequenceFeatureExtractor
    # base class of base class: transformers.feature_extraction_utils.FeatureExtractionMixin
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, sampling_rate=sampling_rate)
    model = HubertModel.from_pretrained(model_path).to(device=device, dtype=torch.bfloat16).eval()

    wav, sr = sf.read(wav_path)
    single_wav = wav[:, 0: 1]
    input_values = feature_extractor(
        single_wav,
        sampling_rate=sampling_rate,
        return_tensors='pt'
    ).input_values
    input_values = input_values.transpose(1, 0).contiguous().to(dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        outputs = model(input_values)
        last_hidden_state = outputs.last_hidden_state
