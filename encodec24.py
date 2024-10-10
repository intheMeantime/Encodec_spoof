from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import torchaudio

# dummy dataset, however you can swap this with an dataset on the 🤗 hub or bring your own
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


# cast the audio data to the correct sampling rate for the model
# 'audio' 열을 Audio()를 활용해 원하는 sr에 맞춰 데이터 타입으로 변환
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

#len(librispeech_dummy)

for i in range(5) : 
    audio_sample = librispeech_dummy[i]["audio"]["array"]
    print(audio_sample.shape)

    # pre-process the inputs
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    # or the equivalent with a forward pass
    audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

    audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes


    ### 오디오로 추출
    audio_values = audio_values.squeeze()

    #오디오파일로 저장
    torchaudio.save(f"spoof_test/encodec24_spoof_{i:03}.wav", audio_values.unsqueeze(0), processor.sampling_rate)

