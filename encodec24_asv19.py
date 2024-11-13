import torchaudio
import torch
from transformers import EncodecModel, AutoProcessor
import pandas as pd
import time

# read path **
file = open("path/path_19D_bona.txt", "r")
path = file.read()
file.close()
file = open("path/path_19D_spoof.txt", "r")
path_spoof = file.read()
file.close()

# read metadata **
df = pd.read_csv(f'{path}/ASVspoofta2019.LA.cm.dev_Bonafide.trn', header=None, sep=' ')
df.columns = ['speakID', 'fileName', 'non', 'model', 'ANS']

# get the local audio path
audio_file_paths =[]
for i in df['fileName'] :
    audio_file_paths.append(path+'/flac/'+i+'.flac')

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


for i, audio_file in enumerate(audio_file_paths):
    start = time.time()

    # Load an audio file
    audio_sample, sr = torchaudio.load(audio_file)
    
    # 샘플링 레이트를 모델에 맞추기 (processor.sampling_rate : 24khz)
    if sr != processor.sampling_rate:
        audio_sample = torchaudio.transforms.Resample(sr, processor.sampling_rate)(audio_sample)

    # 오디오가 이미 모노일 경우: (channels, samples)에서 (samples)로 변환
    if audio_sample.size(0) == 1:  # 1개의 채널 (모노)
        audio_sample = audio_sample.squeeze(0)

    # 모델에 입력할 수 있는 포맷으로 변환
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
    
    # a forward pass
    audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

    # save spoof audio **
    torchaudio.save(f"{path_spoof}/E01_01_wav/E01_01_19D_{i:06}.wav", audio_values.squeeze(0), processor.sampling_rate)

    # metadata **
    filename = audio_file.replace(f'{path}/flac/','').replace('.flac', '')
    df.loc[df['fileName'] == filename, 'fileName'] = f'E01_01_19D_{i:06}'

    # check
    if i % 250 == 0 :
        print(f'now : {i}')
        print(f"{time.time()-start:.2f} sec")         


# save the new metadata **
# E01_01 == Encodec 24khz
df['model'] = 'E01_01'
df['ANS'] = 'spoof'
df.to_csv(f'{path_spoof}/E01_01_19dev_spoof.csv', sep= ' ', index=False, header=False)