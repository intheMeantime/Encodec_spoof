from transformers import EncodecModel, AutoProcessor
import torchaudio
import torch
import pandas as pd
import time

# read path **
file = open("path/path_19E_bona.txt", "r")
path = file.read()
file.close()
file = open("path/path_19E_spoof.txt", "r")
path_spoof = file.read()
file.close()

# read metadata **
df = pd.read_csv(f'{path}/ASVspoof2019.LA.cm.eval_Bonafide.trn', header=None, sep=' ')
df.columns = ['speakID', 'fileName', 'non', 'model', 'ANS']

# get the local audio path
audio_file_paths =[]
for i in df['fileName'] :
    audio_file_paths.append(path+'/flac/'+i+'.flac')

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_48khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

print('up to here completed')

for i, audio_file in enumerate(audio_file_paths):
    start = time.time()

    # Load an audio file
    audio_sample, sr = torchaudio.load(audio_file)

    #print(audio_sample.shape)
    if sr != processor.sampling_rate:
        audio_sample = torchaudio.transforms.Resample(sr, processor.sampling_rate)(audio_sample)   

    # 오디오 샘플이 1차원일 경우 스테레오로 변환
    if audio_sample.ndim == 1:  # 모노 오디오
        audio_sample = audio_sample.unsqueeze(0).repeat(2, 1)  # 스테레오로 변환
    elif audio_sample.ndim > 1:  # 여러 채널일 경우
        audio_sample = audio_sample.mean(dim=0, keepdim=True)  # 모노로 변환
        audio_sample = audio_sample.repeat(2, 1)  # 스테레오로 변환

    # pre-process the inputs
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
    
    # a forward pass
    audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

    # save spoof audio **
    torchaudio.save(f"{path_spoof}/E01_02_eval_wav/E01_02_19E_{i:06}.wav", audio_values.squeeze(0), processor.sampling_rate)

    # metadata **
    filename = audio_file.replace(f'{path}/flac/','').replace('.flac', '')
    df.loc[df['fileName'] == filename, 'fileName'] = f'E01_02_19E_{i:06}'

    # check
    if i % 100 == 0 :
        print(f'now : {i}')
        print(f"{time.time()-start:.2f} sec") 


# save the new metadata **
# E01_02 == Encodec 48khz
df['model'] = 'E01_02'
df['ANS'] = 'spoof'
df.to_csv(f'{path_spoof}/E01_02_19eval_spoof.csv', sep= ' ', index=False, header=False)