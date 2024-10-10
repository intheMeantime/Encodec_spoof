import torchaudio
import torch
from transformers import EncodecModel, AutoProcessor
import pandas as pd

# read path **
file = open("path/path_19E_bona.txt", "r")
path = file.read()
file.close()
file = open("path/path_19E_spoof.txt", "r")
path_spoof = file.read()
file.close()

# 메타데이터 불러오기 **
df = pd.read_csv(f'{path}/ASVspoof2019.LA.cm.eval_Bonafide.trn', header=None, sep=' ')
df.columns = ['speakID', 'fileName', 'non', 'model', 'ANS']


# 로컬 폴더에서 오디오 파일을 불러오기
audio_file_paths =[]
for i in df['fileName'] :
    audio_file_paths.append(path+'/flac/'+i+'.flac')


# 모델과 프로세서 불러오기
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


for i, audio_file in enumerate(audio_file_paths):

    # 오디오 파일을 불러오고 샘플링 레이트 맞추기
    audio_sample, sr = torchaudio.load(audio_file)
    
    # 샘플링 레이트를 모델에 맞추기 (processor.sampling_rate : 24khz)
    if sr != processor.sampling_rate:
        audio_sample = torchaudio.transforms.Resample(sr, processor.sampling_rate)(audio_sample)

    # 오디오가 이미 모노일 경우: (channels, samples)에서 (samples)로 변환
    if audio_sample.size(0) == 1:  # 1개의 채널 (모노)
        audio_sample = audio_sample.squeeze(0)

    # 모델에 입력할 수 있는 포맷으로 변환
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
    
    # 모델을 사용해 오디오 처리
    audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

    # 결과 저장 **
    torchaudio.save(f"{path_spoof}/wav/E01_19E_{i:07}.wav", audio_values.squeeze(0), processor.sampling_rate)

    # metadata 수정 **
    filename = audio_file.replace(f'{path}/flac/','').replace('.flac', '')
    df.loc[df['fileName'] == filename, 'fileName'] = f'E01_19E_{i:07}'

    # 진행 상황 확인
    if i % 250 == 0 : print(f'now : {i}')


# spoof 메타데이터 새로 저장 **
# E01 == Encodec 24khz
df['model'] = 'E01'
df['ANS'] = 'spoof'
df.to_csv(f'{path_spoof}/E01_19eval_spoof.csv', sep= ' ', index=False, header=False)