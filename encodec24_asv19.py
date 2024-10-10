import torchaudio
import torch
from transformers import EncodecModel, AutoProcessor
import pandas as pd

# 메타데이터 불러오기
df = pd.read_csv('/home/aix7101/yoonseo/VF/ASVspoof2019_LA_eval_Bonafide/ASVspoof2019.LA.cm.eval_Bonafide.trn', header=None, sep=' ')
df.columns = ['speakID', 'fileName', 'non', 'model', 'ANS']


# 로컬 폴더에서 오디오 파일을 불러오기
path = "/home/aix7101/yoonseo/VF/ASVspoof2019_LA_eval_Bonafide/flac/"

audio_file_paths =[]
for i in df['fileName'] :
    audio_file_paths.append(path+i+'.flac')


# 모델과 프로세서 불러오기
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")


for i, audio_file in enumerate(audio_file_paths):

    # 오디오 파일을 불러오고 샘플링 레이트 맞추기
    audio_sample, sr = torchaudio.load(audio_file)
    
    # 샘플링 레이트를 모델에 맞추기 (24kHz)
    if sr != processor.sampling_rate:
        audio_sample = torchaudio.transforms.Resample(sr, processor.sampling_rate)(audio_sample)

    # 오디오가 이미 모노일 경우: (channels, samples)에서 (samples)로 변환
    if audio_sample.size(0) == 1:  # 1개의 채널 (모노)
        audio_sample = audio_sample.squeeze(0)  # (samples)로 변환

    # 모델에 입력할 수 있는 포맷으로 변환
    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
    
    # 모델을 사용해 오디오 처리
    audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

    # 결과 저장
    #print(audio_values.shape)
    torchaudio.save(f"/home/aix7101/yoonseo/VF/ASVspoof2019_LA_eval_Spoof/wav/E01_19E_{i:07}.wav", audio_values.squeeze(0), processor.sampling_rate)

    # metadata 수정
    filename = audio_file.replace('/home/aix7101/yoonseo/VF/ASVspoof2019_LA_eval_Bonafide/flac/','').replace('.flac', '')
    df.loc[df['fileName'] == filename, 'fileName'] = f'E01_19E_{i:07}'


    if i % 250 == 0 : print(f'now : {i}')

# 바뀐 메타데이터 새로 저장
df['model'] = 'E01'
df['ANS'] = 'spoof'
df.to_csv('/home/aix7101/yoonseo/VF/ASVspoof2019_LA_eval_Spoof/eval_output.csv', sep= ' ', index=False, header=False)