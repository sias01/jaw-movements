import pandas as pd
import os
from pydub import AudioSegment

df = pd.read_csv("./features.csv")
# print(df.head())

df_mod = df.groupby(['recording']).agg(
    events=pd.NamedAgg(column='event', aggfunc=list),
    start_times=pd.NamedAgg(column='start', aggfunc=list),
    end_times=pd.NamedAgg(column='end', aggfunc=list)
).reset_index()

print(df_mod.head())

os.makedirs("segmented_audios", exist_ok=True)

# print(len(df_mod))

for i in range(len(df_mod)):
    recording_file = AudioSegment.from_wav(os.path.join("audios", df_mod['recording'][i] + ".wav"))
    events = df_mod['events'][i]
    start_times = df_mod['start_times'][i]
    end_times = df_mod['end_times'][i]
    print(i)
    for j in range(len(events)):
        start_time = start_times[j] * 1000  # convert to milliseconds
        # print(start_time)
        end_time = end_times[j] * 1000  # convert to milliseconds
        # print(end_time)
        event = events[j]
        audio_segment = recording_file[start_time:end_time]
        # print(len(audio_segment))
        audio_segment.export(os.path.join("segmented_audios", df_mod['recording'][i] + "_" + event + "_" + str(j) + ".wav"), format="wav")
        print(os.path.join("segmented_audios", df_mod['recording'][i] + "_" + event + "_" + str(j) + ".wav"))
    #     break
    # break
        