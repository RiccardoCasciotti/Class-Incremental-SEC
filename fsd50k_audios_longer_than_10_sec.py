import librosa as lb
import os
import time
import matplotlib.pyplot as plt
import numpy as np

"""
Point is to find out how many FSD50k files are longer than 10 seconds which is
a plausible clip length. If many files are longer than 10 then using 10 second 
threshold might not be a good idea.
"""
PATH_TO_FSD50k_DEV_DIR = r'S:\machine_listening\Datasets\FSD50K\FSD50K.dev_audio'
PATH_TO_FSD50k_EVAL_DIR = r'S:\machine_listening\Datasets\FSD50K\FSD50K.eval_audio'

test_files = os.scandir(PATH_TO_FSD50k_EVAL_DIR)
songs_over_10_secs = 0
counter = 0
audio_lens = []
file_number_limit = -1 # Negative number makes it into a full loop
start = time.perf_counter()
for entry in test_files:
    duration = lb.get_duration(path=entry.path)
    audio_lens.append(int(duration))
    if duration > 10:
        songs_over_10_secs += 1
    counter += 1
    if counter == file_number_limit:
        break

print(f"{songs_over_10_secs} out of {counter} were over 10 seconds long")
end = time.perf_counter()
print(f"{file_number_limit} files took {end-start}")

plt.hist(audio_lens, bins=np.linspace(0, 31))
plt.show()
    