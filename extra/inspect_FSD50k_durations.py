import os
import time

import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import mutagen as mg
from mutagen.wave import WAVE

PATH_TO_FSD50K_TRAIN = r'C:\Users\mp431591\Documents\FSD50K_dev_audio'
PATH_TO_FSD50K_EVAL = r'C:\Users\mp431591\Documents\FSD50K_eval_audio'

durations = []

start = time.time()
with os.scandir(PATH_TO_FSD50K_TRAIN) as fsd_train_root:
    for idx, entry in enumerate(fsd_train_root):

        audio_file = WAVE(entry.path)
        durations.append(int(audio_file.info.length)) # At 44.1 kHz, target is 32 kHz
        
        if (idx+1) % 100 == 0:
            print(f"{idx+1} file durations gotten.")
        if idx == -1:
            break
        idx += 1
end = time.time()
print(f"Elapsed time: {end-start}")

counts, bins = np.histogram(durations)
plt.hist(durations, bins='auto')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.show()