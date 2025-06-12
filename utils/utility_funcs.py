import pickle
import pandas as pd
import torch

# Saves the object to the working directory
def pickle_save(name_for_saved_obj, obj):
    try:
        with open(name_for_saved_obj, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Error during pickling: {ex}")

def pickle_load(name_of_saved_pckl_obj):
    try:
        with open(name_of_saved_pckl_obj, "rb") as file:
            return pickle.load(file)
    except Exception as ex:
        print(f"Error during unpickling: {ex}")

def read_strong_audioset_metatable(path_to_as_table):

    table = pd.read_csv(path_to_as_table, sep="\t")
    table = table.loc[:, ['segment_id', 'label']] # Keep only these
    table.rename(columns={'label': 'mids'}, inplace=True)
    return table

def read_fsd50k_metatable(path_to_fsd_table):

    table = pd.read_csv(path_to_fsd_table)
    table = table.loc[:, ["fname", "mids"]]
    return table
    
def get_valid_audioset_filenames_for_mids(pd_as_table: pd.DataFrame, mids):
    
    pd_as_table.drop_duplicates(inplace=True)
    table_w_valid_rows = pd_as_table[pd_as_table.mids.isin(mids)]
    segments = table_w_valid_rows["segment_id"].unique()
    stripped_segs = [('_').join(seg.split('_')[0:-1]) for seg in segments]
    return stripped_segs

def get_valid_fsd50k_filenames_for_mids(pd_fsd_table: pd.DataFrame, mids):

    table_w_valid_rows = pd_fsd_table[pd_fsd_table.mids.isin(mids)]
    table_w_valid_rows = table_w_valid_rows["fname"].unique()
    return [str(fname) for fname in table_w_valid_rows]

# For use with Audioset and FSD50k metafile tables 
# i.e., there's a 'mids' column
def get_rows_in_mids(table: pd.DataFrame, mids):

    table_w_valid_rows = table[table.mids.isin(mids)]
    return table_w_valid_rows

# For spot checking if a file's labels match
def multihot_labels_translation(labels: list, multihot_to_mid_mapping: dict,
                                mid_to_english: dict):
    sounds = []
    for idx, val in enumerate(labels):
        if val == 1:
            corresponding_mid = multihot_to_mid_mapping[idx]
            corresponding_display_name = mid_to_english[corresponding_mid]
            sounds.append(corresponding_display_name)
    return sounds

def audioset_mid_to_display_name(mid_to_display_name_mapping, mid):
    return mid_to_display_name_mapping[mid_to_display_name_mapping['mids'] == mid]['display_name'].iloc[0]

"""
For Audioset:
There's a disconnect between the filenames of the audiofiles
and the filenames in the metafile.
meta: filename_XXXXX; data: Yfilename
The metafile suffix is assumed to be removed at this point
For FSD50k:
The filenames in the metafile already match the audiofilenames
"""

def filenames_to_txt(filenames: list, txtfile_name: str, 
                     dataset="audioset"):
    if dataset == "audioset":
        appended_filenames = [("Y" + fname) for fname in filenames]
        with open(txtfile_name, 'w') as f:
            for fname in appended_filenames:
                f.write(fname + '\n')
            #f.write("\n".join(appended_filenames))
    if dataset == "fsd50k":
        with open(txtfile_name, 'w') as f_fsd:
            for fname_fsd in filenames:
                f_fsd.write(fname_fsd + '\n')


# Manju's pad or truncate function
def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if x.size(1) <= audio_length:
        return torch.cat((x, torch.zeros(1, audio_length - x.size(1))), dim=1)
    else:
        return x[:, 0: audio_length]

"""
The function takes a suitable Pandas Dataframe that has
filenames and a corresponding mid per row. I.e., if a filename
has several labels, each label is on its own row. Another mids file is 
used to narrow down the files that have the desired labels. A mapping
file is used to make sure that labels (mids) are mapped to the same indices across
files. I.e., file1 and file2 have the same label in the same multihot index.

General discussion/advice on iterating through pandas dataframes
https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe/55557758#55557758
iterrows is slow and dumb, but a solution. Maybe refactor if needed.
"""
def get_multihot_labels_per_file(table: pd.DataFrame, 
                                 mids,
                                 mid_to_multihot_mapping,
                                 dataset='audioset') -> dict[str, list[int]]:
    segment_to_label_dict = {}
    valid_mids = get_rows_in_mids(table, mids)

    for idx, row in valid_mids.iterrows():
        fname = ""
        # Edit out the start_of_clip from filename if Audioset
        if dataset == 'audioset':
            fname = row['segment_id']
            fname = ('_').join(fname.split("_")[0:-1])
        elif dataset == 'fsd50k':
            fname = str(row['fname'])
        mid = row['mids']

        if fname not in segment_to_label_dict:
            segment_to_label_dict[fname] = [0] * len(mids)
            segment_to_label_dict[fname][mid_to_multihot_mapping[mid]] = 1
        else:
            segment_to_label_dict[fname][mid_to_multihot_mapping[mid]] = 1
    return segment_to_label_dict

"""
The function is meant to split FSD50k audiofiles longer than X secs to
X seconds and pad the final lacking to X seconds. 
This is to make it match with the Audioset's files. This can also
make the model training slightly easier to implement since there is uniformity
in the data. What effects could this chunking and label splitting have from the 
model's point of view? Any downsides?

Parameters:
input_audio: This is the mono audio to be split and assumed to be 
a torch.Tensor with a shape of (channels, samples) -> (1, samples)
and a duration longer than the target_duration.
samplerate: Samplerate of the input_audio.
target_duration: The desired duration of each chunk.
"""

def chunk_audio(input_audio: torch.Tensor, samplerate: int,
                target_duration: int) -> list:
    
    nr_of_audio_samples = input_audio.shape[1]
    duration = nr_of_audio_samples / samplerate

    split_chunks = []
    nr_of_chunks = int((duration // target_duration) + 1)

    for i in range(nr_of_chunks):

        start_idx = i * (target_duration * samplerate)
        end_idx = (i + 1) * (target_duration * samplerate)

        if end_idx > nr_of_audio_samples:
            chunk = input_audio[:, start_idx::]
            chunk_dur = chunk.shape[1] / samplerate
            if chunk_dur > 1:
                chunk = pad_or_truncate(chunk, 
                                        audio_length=(target_duration * samplerate))
                split_chunks.append(chunk)
            else:
                pass
        else:
            chunk = input_audio[:, start_idx:(end_idx+1)] # Python slicing is end exclusive
            split_chunks.append(chunk)
    return split_chunks
    