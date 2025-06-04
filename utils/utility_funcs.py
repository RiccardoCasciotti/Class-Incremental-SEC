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
    return table_w_valid_rows["fname"].unique()

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

# For now only applies to Audioset filenames
# There's a disconnect between the filenames of the audiofiles
# and the filenames in the metafile.
# meta: filename_XXXXX; data: Yfilename
# The metafile suffix is assumed to be removed at this point
def filenames_to_txt(filenames: list, txtfile_name: str):
    appended_filenames = [("Y" + fname) for fname in filenames]
    with open(txtfile_name, 'w') as f:
        f.write("\n".join(appended_filenames))

# Manju's pad or truncate function
def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if x.size(1) <= audio_length:
        return torch.cat((x, torch.zeros(1, audio_length - x.size(1))), dim=1)
    else:
        return x[:, 0: audio_length]
    
# 

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
                                 mid_to_multihot_mapping) -> dict[str, list[int]]:
    segment_to_label_dict = {}
    valid_mids = get_rows_in_mids(table, mids)

    for idx, row in valid_mids.iterrows():
        fname = row['segment_id']
        # Edit out the start_of_clip from filename
        fname = ('_').join(fname.split("_")[0:-1])
        mid = row['mids']

        if fname not in segment_to_label_dict:
            segment_to_label_dict[fname] = [0] * len(mids)
            segment_to_label_dict[fname][mid_to_multihot_mapping[mid]] = 1
        else:
            segment_to_label_dict[fname][mid_to_multihot_mapping[mid]] = 1
    return segment_to_label_dict