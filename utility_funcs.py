import pickle
import pandas as pd

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
    return table_w_valid_rows["segment_id"].unique()

def get_valid_fsd50k_filenames_for_mids(pd_fsd_table: pd.DataFrame, mids):

    table_w_valid_rows = pd_fsd_table[pd_fsd_table.mids.isin(mids)]
    return table_w_valid_rows["fname"].unique()