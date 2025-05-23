import pickle

# Saves the object to the working directory
def pickle_save(name_for_saved_obj, obj):
    try:
        with open(name_for_saved_obj, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Error during pickling: {ex}")
