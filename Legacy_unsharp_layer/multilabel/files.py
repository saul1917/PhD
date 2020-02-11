import sys
import os
import pickle

def save_pickle(file_path, obj):
    if os.path.exists (file_path):
        file_path = file_path + "v2"
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle)

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def clean_hidden_files(dirs):
    return filter( lambda f: not f.startswith('.'), dirs)

def get_file_list(a_dir):
    files = next(os.walk(a_dir))[2]
    return clean_hidden_files(files)
