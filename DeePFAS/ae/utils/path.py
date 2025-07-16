import os

def get_absolute_path(relative_path):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(current_file_dir, relative_path)
    normalized_path = os.path.normpath(absolute_path)
    
    return normalized_path

if __name__ == '__main__':
    print(get_absolute_path('../model/beam.py'))