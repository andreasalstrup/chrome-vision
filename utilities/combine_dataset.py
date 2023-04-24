import os
import shutil

def CombineDataset(root_path: str):
    COMBINED_PATH = os.path.join(root_path, '__combined__')
    os.mkdir(COMBINED_PATH)
    
    # Get the subdirectories in input path
    subfolders = [f.path for f in os.scandir(root_path) if os.path.isdir(f.path) and f.path != COMBINED_PATH]
    
    # For each subdirectory, copy it's contents to the combined folder
    for folder in subfolders:
        folder_files = [f.path for f in os.scandir(folder) if os.path.isfile(f.path)]
        for file in folder_files:
            shutil.copy2(file, COMBINED_PATH)