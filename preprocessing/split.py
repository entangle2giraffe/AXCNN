import split_folders
from pathlib import Path
from os import chdir
import glob

# Directories
prep_path = Path("/home/giraffe/Desktop/Animal_CNN/preprocessing")
chdir(prep_path.parent)
data_path = Path.cwd() / 'dataset/raw-img'

print(glob.glob(str(data_path.parent)))

# Split folders
split_folders.ratio(str(data_path), output=str(data_path.parent), seed=1111)

print(glob.glob(str(data_path.parent)))
