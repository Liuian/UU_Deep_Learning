import h5py
def get_dataset_name( file_name_with_dir ) :
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    return dataset_name
filename_path = "Final Project data/Intra/train/rest_105923_1.h5"

with h5py.File(filename_path, "r") as f:
    dataset_name = get_dataset_name(filename_path)
    matrix = f.get(dataset_name)[()]
    print(type(matrix))
    print(matrix.shape)
