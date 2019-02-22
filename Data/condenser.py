# ODO example code
# concatenates several groups from pytables .h5 files into one group in one file

# Installation
# conda install odo
# Or
# pip install odo

# Imports
import tables
from odo import odo, discover, append

# Register 'append' function for custom Odo backend
@append.register(tables.file.File, tables.group.Group)
def append_group_to_root(tgt, src, **kwargs):
	if "/images" in tgt:
		group = tgt.root.images
	else:
		group = tgt.create_group(tgt.root, 'images', 'Arrays of image information')
	src._f_copy_children(group, recursive=True)
	return tgt

# Create new .h5 file to store results in
newH5file = tables.open_file("combined_data.h5", mode="w", title="Combined Training Data")

# Code to combine separate .h5 files into one file
files = ['data_1.h5', 'data_2.h5', 'data_3.h5']
array_names = []

for file_name in files:
	file = tables.open_file("llnl_data_sets/data/" + file_name, mode="r")
	odo(file.root.images, newH5file)

# Print the basic file structure and close file
print(newH5file)
newH5file.close()
