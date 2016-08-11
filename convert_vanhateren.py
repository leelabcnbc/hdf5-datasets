"""create a single HDF5 dataset for all vanHateren images"""

import h5py
import numpy as np
import os.path
from sys import argv


def add_vanhateren_subset(h5handle: h5py.File, prefix, imglist, imgbase):
    length_filename = len(imglist[0])
    for imgname in imglist:
        assert len(imgname) == length_filename
    imglist_dtype = 'S' + str(length_filename)

    imgarray_all = []
    for idx, imgname in enumerate(imglist):
        with open(os.path.join(imgbase, imgname), 'rb') as f:
            imgarray = np.fromfile(f, dtype='>u2').reshape((1024, 1536))
            assert imgarray.size == 1024 * 1536
            imgarray = imgarray[:, 2:-2]  # remove black stuff.
            imgarray_all.append(imgarray)
        if idx % 100 == 0:
            print(idx)
    data_to_write = np.asarray(imgarray_all)
    print(data_to_write.shape, data_to_write.dtype)
    h5handle.create_dataset(prefix + '_data', data=data_to_write,
                            chunks=(1, 256, 1532),  # basically, make each chunk 1/4 of an image.
                            compression='gzip', shuffle=True, fletcher32=True)
    h5handle.create_dataset(prefix + '_filelist', data=np.array(imglist, dtype=imglist_dtype))


def main():
    numarg = len(argv)
    if numarg == 1:
        dataset_root = '/data2/leelab/standard_datasets'
    elif numarg == 2:
        dataset_root = argv[1]
    else:
        raise RuntimeError('either supply dataset root dir or not!')

    with open('imc_list.txt', 'rt') as f:
        imc_imglist = f.read().splitlines()
    with open('iml_list.txt', 'rt') as f:
        iml_imglist = f.read().splitlines()

    file_handle = h5py.File('results/vanHateren.hdf5', 'w-')
    add_vanhateren_subset(file_handle, 'imc', imc_imglist, os.path.join(dataset_root, 'vanHateren/vanhateren_imc'))
    add_vanhateren_subset(file_handle, 'iml', iml_imglist, os.path.join(dataset_root, 'vanHateren/vanhateren_iml'))
    file_handle.close()


if __name__ == '__main__':
    main()
