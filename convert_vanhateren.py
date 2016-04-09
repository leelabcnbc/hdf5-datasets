"""create a single HDF5 dataset for all vanHateren images"""

import h5py
import numpy as np
import os.path


def add_vanhateren_subset(h5handle: h5py.File, prefix, imglist, imgbase):
    length_filename = len(imglist[0])
    for imgname in imglist:
        assert len(imgname) == length_filename
    imglist_dtype = 'S' + str(length_filename)

    data_handle = h5handle.create_dataset(prefix + '_data', shape=(1024, 1532, len(imglist)),
                                          dtype=np.uint16, chunks=(1024, 1532, 1),
                                          compression='gzip', shuffle=True, fletcher32=True)
    h5handle.create_dataset(prefix + '_filelist', shape=(len(imglist),),
                            dtype=imglist_dtype, data=np.array(imglist, dtype=imglist_dtype))

    for idx, imgname in enumerate(imglist):
        with open(os.path.join(imgbase, imgname), 'rb') as f:
            imgarray = np.fromfile(f, dtype='>u2').reshape((1024, 1536))
            assert imgarray.size == 1024 * 1536
            imgarray = imgarray[:, 2:-2]  # remove black stuff.
            data_handle[:, :, idx] = imgarray  # endian ness will be handled by h5py.
        print(idx)


def main():
    with open('imc_list.txt', 'rt') as f:
        imc_imglist = f.read().splitlines()
    with open('iml_list.txt', 'rt') as f:
        iml_imglist = f.read().splitlines()

    file_handle = h5py.File('vanHateren.hdf5', 'w-')
    add_vanhateren_subset(file_handle, 'imc', imc_imglist, '/home/leelab_share/datasets/vanHateren/vanhateren_imc')
    add_vanhateren_subset(file_handle, 'iml', iml_imglist, '/home/leelab_share/datasets/vanHateren/vanhateren_iml')
    file_handle.close()


if __name__ == '__main__':
    main()
