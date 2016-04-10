"""compare the MAT version and HDF5 version"""

import h5py
import numpy as np
import os.path
import scipy.io as sio
import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print(self.name)
        print('Elapsed: {}'.format(time.time() - self.tstart))


def main():
    file_handle = h5py.File('vanHateren.hdf5', 'r+')
    imc_dataset = file_handle['imc_data']
    iml_dataset = file_handle['iml_data']

    for chunk_idx in range(42):
        with Timer('read hdf5'):
            if chunk_idx < 41:
                imc_subset_hdf5 = imc_dataset[:, :, chunk_idx * 100:(chunk_idx + 1) * 100]
                iml_subset_hdf5 = iml_dataset[:, :, chunk_idx * 100:(chunk_idx + 1) * 100]
            else:
                imc_subset_hdf5 = imc_dataset[:, :, chunk_idx * 100:]
                iml_subset_hdf5 = iml_dataset[:, :, chunk_idx * 100:]
        with Timer('read mat'):
            iml_subset_mat = sio.loadmat(os.path.join('/home/leelab_share/datasets/vanHateren_iml_MATLAB',
                                                      'vanHaterenIML_{:02d}.mat'.format(chunk_idx + 1)))[
                'vanHaterenIML'][
                0, 0]['images']
            imc_subset_mat = sio.loadmat(os.path.join('/home/leelab_share/datasets/vanHateren_imc_MATLAB',
                                                      'vanHaterenIMC_{:02d}.mat'.format(chunk_idx + 1)))[
                'vanHaterenIMC'][
                0, 0]['images']
        assert np.array_equal(iml_subset_mat, iml_subset_hdf5)
        assert np.array_equal(imc_subset_mat, imc_subset_hdf5)
        print(chunk_idx)

    file_handle.close()


if __name__ == '__main__':
    main()
