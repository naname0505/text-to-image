import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='Data/sample_captions.txt',
                       help='caption file')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data Directory')
    parser.add_argument('--data_set', type=str, default='flowers',help="Define the name of data sets")

    args = parser.parse_args()
    _n_labels = 4096
    if args.data_set == "ImageNet":
        with open("./Data/sample_caption_ImageNet.txt") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap)>0]
        caption_vector_list = []
        for cap in captions:
            _n = cap.split(',')
            _zeros0 = np.zeros(_n_labels)
            _zeros1 = np.zeros(_n_labels)
            _zeros0[int(_n[0])] = 1
            _zeros1[int(_n[0])] = 1
            _onehot = np.concatenate([_zeros0,_zeros1],axis=0)
            caption_vector_list.append(_onehot)
        print(len(caption_vector_list), len(caption_vector_list[0]))
        h = h5py.File(join(args.data_dir, 'sample_caption_ImageNet.hdf5'))
        h.create_dataset('vectors', data=caption_vector_list)        
        h.close()
 


    if args.data_set == "flowers":
        with open( args.caption_file ) as f:
            captions = f.read().split('\n')

        captions = [cap for cap in captions if len(cap) > 0]
        print(captions)
        model = skipthoughts.load_model()
        caption_vectors = skipthoughts.encode(model, captions)
        print(caption_vectors)
        print(caption_vectors.shape, len(caption_vectors[0]))

        if os.path.isfile(join(args.data_dir, 'sample_caption_vectors.hdf5')):
            os.remove(join(args.data_dir, 'sample_caption_vectors.hdf5'))
        h = h5py.File(join(args.data_dir, 'sample_caption_vectors.hdf5'))
        h.create_dataset('vectors', data=caption_vectors)        
        h.close()

if __name__ == '__main__':
    main()
