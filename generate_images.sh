#!/bin/sh
python generate_images.py --data_set="ImageNet" \
                         --caption_vector_length=8192 \
                         --model_path="Data/Models/200_8192CapVecDims_ImageNet_model.ckpt" \
                         --caption_thought_vectors="./Data/sample_caption_ImageNet.hdf5"
    
