import multiprocessing as mp
import concurrent.futures
import regex as re
import os
import tqdm
import time
import pickle
import heapq


def main():
    data_str = 'ts_train'
    time1 = time.time()
    if (data_str == 'ts_valid'): # 22502601 bytes (.023 GB)
        input_path= "../data/TinyStoriesV2-GPT4-valid.txt" 
        vocab_size = 10000
    elif (data_str == 'ts_train'): # 2227753162 bytes (2.23 GB)
        input_path= "../data/TinyStoriesV2-GPT4-train.txt" 
        vocab_size = 10000 
    elif(data_str == 'owt_train'): # 11920511059 bytes (11.92 GB)
        input_path= "../data/owt_train.txt"
        vocab_size = 32000
    elif(data_str == 'owt_valid'): # 289998753 bytes (.29 GB)
        input_path= "../data/owt_valid.txt"
        vocab_size = 32000
    else:
        print('Invalid data_str. Exiting...')
        exit(1)

    t_results_fp = "t_results/"
        
    with open(t_results_fp+data_str+f'_vocab_newP_{vocab_size}.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    
    longest_value = ''
    for key, value in vocab.items():
        if (len(longest_value) < len(value)):
            longest_value = value
    
    print('longest token is:', longest_value.decode())
    breakpoint()

if __name__ == '__main__':
    main()
