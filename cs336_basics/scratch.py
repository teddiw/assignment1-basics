import multiprocessing as mp
import concurrent.futures
import regex as re
import os
import tqdm
import time
import pickle


def main():
    vocab_files = ['ts_train_vocab.pickle', 'TinyStoriesV2_vocab.pickle']  # Add other vocab files as needed

    vocabularies = {}
    vocabularies_i = 0
    for vocab_file in vocab_files:
        with open(vocab_file, 'rb') as f:
            vocabularies[vocabularies_i] = pickle.load(f)
        vocabularies_i += 1
    assert len(vocabularies[0]) == len(vocabularies[1])

    longest_token0 = ""
    longest_token1 = ""
    for i in range(len(vocabularies[0])):
        token0 = vocabularies[0][i]
        token1 = vocabularies[1][i]
        if len(token0) > len(longest_token0):
            longest_token0 = token0
        if len(token1) > len(longest_token1):
            longest_token1 = token1
    
    print(longest_token0)
    print(longest_token1)
    breakpoint()


if __name__ == '__main__':
    main()
