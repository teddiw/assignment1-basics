import regex as re
from typing import List, Tuple, Dict
import cProfile
import multiprocessing as mp 
import time
import pickle 
from helpers import pretokenize_and_count_frequencies
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _train_bpe(pre_token_frequencies: Dict[tuple[bytes], int],
                 vocab_size: int,
                 special_tokens: List[str] = []):
    
    vocab: Dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    successive_frequencies: Dict[tuple[bytes], int] = {}
    for pre_token, count in pre_token_frequencies.items():
        successive_frequencies = count_successive_tokens(pre_token, count, successive_frequencies)

    num_merges = vocab_size - len(special_tokens) - 256
    merges: List[tuple[bytes]] = []
    for i in range(num_merges):

        # pick the successive_candidates entry with the highest frequency
        maxValue = max(successive_frequencies.values())
        successives_to_merge = max([k for k, v in successive_frequencies.items() if v == maxValue])
        merged_bytes = successives_to_merge[0] + successives_to_merge[1]

        merges.append(successives_to_merge)
        vocab[len(vocab)] = merged_bytes
        
        pre_token_ls = list(pre_token_frequencies.keys())
        for pre_token_bytes_tuple in pre_token_ls:
            need_to_process = False
            i = 0
            while i < len(pre_token_bytes_tuple) - 1: 
                if pre_token_bytes_tuple[i:i+2] == successives_to_merge:
                    need_to_process = True
                    break
                i += 1
            if need_to_process:
                # create the updated pre_token_bytes_tuple and replace the old one in pre_token_frequencies
                # remove entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, remove AB and CD counts
                
                new_pre_token_bytes_tuple, successive_frequencies = merge_successives(pre_token_bytes_tuple, successives_to_merge, successive_frequencies, pre_token_frequencies[pre_token_bytes_tuple], pre_token_bytes_tuple)
                pre_token_frequencies[new_pre_token_bytes_tuple] = pre_token_frequencies[pre_token_bytes_tuple]
                del pre_token_frequencies[pre_token_bytes_tuple]

                # add new entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, add ABC and BCD entries
                successive_frequencies = count_successive_tokens(new_pre_token_bytes_tuple, pre_token_frequencies[new_pre_token_bytes_tuple], successive_frequencies, merged_bytes)
        
        del successive_frequencies[successives_to_merge]
    return vocab, merges

def merge_successives(pre_token_bytes_tuple: tuple[bytes], 
                      successives_to_merge: tuple[bytes],
                      successive_frequencies: Dict[tuple[bytes], int],
                      counts: int,
                      target_token: bytes = None
                      ) -> tuple[tuple[bytes], Dict[tuple[bytes], int]]:
    # function to find and merge all of the successive bytes in a tuple of bytes
    # also removes counts of successive candidates from successive_frequencies that no longer exist
    i = 0
    while i < len(pre_token_bytes_tuple) - 1:
        if pre_token_bytes_tuple[i:i+2] == successives_to_merge:
            if (i > 0):
                if (pre_token_bytes_tuple[i-1:i+1] in successive_frequencies.keys()): # rules out the case where the most recently merged bytes precede the next bytes to merge. E.g. inging when merging (b'in', b'g') --> (ing in) g --> (ing in) does not exist in successive_frequencies yet (it gets added in count_successive_tokens)
                    successive_frequencies[pre_token_bytes_tuple[i-1:i+1]] -= counts

            if (i < len(pre_token_bytes_tuple) - 2):
                successive_frequencies[pre_token_bytes_tuple[i+1:i+3]] -= counts

            pre_token_bytes_tuple = pre_token_bytes_tuple[:i]+(pre_token_bytes_tuple[i]+pre_token_bytes_tuple[i+1],)+pre_token_bytes_tuple[i+2:]
        i += 1
    return pre_token_bytes_tuple, successive_frequencies

def count_successive_tokens(pre_token_bytes_tuple: tuple[bytes], 
                            count: int,
                            successive_frequencies: Dict[tuple[bytes], int],
                            target_token: bytes = None) -> Dict[tuple[bytes], int]:
                            
    
    # function to count all of the successive tokens in a tuple of bytes and record the frequencies in successive_frequencies
    for i in range(len(pre_token_bytes_tuple) - 1):
        successive_candidates = pre_token_bytes_tuple[i:i+2]
        if (not target_token or target_token in successive_candidates):
            successive_frequencies[successive_candidates] = successive_frequencies.get(successive_candidates, 0) + count

    return successive_frequencies

def combine_pre_token_frequencies(pre_token_frequencies_list: List[Dict[tuple[bytes], int]]) -> Dict[tuple[bytes], int]:
    # function to combine the pre_token_frequencies dictionaries from each process
    combined_pre_token_frequencies = {}
    for pre_token_frequencies in pre_token_frequencies_list:
        for pre_token, count in pre_token_frequencies.items():
            combined_pre_token_frequencies[pre_token] = combined_pre_token_frequencies.get(pre_token, 0) + count
    return combined_pre_token_frequencies

def read_data(filename):
    with open(filename) as file:
        document = ""
        for line in file:
            if ('<|endoftext|>' in line):
                line_components = line.split('<|endoftext|>')
                document += line_components[0]
                yield document
                document = line_components[1]
            else:
                document += line

if __name__ == "__main__":
    data_str = 'owt_train'
    time1 = time.time()
    if (data_str == 'ts_valid'):
        input_path= "/nlp/scr/worledge/data/TinyStoriesV2-GPT4-valid.txt" 
        vocab_size = 10000
    elif (data_str == 'ts_train'):
        input_path= "../data/TinyStoriesV2-GPT4-train.txt" 
        vocab_size = 10000
    elif(data_str == 'owt_train'):
        input_path= "../data/owt_train.txt"
        vocab_size = 32000
    else:
        print('Invalid data_str. Exiting...')
        exit(1)

    special_tokens = ['<|endoftext|>']

    profiler = cProfile.Profile()
    profiler.enable()

    # with open(input_path, "r", encoding="utf-8") as f:
    #     text = f.read()

    data_iterator = read_data(input_path)
    print('Created data iterator. Collecting results from processes...')
    # pre_token_frequencies = pretokenize(text)
    # documents = [d for d in text.split("<|endoftext|>")]

    pre_token_frequencies = {}
    def log_pre_token_frequencies(result: Dict[tuple[bytes], int]) -> Dict[tuple[bytes], int]:
        # This is called whenever the pool returns a result.
        # result_list is modified only by the main process, not the pool workers.
        for pre_token, count in result.items():
            pre_token_frequencies[pre_token] = pre_token_frequencies.get(pre_token, 0) + count
        return pre_token_frequencies

    pool = mp.Pool(processes=8)
    for x in tqdm(data_iterator):
        pool.apply_async(pretokenize_and_count_frequencies, args = (x,special_tokens), callback = log_pre_token_frequencies)
    pool.close()
    pool.join()

    # pool = mp.Pool(processes=8)
    # results = [pool.apply_async(pretokenize, args=(x,special_tokens)) for x in data_iterator]
    # print('Collecting results from processes...')
    # result_ls = [r.get() for r in results]
    # pre_token_frequencies = combine_pre_token_frequencies(result_ls)
    print('Collected! Training BPE...')
    
    vocab, merges = _train_bpe(
        pre_token_frequencies,
        vocab_size=vocab_size, 
        special_tokens=special_tokens)
    print('Done.')
    
    with open(data_str+'_vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_str+'_merges.pickle', 'wb') as handle:
        pickle.dump(merges, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Total time: ", time.time() - time1)
    profiler.disable()
    profiler.print_stats(sort='cumtime')    
