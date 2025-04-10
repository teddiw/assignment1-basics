import regex as re
from typing import List, Tuple, Dict
import cProfile
import multiprocessing as mp 
import time
import pickle 
from cs336_basics.helpers import pretokenize_and_count_frequencies
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

        successives_to_merge = get_max_successive(successive_frequencies) 

        merged_bytes = successives_to_merge[0] + successives_to_merge[1]

        merges.append(successives_to_merge)
        vocab[len(vocab)] = merged_bytes
        
        # iterate through to find the pre_tokens that contain the successive bytes
        pre_token_ls = list(pre_token_frequencies.keys())
        for pre_token_bytes_tuple in pre_token_ls:
            need_to_process = False
            if (successives_to_merge[0] in pre_token_bytes_tuple) and (successives_to_merge[1] in pre_token_bytes_tuple):
                need_to_process = get_need_to_process(pre_token_bytes_tuple, successives_to_merge) 
            
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

def get_need_to_process(pre_token_bytes_tuple: tuple[bytes],
                       successives_to_merge: tuple[bytes]) -> bool:
    # function to check if the successive bytes are in the pre_token_bytes_tuple
    need_to_process = False
    i = 0
    while i < len(pre_token_bytes_tuple) - 1: 
        if pre_token_bytes_tuple[i:i+2] == successives_to_merge:
            need_to_process = True
            break
        i += 1
    return need_to_process

def get_max_successive(successive_frequencies: Dict[tuple[bytes], int]) -> tuple[bytes]:
    maxValue = max(successive_frequencies.values())
    successives_to_merge = max([k for k, v in successive_frequencies.items() if v == maxValue])
    return successives_to_merge

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

def read_data(filename, special_tokens):
    num_lines_per_chunk = 4096 # this is arbitrary
    with open(filename, 'r') as file:
        document_chunk = ""
        pos = file.tell()
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell() 
        file.seek(pos)
        while pos < file_size:
            i = 0
            while (i < num_lines_per_chunk) or ((i >= num_lines_per_chunk) and ('<|endoftext|>' not in document_chunk)): # ensures that <|endoftext|> is in the chunk
                line = file.readline()
                if (line == ''):  # if eof is reached
                    if (len(document_chunk) > 0):
                        yield (filename, pos, len(document_chunk.encode('utf-8')), special_tokens)
                    break
                else: 
                    document_chunk += line
                i += 1
            if (line == ''):
                break
            document_chunk_splits = document_chunk.split('<|endoftext|>')
            curr_document_chunk = '<|endoftext|>'.join(document_chunk_splits[:-1])+'<|endoftext|>'
            document_chunk = document_chunk_splits[-1] # start of the next document_chunk
            next_pos = file.tell() - len(document_chunk.encode('utf-8')) 
            size = len(curr_document_chunk.encode('utf-8'))
            yield (filename, pos, size, special_tokens)
            pos = next_pos

if __name__ == "__main__":
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
    special_tokens = ['<|endoftext|>']

    data_iterator = read_data(input_path, special_tokens)
    print('Created data iterator. Collecting results from processes...')

    pre_token_frequencies = {}
    def log_pre_token_frequencies(result: Dict[tuple[bytes], int]) -> Dict[tuple[bytes], int]: 
        # This is called whenever the pool returns a result.
        # result_list is modified only by the main process, not the pool workers.
        for pre_token, count in result.items():
            pre_token_frequencies[pre_token] = pre_token_frequencies.get(pre_token, 0) + count
        return pre_token_frequencies

    with mp.Pool(processes=8)  as pool:
        for document in tqdm(data_iterator): # send over the file and the byte indices in larger chunks. The use seek to get the text.
            pool.apply_async(pretokenize_and_count_frequencies, args = document, callback=log_pre_token_frequencies)
        pool.close()
        pool.join()

    print('Took this many seconds to get pre_token_frequencies', time.time() - time1)
    with open(t_results_fp+data_str+f'_pre_token_frequencies_{vocab_size}.pickle', 'wb') as handle:
        pickle.dump(pre_token_frequencies, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('Collected! Training BPE...')
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    vocab, merges = _train_bpe(
        pre_token_frequencies,
        vocab_size=vocab_size, 
        special_tokens=special_tokens)
    # profiler.disable()
    # profiler.print_stats(sort='cumtime')   
    print('Done.')
    
    with open(t_results_fp+data_str+f'_vocab_newP_{vocab_size}.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(t_results_fp+data_str+f'_merges_newP_{vocab_size}.pickle', 'wb') as handle:
        pickle.dump(merges, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Total time: ", time.time() - time1) 

