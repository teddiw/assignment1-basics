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
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab: Dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    successive_frequencies: Dict[tuple[bytes], int] = {}
    for pre_token, count in pre_token_frequencies.items():
        successive_frequencies = count_successive_tokens(pre_token, count, successive_frequencies)
    
    print(f'len of successive_frequencies: {len(successive_frequencies)}')
    print(f'len of pre_token_frequencies: {len(pre_token_frequencies)}')
    
    num_merges = vocab_size - len(special_tokens) - 256
    merges: List[tuple[bytes]] = []
    
    starting_merge_idx = 0
    successives_to_merge = get_max_successive(successive_frequencies) 
    re_use = True
    for i in tqdm(range(num_merges)):
    # for i in tqdm(range(4)): # TODO revert
        if (re_use == False):
            successives_to_merge = get_max_successive(successive_frequencies) 

        merged_bytes = successives_to_merge[0] + successives_to_merge[1]

        merges.append(successives_to_merge)
        vocab[len(vocab)] = merged_bytes
        save_dict = {}
        save_dict[successives_to_merge] = successive_frequencies[successives_to_merge]
        del successive_frequencies[successives_to_merge] # TODO might be doing this before it's last used

        next_successives_to_merge = get_max_successive(successive_frequencies) 

        successive_frequencies[successives_to_merge] = save_dict[successives_to_merge]
        # TODO re-use next_successives_to_merge if possible!
        do_update = needs_update(next_successives_to_merge, merges, successive_frequencies, starting_merge_idx)
          
        if (do_update):
            pre_token_frequencies, successive_frequencies = update_records(pre_token_frequencies, successive_frequencies, merges, starting_merge_idx)
            starting_merge_idx = len(merges)
            re_use = False
            del successive_frequencies[successives_to_merge]
        else:
            del successive_frequencies[successives_to_merge]
            successives_to_merge = next_successives_to_merge
            re_use = True
        
    profiler.disable()
    profiler.print_stats(sort='cumtime') 
    # breakpoint()
    return vocab, merges

# Is there a way we can aggregate changes we'll have to make and only make them when we have to? I.e. when the max frequency changes 
    # We want to update pre_token_frequencies when:
        # The tuple with the max frequency is decremented in successive_frequencies. This happens when one half of the merged bytes is in the max freq tuple.
            #  If the max freq tuple contains either half of the recently merged bytes, then it is possible that the max tuple frequency will decrease.
        # This is impossible: The max frequency is smaller than either of the counts of new (a, merged) and (merged, b) pairs. Then, the max frequency may no longer be the max!
            #  If the max freq is smaller than the counts of either (a, mer) or (ged, b) pairs, then it is possible that the max freq must be replaced

def needs_update(next_successives_to_merge:tuple[bytes],
                 successives_to_merge_ls:tuple[bytes],
                 successive_frequencies:Dict[tuple[bytes], int],
                 starting_merge_idx: int
                 ) -> bool:
    # TODO can use a set of the bytes about to be merged (that resets after each merge)
    for i in range(starting_merge_idx, len(successives_to_merge_ls)):
        successives_to_merge = successives_to_merge_ls[i]
        if (successives_to_merge[0] in next_successives_to_merge) or (successives_to_merge[1] in next_successives_to_merge):
            # If the next merge involves bytes in the merge queue 
            return True
        
    return False

# Updating pre_token_frequencies includes the following:
    # Merging successive bytes in each pre_token
    # Adding new (a, merged) and (merged, b) pairs to successive_frequencies
    # Decrementing counts for (c, mer) and (ged, d) in successive_frequencies     

def update_records(pre_token_frequencies: Dict[tuple[bytes], int],
                   successive_frequencies: Dict[tuple[bytes], int],
                   successives_to_merge_ls: List[tuple[bytes]],
                   starting_merge_idx: int
                   ) -> Dict[tuple[bytes], int]:
    
    successives_to_merge_bytes_ls = [b''.join(successives_to_merge) for successives_to_merge in successives_to_merge_ls]
    
    # Iterate through pre_token_frequencies to find the pre_tokens that need (1) to be merged and (2) will impact updates to successive_frequencies
    pretokens_to_update_set: Dict[tuple[bytes], int] = {} # set of pre_tokens that need to be updated
    
    for pre_token_bytes_tuple in pre_token_frequencies.keys():
        pre_token_bytes = b''.join(pre_token_bytes_tuple)
        for i in range(starting_merge_idx, len(successives_to_merge_ls)):
            successives_to_merge_bytes = successives_to_merge_bytes_ls[i]
            if (successives_to_merge_bytes in pre_token_bytes):
                pretokens_to_update_set[pre_token_bytes_tuple] = 0

    # create the updated pre_token_bytes_tuple and replace the old one in pre_token_frequencies
    # remove entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, remove AB and CD counts
    # add new entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, add ABC and BCD entries

    # Replicate the loops in my original implementation, but this time, only iterate over the pre_tokens that need to be updated
    # Update these pre_tokens for each outer loop (each merge)
    id_to_pretoken_to_update: Dict[int, tuple[bytes]] = {}
    temp_ls = list(pretokens_to_update_set.keys())
    for i in range(len(temp_ls)):
        id_to_pretoken_to_update[i] = temp_ls[i]

    for i in range(starting_merge_idx, len(successives_to_merge_ls)):
        successives_to_merge = successives_to_merge_ls[i]
        merged_bytes = successives_to_merge[0] + successives_to_merge[1]

        num_pretokens_to_update = len(id_to_pretoken_to_update)
        for j in range(num_pretokens_to_update):
            pre_token_bytes_tuple = id_to_pretoken_to_update[j]
            needs_merge = get_need_to_process(pre_token_bytes_tuple, successives_to_merge)
            if (needs_merge):
                # create the updated pre_token_bytes_tuple and replace the old one in pre_token_frequencies
                # remove entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, remove AB and CD counts

                new_pre_token_bytes_tuple, successive_frequencies = merge_successives(pre_token_bytes_tuple, successives_to_merge, successive_frequencies, pre_token_frequencies[pre_token_bytes_tuple])
                
                id_to_pretoken_to_update[j] = new_pre_token_bytes_tuple # update to the most current merge version

                # if (len(new_pre_token_bytes_tuple) == 0):
                #     del pre_token_frequencies[pre_token_bytes_tuple] # remove entry entirely if it has been fully merged
                #     return pre_token_frequencies, successive_frequencies
                
                if (pre_token_bytes_tuple != new_pre_token_bytes_tuple):
                    temp = pre_token_frequencies[pre_token_bytes_tuple]
                    pre_token_frequencies[new_pre_token_bytes_tuple] = temp
                    del pre_token_frequencies[pre_token_bytes_tuple]

                # add new entries to successive_frequencies for the pre-tokens that contain the successive bytes. I.e. for ABCD, if merging BC, add ABC and BCD entries
                successive_frequencies = debug_count_successive_tokens(new_pre_token_bytes_tuple, pre_token_frequencies[new_pre_token_bytes_tuple], successive_frequencies, merged_bytes)
    return pre_token_frequencies, successive_frequencies

# Redundant
# def apply_merges(pre_token_bytes_tuple, merge_ls):
#     for successives_to_merge in merge_ls:
#         i = 0
#         while i < len(pre_token_bytes_tuple) - 1:
#             if pre_token_bytes_tuple[i:i+2] == successives_to_merge:
#                 pre_token_bytes_tuple = pre_token_bytes_tuple[:i]+(pre_token_bytes_tuple[i]+pre_token_bytes_tuple[i+1],)+pre_token_bytes_tuple[i+2:]
#             i += 1
#     return pre_token_bytes_tuple


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
                      ) -> tuple[tuple[bytes], Dict[tuple[bytes], int]]:
    # function to find and merge all of the successive bytes in a tuple of bytes
    # also removes counts of successive candidates from successive_frequencies that no longer exist
    i = 0
    while i < len(pre_token_bytes_tuple) - 1:
        if pre_token_bytes_tuple[i:i+2] == successives_to_merge:
            if (i > 0):
                if (pre_token_bytes_tuple[i-1:i+1] in successive_frequencies.keys()): # rules out the case where the most recently merged bytes precede the next bytes to merge. E.g. inging when merging (b'in', b'g') --> (ing in) g --> (ing in) does not exist in successive_frequencies yet (it gets added in count_successive_tokens)
                    successive_frequencies[pre_token_bytes_tuple[i-1:i+1]] -= counts

            if (pre_token_bytes_tuple[i+1:i+3] in successive_frequencies.keys()) and (i < len(pre_token_bytes_tuple) - 2): # TODO revert away the first clause of the if statement # (pre_token_bytes_tuple[i+1:i+3] in successive_frequencies.keys()) and 
                successive_frequencies[pre_token_bytes_tuple[i+1:i+3]] -= counts

            pre_token_bytes_tuple = pre_token_bytes_tuple[:i]+(pre_token_bytes_tuple[i]+pre_token_bytes_tuple[i+1],)+pre_token_bytes_tuple[i+2:]
        i += 1
    return pre_token_bytes_tuple, successive_frequencies

def count_successive_tokens(pre_token_bytes_tuple: tuple[bytes], 
                            count: int,
                            successive_frequencies: Dict[tuple[bytes], int],
                            target_token: bytes = None) -> Dict[tuple[bytes], int]:
                            
    # TODO maybe update values of the tokens once per the outer loop? (check if that is ok for correctness)
    # TODO the threads are blocking a lot... probably because they're waiting to access successive_frequencies
    # function to count all of the successive tokens in a tuple of bytes and record the frequencies in successive_frequencies
    for i in range(len(pre_token_bytes_tuple) - 1):
        successive_candidates = pre_token_bytes_tuple[i:i+2]
        if (not target_token or target_token in successive_candidates): 
            successive_frequencies[successive_candidates] = successive_frequencies.get(successive_candidates, 0) + count

    return successive_frequencies

def debug_count_successive_tokens(pre_token_bytes_tuple: tuple[bytes], 
                            count: int,
                            successive_frequencies: Dict[tuple[bytes], int],
                            target_token: bytes = None) -> Dict[tuple[bytes], int]:

    # function to count all of the successive tokens in a tuple of bytes and record the frequencies in successive_frequencies
    for i in range(len(pre_token_bytes_tuple) - 1):
        successive_candidates = pre_token_bytes_tuple[i:i+2]
        if (not target_token or target_token in successive_candidates): 
            successive_frequencies[successive_candidates] = successive_frequencies.get(successive_candidates, 0) + count
            pass

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
    data_str = 'owt_valid'
    time1 = time.time()
    if (data_str == 'ts_valid'): # 22502601 bytes (.023 GB) # len of successive_frequencies: 932 # len of pre_token_frequencies: 13125
        input_path= "../data/TinyStoriesV2-GPT4-valid.txt" 
        vocab_size = 10000
    elif (data_str == 'ts_train'): # 2227753162 bytes (2.23 GB)
        input_path= "../data/TinyStoriesV2-GPT4-train.txt" 
        vocab_size = 10000 
    elif(data_str == 'owt_train'): # 11920511059 bytes (11.92 GB) # len of successive_frequencies: 19593 # len of pre_token_frequencies: 6602749
        input_path= "../data/owt_train.txt"
        vocab_size = 32000
    elif(data_str == 'owt_valid'): # 289998753 bytes (.29 GB) # len of successive_frequencies: 11852 # len of pre_token_frequencies: 627577
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

