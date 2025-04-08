import regex as re
from typing import List, Dict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_on_ST(text, special_tokens):
    pattern = '|'.join(map(re.escape, special_tokens))
    text_split_by_ST = re.split(pattern, text)
    return [x for x in text_split_by_ST if x != '']  

def train_bpe(input_path: str,
                 vocab_size: int,
                 special_tokens: List[str] = []):

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pre_token_frequencies: Dict[tuple[bytes], int] = {}
    # for pre_token in re.finditer(PAT, text):
    #     pre_token_bytes = pre_token.group().encode("utf-8")
    #     pre_token_bytes_tuple = tuple(bytes([byte]) for byte in pre_token_bytes)
    #     pre_token_frequencies[pre_token_bytes_tuple] = pre_token_frequencies.get(pre_token_bytes_tuple, 0) + 1

    vocab: Dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
        # if (special_token != "<|endoftext|>"): # we remove <|endoftext|> when we break the text into documents
        #     text = text.replace(special_token, "") 
    
    text_ls = split_on_ST(text, special_tokens)

    for text_fragment in text_ls:
        for pre_token in re.finditer(PAT, text_fragment):
            pre_token_bytes = pre_token.group().encode("utf-8")
            pre_token_bytes_tuple = tuple(bytes([byte]) for byte in pre_token_bytes)
            pre_token_frequencies[pre_token_bytes_tuple] = pre_token_frequencies.get(pre_token_bytes_tuple, 0) + 1
    
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
                
                new_pre_token_bytes_tuple, successive_frequencies = merge_successives(pre_token_bytes_tuple, successives_to_merge, successive_frequencies, pre_token_frequencies[pre_token_bytes_tuple])
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
