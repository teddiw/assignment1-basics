from typing import Dict, List, Tuple
import regex as re
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_and_count_frequencies(filename, pos, size, special_tokens): 
    text = get_text_from_file(filename, pos, size)

    # for special_token in special_tokens:
    #     text = text.replace(special_token, "") 

    text_ls = split_on_ST(text, special_tokens)
    pre_token_frequencies: Dict[tuple[bytes], int] = {} 
    for text_fragment in text_ls:
        for pre_token in re.finditer(PAT, text_fragment):
            pre_token_bytes = pre_token.group().encode("utf-8")
            pre_token_bytes_tuple = tuple(bytes([byte]) for byte in pre_token_bytes)
            pre_token_frequencies[pre_token_bytes_tuple] = pre_token_frequencies.get(pre_token_bytes_tuple, 0) + 1
    return pre_token_frequencies

    # pre_token_frequencies: Dict[tuple[bytes], int] = {}
        
    # text_ls = split_on_ST(text, special_tokens)

    # for text_fragment in text_ls:
    #     for pre_token in re.finditer(PAT, text_fragment):
    #         pre_token_bytes = pre_token.group().encode("utf-8")
    #         pre_token_bytes_tuple = tuple(bytes([byte]) for byte in pre_token_bytes)
    #         pre_token_frequencies[pre_token_bytes_tuple] = pre_token_frequencies.get(pre_token_bytes_tuple, 0) + 1

# TODO add memory profiling for get_text_from_file
# TODO add memory profiling for the BPE algorithm 
def get_text_from_file(filename, pos, size):
     with open(filename, 'r') as file:
        file.seek(pos)
        text = file.read(size)
        return text
     
    # # read the file in chunks
    # text = ''
    # with open(filename, 'r') as file:
    #     # TODO try reading line by line
    #     file.seek(pos)    
    #     line = file.readline()
    #     line_bytes = line.encode('utf-8')
    #     proposed_text_size = len(line_bytes) + len(text.encode('utf-8'))
    #     if (proposed_text_size <= size):
    #         text += line
    #     else:
    #         text += line_bytes[:-(proposed_text_size - size)].encode('utf-8')

def split_on_ST(text, special_tokens):
    # handle overlapping special tokens
    # does not include special tokens in the result
    texts = [text]
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    for special_token in ordered_special_tokens:
        i = len(texts) - 1
        while i >= 0:
            curr_text = texts[i]
            new_texts = []
            if (curr_text not in ordered_special_tokens):
                if (special_token in curr_text):
                    split_text = curr_text.split(special_token)
                    new_texts.extend(split_text) 
                else: 
                    new_texts.append(curr_text)
            else:
                new_texts.append(curr_text)
            texts = texts[:i] + new_texts + texts[i+1:]
            i -= 1
    return [x for x in texts if x != ''] 
