from typing import Dict, List, Tuple
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_on_ST(text, special_tokens):
    pattern = '|'.join(map(re.escape, special_tokens))
    text_split_by_ST = re.split(pattern, text)
    return [x for x in text_split_by_ST if x != '']  

def pretokenize_and_count_frequencies(text, special_tokens): 
    for special_token in special_tokens:
        text = text.replace(special_token, "") 
    text_ls = split_on_ST(text, special_tokens)
    pre_token_frequencies: Dict[tuple[bytes], int] = {} 
    for text in text_ls:
        for pre_token in re.finditer(PAT, text):
            pre_token_bytes = pre_token.group().encode("utf-8")
            pre_token_bytes_tuple = tuple(bytes([byte]) for byte in pre_token_bytes)
            pre_token_frequencies[pre_token_bytes_tuple] = pre_token_frequencies.get(pre_token_bytes_tuple, 0) + 1
    return pre_token_frequencies 