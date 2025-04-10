import regex as re
from typing import List, Dict
from cs336_basics.train_bpe_tokenizer_fast import _train_bpe
from cs336_basics.helpers import pretokenize_and_count_frequencies

def train_bpe(input_path: str,
                 vocab_size: int,
                 special_tokens: List[str] = []):

    # get the size of the file
    with open(input_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()

    pre_token_frequencies: Dict[tuple[bytes], int] = pretokenize_and_count_frequencies(input_path, 0, size, special_tokens)

    return _train_bpe(pre_token_frequencies, vocab_size, special_tokens)