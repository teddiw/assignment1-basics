import regex as re
from typing import List, Dict, Iterable, Iterator, Tuple
import pickle 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[tuple[bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_to_id = {}
        for key, value in vocab.items():
            self.vocab_to_id[value] = key

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None =None):
        with open(vocab_filepath, 'rb') as handle:
            vocab = pickle.load(handle)
        with open(merges_filepath, 'rb') as handle:
            merges = pickle.load(handle)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        return self._encode(text)
    
    def encode_iterable(self, file: Iterable[str]) -> Iterator[int]:
        for line in read_data_for_tokenization(file):
            ids =  self._encode(line)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        decoded_text = b''
        for token_id in ids:
            decoded_text += self.vocab[token_id]
        return bytes(decoded_text).decode("utf-8", errors='replace')
    
    # TODO fix split_on_ST in helpers
    # def _split_keep_delimiters(self, text):
    #     # TODO order special tokens by length (longest to shortest)
    #     # TODO split on the special tokens and keep the delimiters
    #     # TODO append the results together into one list 
    #     texts = [text]
    #     ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
    #     for special_token in ordered_special_tokens:
    #         pattern = '|'.join(map(re.escape, [special_token]))
    #         new_texts = []
    #         for text in texts:
    #             new_texts.extend([x for x in re.split(f'({pattern})', text) if x != '']) 
    #         texts = new_texts
    #     return texts
    
    # TODO fix split_on_ST in helpers
    # def _split_keep_delimiters(self, text):
    #     # TODO order special tokens by length (longest to shortest)
    #     # TODO split on the special tokens and keep the delimiters
    #     # TODO append the results together into one list 
    #     texts = [text]
    #     ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
    #     for special_token in ordered_special_tokens:
    #         pattern = '|'.join(map(re.escape, [special_token]))
    #         new_texts = []
    #         for text in texts:
    #             new_texts.extend([x for x in re.split(f'({pattern})', text) if x != '']) 
    #         texts = new_texts
    #     return texts
    
    def _split_keep_delimiters(self, text):
        # order special tokens by length (longest to shortest)
        # split on the special tokens and keep the delimiters
        # append the results together into one list 
        texts = [text]
        ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        for special_token in ordered_special_tokens:
            i = len(texts) - 1
            while i >= 0:
                curr_text = texts[i]
                new_texts = []
                if (curr_text not in ordered_special_tokens):
                    if (special_token in curr_text):
                        split_text = curr_text.split(special_token)
                        split_text_with_delimiter = []
                        for j in range(len(split_text)-1):
                            split_text_with_delimiter.append(split_text[j])
                            split_text_with_delimiter.append(special_token)
                        split_text_with_delimiter.append(split_text[-1])
                        new_texts.extend(split_text_with_delimiter) 
                    else: 
                        new_texts.append(curr_text)
                else:
                    new_texts.append(curr_text)

                texts = texts[:i] + new_texts + texts[i+1:]
                i -= 1
        return [x for x in texts if x != ''] 

    def _encode(self, text): 
        """Encodes the text using the merges and vocab_to_id. The text may contain special tokens."""
        # Split x on the special tokens but remember the special tokens and re-insert them later
        text_split_by_ST = self._split_keep_delimiters(text) # an ordered list of the strings and special tokens in x
        all_tokenized_text = []
        all_token_ids = []
        for i in range(len(text_split_by_ST)):
            text_fragment = text_split_by_ST[i]
            if (text_fragment in self.special_tokens):
                all_tokenized_text.append(text_fragment)
                all_token_ids.append(self.vocab_to_id[text_fragment.encode("utf-8")])
            else:
                tokenized_text, token_ids = self._encode_nonspecial_text(text_fragment)
                all_tokenized_text.extend(tokenized_text)
                all_token_ids.extend(token_ids)
        # text is a document or fragment of a document
        return all_token_ids

    def _merge_in_pretoken_tuple(self, merge: List[bytes], pretoken_tuple: Tuple[bytes]):
        """Merges the two bytes in the pretoken tuple if they are adjacent."""
        i = 0
        while i < len(pretoken_tuple) - 1:
            pretoken_merge_candidate = pretoken_tuple[i:i+2]
            if (pretoken_merge_candidate[0] == merge[0]) and (pretoken_merge_candidate[1] == merge[1]):
                pretoken_tuple = pretoken_tuple[:i] + (pretoken_tuple[i] + pretoken_tuple[i+1],) + pretoken_tuple[i+2:]
            i += 1
        return pretoken_tuple

    def _encode_nonspecial_text(self, text):
        """Encodes the text using the merges and vocab_to_id. The text does not contain special tokens."""
        pretokens = re.findall(PAT, text) 
        tokenized_text = []
        token_ids = []
        for pretoken in pretokens:
            pretoken = pretoken.encode("utf-8")
            pretoken_tuple = tuple(bytes([byte]) for byte in pretoken)
            i = 0
            while (i < len(self.merges) and len(pretoken_tuple) > 1):
                merge = self.merges[i]
                pretoken_tuple = self._merge_in_pretoken_tuple(merge, pretoken_tuple)
                i += 1
            for token_byte in pretoken_tuple:
                token_id = self.vocab_to_id[token_byte]
                token_ids.append(token_id)
                tokenized_text.append(token_byte)
        return tokenized_text, token_ids

def read_data_for_tokenization(file):
    # with open(filename) as file: # must be called outside the function
    for line in file:
        yield line

# if __name__ == "__main__":
    
#     time1 = time.time()

#     train_data_str = 'ts_train' # owt_train

#     with open(train_data_str+'_vocab.pickle', 'rb') as handle:
#         vocab = pickle.load(handle)

#     with open(train_data_str+'_merges.pickle', 'rb') as handle:
#         merges = pickle.load(handle)

#     vocab_to_id = {}
#     for key, value in vocab.items():
#         vocab_to_id[value] = key
    
#     # if (data_str == 'ts_train'):
#     #     pass
#     # elif(data_str == 'owt_train'):
#     #     pass
#     # else:
#     #     print('Invalid data_str. Exiting...')
#     #     exit(1)

#     special_tokens = ['<|endoftext|>']

#     # data_iterator = read_data_for_tokenization(None, text="the cat sat on the mat")
#     x = "the cat sat on the mat.'<|endoftext|>supercalifragilisticexpialidocious'"
#     print('Created data iterator. Collecting results from processes...')

#     tokenized_documents = []
#     def aggregate_results(result): # could also write to file 
#         tokenized_documents.extend(result)

#     # pool = mp.Pool(processes=8)
#     # for x in tqdm(data_iterator):
#     #     pool.apply_async(_encode, args = (x, special_tokens, vocab_to_id, merges), callback=aggregate_results)
#     # pool.close()
#     # pool.join()
#     all_tokenized_text, all_tokenized_numbers = _encode(x, special_tokens, vocab_to_id, merges)
    
#     print("Total time: ", time.time() - time1)
