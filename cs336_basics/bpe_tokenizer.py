import regex as re
import multiprocessing as mp 
from tqdm import tqdm
from typing import List, Dict, Iterable, Iterator, Tuple
import pickle 
import numpy as np
import cProfile
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer:
    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[tuple[bytes]], 
                 special_tokens: list[str] | None = None,
                 vocab_filename: str = None,
                 merges_filename: str = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_filename = vocab_filename
        self.merges_filename = merges_filename
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
        return bytes(decoded_text).decode("utf-8")
    
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
    
    def parallel_encode(self, filename: str) -> List[int]:
        assert self.vocab_filename
        assert self.merges_filename
        data_iterator = read_data(filename, self.vocab_filename, self.merges_filename, self.special_tokens)

        all_token_ids = []
        def accumulate_token_ids(result: List[int]) -> List[int]: 
            # This is called whenever the pool returns a result.
            # all_token_ids is modified only by the main process, not the pool workers.
            all_token_ids.extend(result)
            return all_token_ids
        # ############# local run
        # global vocab
        # vocab = self.vocab
        # global merges
        # merges = self.merges
        # global vocab_to_id
        # vocab_to_id = {}
        # for key, value in vocab.items():
        #     vocab_to_id[value] = key
        # global previously_encoded_pretokens
        # previously_encoded_pretokens = {}

        # profiler = cProfile.Profile()
        # profiler.enable()
        # for document in data_iterator:
        #     result = _parallelizable_encode(*document)
        #     all_token_ids.extend(result)
        # profiler.disable()
        # profiler.print_stats(sort='cumtime') 
        # ############# end local run

        # ############# start parallel 1
        # with mp.Pool(initializer=init_worker, initargs=(self.vocab_filename, self.merges_filename), processes=8)  as pool:
        #     for document in tqdm(data_iterator): # send over the file and the byte indices in larger chunks. The use seek to get the text.
        #         pool.apply_async(_parallelizable_encode, args=document, callback=accumulate_token_ids)
        #     pool.close()
        #     pool.join()
        # ############# end parallel 1    

        with mp.Pool(initializer=init_worker, initargs=(self.vocab_filename, self.merges_filename), processes=8)  as pool:
            tasks = [pool.apply_async(_parallelizable_encode, args=document) for document in data_iterator]
            pool.close()
            pool.join()
        
        all_token_ids = []
        for task in tasks:
            all_token_ids.extend(task.get())

        # for i, task in enumerate(tasks):
        #     try:
        #         result = task.get()
        #         print(f"Task {i} result: {result}")
        #     except Exception as e:
        #         print(f"Task {i} failed with exception: {e}")

        return all_token_ids
            
    ############################################
########################################

vocab = None
merges = None
vocab_to_id = None

def init_worker(vocab_filename, merges_filename):
    global vocab
    with open(vocab_filename, 'rb') as handle: 
        vocab = pickle.load(handle)
    # with open(vocab_filename, "r") as f: # for running tests
    #     vocab = json.load(f)

    global merges
    with open(merges_filename, 'rb') as handle: 
        merges = pickle.load(handle)
    # with open(merges_filename, "r") as f: # for running tests
    #     merges = f.read()

    global vocab_to_id 
    vocab_to_id = {}
    for key, value in vocab.items():
        vocab_to_id[value] = key

    global previously_encoded_pretokens
    previously_encoded_pretokens = {}

def read_data_for_tokenization(file):
    # with open(filename) as file: # must be called outside the function
    for line in file:
        yield line

def read_data(file_name, vocab_filename, merges_filename, special_tokens):
    num_lines_per_chunk = 4096 # this is arbitrary 
    document_chunk = ""
    with open(file_name, 'rb') as file:
        pos = file.tell()
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell() 
        file.seek(pos)
        finished = False
        while pos < file_size:
            i = 0
            while ((i < num_lines_per_chunk) or ((i >= num_lines_per_chunk) and ('<|endoftext|>' not in document_chunk))): # ensures that <|endoftext|> is in the chunk 
                line = file.readline()
                if (line == b''):  # if eof is reached
                    if (len(document_chunk) > 0):
                        finished = True
                        yield (file_name, pos, len(document_chunk.encode('utf-8')), vocab_filename, merges_filename, special_tokens)
                    return
                else: 
                    document_chunk += line.decode('utf-8')
                i += 1
            if (finished):
                break
            document_chunk_splits = document_chunk.split('<|endoftext|>')
            curr_document_chunk = '<|endoftext|>'.join(document_chunk_splits[:-1])+'<|endoftext|>'
            document_chunk = document_chunk_splits[-1] # start of the next document_chunk
            next_pos = file.tell() - len(document_chunk.encode('utf-8')) 
            size = len(curr_document_chunk.encode('utf-8'))
            yield (file_name, pos, size, vocab_filename, merges_filename, special_tokens)
            pos = next_pos

def get_text_from_file(file, pos, size):
    file.seek(pos)
    text = file.read(size)
    return text
###########################################

def _parallelizable_split_keep_delimiters(text, special_tokens):
        # order special tokens by length (longest to shortest)
        # split on the special tokens and keep the delimiters
        # append the results together into one list 
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

        # return [x for x in texts if x != '']
        for x in texts:
            if x != '':
                yield x

def _parallelizable_merge_in_pretoken_tuple(merge: List[bytes], pretoken_tuple: Tuple[bytes]):
        """Merges the two bytes in the pretoken tuple if they are adjacent."""
        i = 0
        while i < len(pretoken_tuple) - 1:
            pretoken_merge_candidate = pretoken_tuple[i:i+2]
            if (pretoken_merge_candidate[0] == merge[0]) and (pretoken_merge_candidate[1] == merge[1]):
                pretoken_tuple = pretoken_tuple[:i] + (pretoken_tuple[i] + pretoken_tuple[i+1],) + pretoken_tuple[i+2:]
            i += 1
        return pretoken_tuple

def _parallelizable_encode_nonspecial_text(text, merges, vocab_to_id):
        """Encodes the text using the merges and vocab_to_id. The text does not contain special tokens."""
        # TODO split with re.findall(PAT, text) when splitting on pretokens
        pretokens = re.findall(PAT, text) 
        token_ids = []
        for pretoken in pretokens:
            curr_token_ids = [vocab_to_id.get(pretoken.encode("utf-8"), -1)]
            if (curr_token_ids[0] == -1):            
                curr_token_ids = previously_encoded_pretokens.get(pretoken, None)
                if not curr_token_ids:
                    pretoken = pretoken.encode("utf-8")
                    pretoken_tuple = tuple(bytes([byte]) for byte in pretoken)
                    i = 0
                    
                    while (i < len(merges) and len(pretoken_tuple) > 1):
                        merge = merges[i]
                        if (merge[0] in pretoken_tuple) and (merge[1] in pretoken_tuple):
                            pretoken_tuple = _parallelizable_merge_in_pretoken_tuple(merge, pretoken_tuple)
                        i += 1
                    curr_token_ids = []
                    
                    for token_byte in pretoken_tuple:
                        token_id = vocab_to_id[token_byte] # TODO THIS line is causing trouble! For the command below. 
                        # (You'll have to revert to the "for running tests" vocab + merge loads)
                        # `uv run pytest -k test_parallel_encode_iterable_tinystories_sample_roundtrip`
                        curr_token_ids.append(token_id)
                    previously_encoded_pretokens[pretoken] = curr_token_ids
            token_ids.extend(curr_token_ids)
        return token_ids

def _parallelizable_encode(data_filename, pos, size, vocab_filename, merges_filename, special_tokens): 
    """Encodes the text using the merges and vocab_to_id. The text may contain special tokens."""
    # with open(vocab_filename, 'rb') as handle:
    #     vocab = pickle.load(handle)
    # with open(merges_filename, 'rb') as handle:
    #     merges = pickle.load(handle)

    # vocab_to_id = {}
    # for key, value in vocab.items():
    #     vocab_to_id[value] = key
    
    with open(data_filename, 'rb') as file:
        file.seek(pos)
        text = file.read(size).decode('utf-8')

    # an iterator over an ordered list of the strings and special tokens in x
    chunk_iterator = _parallelizable_split_keep_delimiters(text, special_tokens)

    all_token_ids = []
    for text_fragment in chunk_iterator:
        if (text_fragment in special_tokens):
            all_token_ids.append(vocab_to_id[text_fragment.encode("utf-8")])
        else:
            token_ids = _parallelizable_encode_nonspecial_text(text_fragment, merges, vocab_to_id)
            all_token_ids.extend(token_ids)

    return all_token_ids



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
