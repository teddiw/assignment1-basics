import os
import tqdm
import time
import pickle
from cs336_basics.bpe_tokenizer import BPE_Tokenizer
from cs336_basics.helpers import get_text_from_file

def get_n_docs(filename, n_docs):
    docs = []
    i = 1
    while len(docs) <= n_docs:
        text = get_text_from_file(filename, 0, i*1024)
        docs = text.split('<|endoftext|>')
        i += 1
    return '<|endoftext|>'.join(docs[:n_docs])


    # docs = []
    # pos = 0
    # curr_doc = '' 
    # while len(docs) < n_docs:
    #     text = get_text_from_file(filename, pos, 1024)
    #     text_fragments = text.split('<|endoftext|>')
    #     curr_doc += text_fragments[0]+'<|endoftext|>'
    #     if (len(text_fragments) > 1):
    #         docs.append(curr_doc)
    #         curr_doc = ''
    #     for i in range(1, len(text_fragments)):
    #         if (len(text_fragments[i]) > 0):
    #             docs.append(text_fragments[i])
        
    #     pos += 1024

def main():
    ts_valid_input_path= "../data/TinyStoriesV2-GPT4-valid.txt" 
    ts_vocab_size = 10000
    owt_valid_input_path= "../data/owt_valid.txt"
    owt_valid_vocab_size = 32000

    t_results_fp = "t_results/"
    special_tokens = ['<|endoftext|>']
    
    owt_vocab_filename = t_results_fp+'owt_train'+f'_vocab_newP_{owt_valid_vocab_size}.pickle'
    with open(owt_vocab_filename, 'rb') as handle:
        owt_train_vocab = pickle.load(handle)
    owt_merges_filename = t_results_fp+'owt_train'+f'_merges_newP_{owt_valid_vocab_size}.pickle'
    with open(owt_merges_filename, 'rb') as handle:
        owt_train_merges = pickle.load(handle)

    ts_vocab_filename = t_results_fp+'ts_train'+f'_vocab_newP_{ts_vocab_size}.pickle'
    with open(ts_vocab_filename, 'rb') as handle:
        ts_train_vocab = pickle.load(handle)
    ts_merges_filename = t_results_fp+'ts_train'+f'_merges_newP_{ts_vocab_size}.pickle'
    with open(ts_merges_filename, 'rb') as handle:
        ts_train_merges = pickle.load(handle)

    ts_tokenizer = BPE_Tokenizer(ts_train_vocab, ts_train_merges, special_tokens, vocab_filename=ts_vocab_filename, merges_filename=ts_merges_filename)
    owt_tokenizer = BPE_Tokenizer(owt_train_vocab, owt_train_merges, special_tokens, vocab_filename=owt_vocab_filename, merges_filename=owt_merges_filename)
    #####################
    # ts_text = get_n_docs(ts_valid_input_path, 10) 
    # ts_text_num_bytes = len(ts_text.encode("utf-8"))
    # owt_text = get_n_docs(owt_valid_input_path, 10) 
    # owt_text_num_bytes = len(owt_text.encode("utf-8"))

    # ts_token_ids = ts_tokenizer.encode(ts_text)
    # owt_token_ids = owt_tokenizer.encode(owt_text)

    # print('TS Compression Ratio:', ts_text_num_bytes/len(ts_token_ids))
    # print('OWT Compression Ratio:', owt_text_num_bytes/len(owt_token_ids))

    # ts_token_ids = ts_tokenizer.encode(owt_text)
    # owt_token_ids = owt_tokenizer.encode(ts_text)

    # print('TS compression ratio on OWT:', owt_text_num_bytes/len(ts_token_ids))
    # print('OWT compression ratio on TS:', ts_text_num_bytes/len(owt_token_ids))
    ####################
    # ts_text = get_n_docs(ts_valid_input_path, 500) 
    # ts_text_num_bytes = len(ts_text.encode("utf-8"))
    # owt_text = get_n_docs(owt_valid_input_path, 500) 
    # owt_text_num_bytes = len(owt_text.encode("utf-8"))

    # with open('t_results/owt_500_docs.txt', "w") as f:
    #     f.write(owt_text)

    # with open('t_results/ts_500_docs.txt', "w") as f:
    #     f.write(ts_text)
    ####################
    # START# time test without parallization
    # time1 = time.time()
    # all_ids = []
    # with open('t_results/ts_500_docs.txt', "r") as f:
    #     for _id in ts_tokenizer.encode_iterable(f):
    #         all_ids.append(_id)
            
    # print(f'The throughput for the TS tokenizer is {ts_text_num_bytes/(time.time() - time1)} bytes/seconds')
    # decoded_text = ts_tokenizer.decode(all_ids)
    # breakpoint()

    # time2 = time.time()
    # all_ids = []
    # with open('t_results/owt_500_docs.txt', "r") as f:
    #     for _id in owt_tokenizer.encode_iterable(f):
    #         all_ids.append(_id)
    # print(f'the throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time2)} bytes/seconds')
    # #END# time test without parallization

    ts_small_filename = ts_valid_input_path
    with open(ts_small_filename, "r") as f:
        ts_text = f.read()

    owt_small_filename = owt_valid_input_path
    with open(owt_small_filename, "r") as f:
        owt_text = f.read()

    ts_text_num_bytes = len(ts_text.encode("utf-8"))
    owt_text_num_bytes = len(owt_text.encode("utf-8"))

    time1 = time.time()
    all_ids = ts_tokenizer.parallel_encode(ts_small_filename)
            
    print(f'The throughput for the TS tokenizer is {ts_text_num_bytes/(time.time() - time1)} bytes/seconds')
    # ts_decoded_text = ts_tokenizer.decode(all_ids)

    time2 = time.time()
    all_ids = owt_tokenizer.parallel_encode(owt_small_filename)
    print(f'the throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time2)} bytes/seconds')
    # owt_decoded_text = owt_tokenizer.decode(all_ids)

    # time1 = time.time()
    # ts_tokenizer.encode(ts_text)
    # print(f'The throughput for the TS tokenizer is {ts_text_num_bytes/(time.time() - time1)} bytes/seconds')

    # time2 = time.time()
    # owt_tokenizer.encode(owt_text)
    # print(f'the throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time2)} bytes/seconds')
    breakpoint()
if __name__ == '__main__':
    main()
