import os
import tqdm
import time
import pickle
import numpy as np
from cs336_basics.bpe_tokenizer import BPE_Tokenizer
from cs336_basics.helpers import get_text_from_file


def main():
    data_dir = '/data/c-worledge/data/' # '../data/'
    save_dir = '/data/c-worledge/t_results/'
    
    owt_train_input_path= data_dir+"owt_train2.txt"
    owt_valid_vocab_size = 32000

    t_results_fp = "t_results/"
    special_tokens = ['<|endoftext|>']
    
    owt_vocab_filename = save_dir+'owt_train'+f'_vocab_newP_{owt_valid_vocab_size}.pickle'
    with open(owt_vocab_filename, 'rb') as handle:
        owt_train_vocab = pickle.load(handle)
    owt_merges_filename = save_dir+'owt_train'+f'_merges_newP_{owt_valid_vocab_size}.pickle'
    with open(owt_merges_filename, 'rb') as handle:
        owt_train_merges = pickle.load(handle)

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

    # time2 = time.time()
    # all_ids = []
    # with open('t_results/owt_500_docs.txt', "r") as f:
    #     for _id in owt_tokenizer.encode_iterable(f):
    #         all_ids.append(_id)
    # print(f'the throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time2)} bytes/seconds')
    # #END# time test without parallization


    time0 = time.time()
    all_ids = []
    ################################################################################################################################################
    # ts_small_filename = ts_valid_input_path # 't_results/ts_500_docs.txt' # '../tests/fixtures/tinystories_sample.txt'  # ts_valid_input_path
    # with open(ts_small_filename, "r") as f:
    #     ts_text = f.read()
    # ts_text_num_bytes = len(ts_text.encode("utf-8"))

    # time1 = time.time()
    # all_ids.extend(ts_tokenizer.parallel_encode(ts_small_filename))
    # print(f'The throughput for the TS tokenizer is {ts_text_num_bytes/(time.time() - time1)} bytes/seconds')

    # save_fp = save_dir+'ts_val_tokens'
    # np.save(f"{save_fp}.npy", np.array(all_ids).astype(np.uint16))
    # print(f'The total time is {time.time() - time0} seconds')
    # arr_loaded = np.load(f"{save_fp}.npy")
    ###############################################################################################################################################
    # ts_small_filename = ts_train_input_path # 't_results/ts_500_docs.txt' # '../tests/fixtures/tinystories_sample.txt'  # ts_valid_input_path
    # with open(ts_small_filename, "r") as f:
    #     ts_text = f.read()
    # ts_text_num_bytes = len(ts_text.encode("utf-8"))

    # time1 = time.time()
    # all_ids.extend(ts_tokenizer.parallel_encode(ts_small_filename))
    # print(f'The throughput for the TS tokenizer is {ts_text_num_bytes/(time.time() - time1)} bytes/seconds')

    # save_fp = save_dir+'ts_train_tokens'
    # np.save(f"{save_fp}.npy", np.array(all_ids).astype(np.uint16))
    # print(f'The total time is {time.time() - time0} seconds')
    # arr_loaded = np.load(f"{save_fp}.npy")
    # ################################################################################################################################################
    # owt_small_filename = owt_valid_input_path # 't_results/ts_500_docs.txt' # '../tests/fixtures/tinystories_sample.txt'  # ts_valid_input_path
    # with open(owt_small_filename, "r") as f:
    #     owt_text = f.read()
    # owt_text_num_bytes = len(owt_text.encode("utf-8"))

    # time1 = time.time()
    # all_ids.extend(owt_tokenizer.parallel_encode(owt_small_filename))
    # print(f'The throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time1)} bytes/seconds')

    # save_fp = save_dir+'owt_val_tokens2'
    # np.save(f"{save_fp}.npy", np.array(all_ids).astype(np.uint16))
    # print(f'The total time is {time.time() - time0} seconds')
    # arr_loaded = np.load(f"{save_fp}.npy")
    ################################################################################################################################################
    owt_small_filename = owt_train_input_path # 't_results/ts_500_docs.txt' # '../tests/fixtures/tinystories_sample.txt'  # ts_valid_input_path
    # with open(owt_small_filename, "r") as f:
    #     owt_text = f.read()
    # owt_text_num_bytes = len(owt_text.encode("utf-8"))

    time1 = time.time()
    all_ids.extend(owt_tokenizer.parallel_encode(owt_small_filename))
    # print(f'The throughput for the OWT tokenizer is {owt_text_num_bytes/(time.time() - time1)} bytes/seconds')

    save_fp = save_dir+'owt_train_tokens2'
    np.save(f"{save_fp}.npy", np.array(all_ids).astype(np.uint16))
    print(f'The total time is {time.time() - time0} seconds')
    # arr_loaded = np.load(f"{save_fp}.npy")

    ################################################################################################################################################

    ##########
    # For OWT Valid:
    # The throughput for the OWT tokenizer is 191662.71726581582 bytes/seconds
    # The total time is 1515.2162628173828 seconds

    

if __name__ == '__main__':
    main()
