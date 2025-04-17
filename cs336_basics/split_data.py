data_dir = '/data/a1-basics/'
filename = data_dir+"owt_train.txt"
with open(filename, 'r') as file:
    pos = file.tell()
    file_size = file.seek(0, 2)
    midway_byte = int(file_size//2)+2048*6+1024+512+23

def split_file_at_byte_streaming(input_path, split_byte, output_path_1, output_path_2, chunk_size=1024 * 1024):
    with open(input_path, "rb") as infile, \
         open(output_path_1, "wb") as out1, \
         open(output_path_2, "wb") as out2:

        bytes_read = 0

        while bytes_read < split_byte:
            to_read = min(chunk_size, split_byte - bytes_read)
            chunk = infile.read(to_read)
            if not chunk:
                break
            out1.write(chunk)
            bytes_read += len(chunk)

        # Now write the rest to the second file
        while True:
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            out2.write(chunk)

split_file_at_byte_streaming(data_dir+"owt_train.txt", midway_byte, "/data/c-worledge/data/owt_train1.txt", "/data/c-worledge/data/owt_train2.txt")

# filename = "/data/c-worledge/data/owt_train1.txt"
# with open(filename, 'r') as file:
#     pos = file.tell()
#     file_size = file.seek(0, 2)
#     midway_byte = int(file_size//2)+2048*6+1024+512+23

# filename = "/data/c-worledge/data/owt_train2.txt"
# with open(filename, 'r') as file:
#     pos = file.tell()
#     file_size = file.seek(0, 2)
#     midway_byte = int(file_size//2)+2048*6+1024+512+23
