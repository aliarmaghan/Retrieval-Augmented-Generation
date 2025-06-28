import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)


# Function to split data from a JSON file into chunks
# Each chunk will have metadata including the source and sequence ID
def split_data_from_file(file):
    #---- Define a variable to accumulate chunk records
    chunks_with_metadata = []

    #### Load json file using UTF-8 encoding
    with open(file, 'r', encoding='utf-8') as f:
        file_as_object = json.load(f)

    keys = list(file_as_object.keys())
    print(keys)

    #### pull these keys from the json file
    for item in keys:
        print(f'Processing {item} from {file}')

        #### grab the text of the item
        item_text = file_as_object[item]

        #### split the text into chunks
        item_text_chunks = text_splitter.split_text(item_text)

        chunk_seq_id = 0
        #### loop through chunks
        for chunk in item_text_chunks:
            #### extract file name from each chunk
            form_name = file[file.rindex('/') + 1:file.rindex('.')]

            #### create a record with metadata and the chunk text
            chunks_with_metadata.append({
                'text': chunk,
                'Source': item,
                'chunkSeqId': chunk_seq_id,
                'chunkId': f'{form_name}-{item}-chunk{chunk_seq_id:04d}',
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')

    return chunks_with_metadata
