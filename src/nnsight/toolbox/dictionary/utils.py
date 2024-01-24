from datasets import load_dataset
import zstandard as zstd
import io
import json

def hf_dataset_to_generator(dataset_name, split='train', streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    def gen():
        for x in iter(dataset):
            yield x['text']
    
    return gen()

def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, 'rb')
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    def generator():
        for line in text_stream:
            yield json.loads(line)['text']
    return generator()