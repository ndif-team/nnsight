from dictionary import AutoEncoder
from buffer import ActivationBuffer

import os
import torch as t
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from argparse import ArgumentParser
from nnsight import LanguageModel

def load_submodule(model, submodule_str):
    if "." not in submodule_str:
        return getattr(model, submodule_str)
    
    submodules = submodule_str.split(".")
    curr_module = None
    for module in submodules:
        if module == "model":
            continue
        if not curr_module:
            curr_module = getattr(model, module)
            continue
        curr_module = getattr(curr_module, module)
    return curr_module


def load_word_labels(dataset):
    """
    Returns dictionary of the following format:
    {
        num_spans: 1, text: {label_1: [word_idx_1, ..., word_idx_n]}
    }
    where `word_idx_i` is an int corresponding to the position of the word in the text, and
    `num_spans` is the number of components to consider.
    """
    label_set = set()
    word_labels = defaultdict(lambda: defaultdict(list))

    word_labels["num_spans"] = 1 if dataset[0]["span2"] is None else 2

    for example in dataset:
        text = example["text"]
        labels = example["labels"]
        span1_start = example["span1"]["word_start"]
        span1_end = example["span1"]["word_end"] - 1
        if example["span2"]:
            span2_start = example["span2"]["word_start"]
            span2_end = example["span2"]["word_end"] - 1
        else:
            span2_start = None
            span2_end = None

        for label in labels:
            word_labels[text][label].append([(span1_start, span1_end), (span2_start, span2_end)])
    
    return word_labels


def get_activations(text, model, submodule, dictionary):
    """
    Load activations of `dictionary` on every token of `text`.
    """
    with model.invoke(text) as invoker:
        x = submodule.output
        f = dictionary.encode(x)
        f_saved = f.save()
        y = dictionary.decode(f)
        submodule.output = y
    return f_saved.value


def convert_spans(text, spans_lists, tokenizer):
    """
    Take `text` (string) and a dictionary of lists of span boundary tuples, formatted as follows:
        {
            label_str_1: [[(span_1_start, span_1_end), (span_2_start, span_2_end)], ...],
            label_str_2: [[(...), (...)], [(...), (...)], ...]
        }
    Return a dictionary of lists of span boundary tuples where each span boundary integer has been converted
    what it should be after tokenization with `tokenizer`.
    """
    word_idx_to_tokenized_span = {}

    tokenized_text = tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])

    word_idx = 0
    word_start = 0
    for token_idx, token in enumerate(tokenized_text):
        if token.startswith("Ġ"):
            # Add previous word to dictionary
            word_idx_to_tokenized_span[word_idx] = (word_start, token_idx-1)
            # Start next word
            word_idx += 1
            word_start = token_idx
    # Add final word to list
    word_idx_to_tokenized_span[word_idx] = (word_start, len(tokenized_text)-1)

    posttok_spans = {}
    for label in spans_lists:
        posttok_spans[label] = t.zeros(len(tokenized_text))
        for span_pair in spans_lists[label]:
            span_1, span_2 = span_pair
            span_1 = list(span_1)
            span_2 = list(span_2)
            span_1[0] = word_idx_to_tokenized_span[span_1[0]][0]
            span_1[1] = word_idx_to_tokenized_span[span_1[1]][1]
            posttok_spans[label][span_1[0]:span_1[1]+1] = 1.0
            if span_2[0] is not None:
                span_2[0] = word_idx_to_tokenized_span[span_2[0]][0]
                span_2[1] = word_idx_to_tokenized_span[span_2[1]][1]
                posttok_spans[label][span_2[0]:span_2[1]+1] = 1.0

    return posttok_spans


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dictionary", type=str, default="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer4/1_32768/ae.pt")
    parser.add_argument("--submodule", type=str, default="model.gpt_neox.layers.4.mlp.dense_4h_to_h")
    parser.add_argument("--layer_num", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", "-d", type=str, default="/home/aaron/edge_probe/ewt-pos.json")
    parser.add_argument("--num_examples", "-n", type=int, default=None)
    args = parser.parse_args()

    model = LanguageModel(args.model_name)
    submodule = load_submodule(model, args.submodule)
    submodule_width = submodule.out_features
    autoencoder_size = 32768
    if "_sz" in args.dictionary:
        autoencoder_size = int(args.dictionary.split("_sz")[1].split("_")[0].split(".")[0])
    elif "_dict" in args.dictionary:
        autoencoder_size = int(args.dictionary.split("_dict")[1].split("_")[0].split(".")[0])
    elif "/0_" in args.dictionary:
        autoencoder_size = int(args.dictionary.split("0_")[1].split("/")[0])
    elif "/1_" in args.dictionary:
        autoencoder_size = int(args.dictionary.split("1_")[1].split("/")[0])
    dictionary = AutoEncoder(submodule_width, autoencoder_size).cuda()

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    train_examples = dataset["train"][0]
    word_labels = load_word_labels(train_examples)
    if args.num_examples and args.num_examples < len(word_labels.keys()):
        word_labels = {key:value for key,value in list(word_labels.items())[:args.num_examples]}
    num_texts = len(word_labels.keys())

    precisions = {}
    recalls = {}

    for text in tqdm(word_labels, desc="Example", total=len(word_labels.keys())):
        if text == "num_spans":
            continue
        feature_activations = get_activations(text, model, submodule, dictionary)
        # collapse into binary variable: 1 if greater than epsilon, 0 otherwise
        feature_activates = t.where(feature_activations > 0.01, 1.0, 0.0)
        feature_activates = feature_activates[0].T.int()  # shape: [num_features x num_tokens]
        
        # dictionary of Tensors where each token is labeled
        labeled_tokens = convert_spans(text, word_labels[text], model.tokenizer)

        for label in labeled_tokens:
            if label not in precisions:
                precisions[label] = t.zeros(autoencoder_size)
                recalls[label] = t.zeros(autoencoder_size)
            # TODO: replace for loop with tensor operation
            for feature_idx in range(feature_activates.shape[0]):
                if not t.any(feature_activates[feature_idx]):
                    continue
                labeled_tokens[label] = labeled_tokens[label].int()
                TP = t.sum(feature_activates[feature_idx] & labeled_tokens[label])
                FP = t.sum(feature_activates[feature_idx] & (labeled_tokens[label] == False).int())
                FN = t.sum((feature_activates[feature_idx] == False).int() & labeled_tokens[label])
                precisions[label][feature_idx] += TP / (TP + FP)
                recalls[label][feature_idx] += TP / (TP + FN)
    
    for label in precisions:
        prec = precisions[label].div(num_texts)
        recall = recalls[label].div(num_texts)
        f1s = (2 * prec * recall) / (prec + recall)
        f1s = t.nan_to_num(f1s)
        print(label, ":\n")
        print("\tF1:\t", t.topk(f1s, 10))
        print("\tPrecision:\t", t.topk(prec, 10))
        print("-------------------")