import json
import html
from collections import Counter
import pathlib
import os

from joblib import Parallel, delayed
from gensim.parsing import preprocessing

custom_filters = [
    lambda x: x.lower(),
    lambda x: x.replace("\n", " ###newline### "),
    preprocessing.strip_multiple_whitespaces,
    html.unescape,
    html.unescape,
    html.unescape,
    html.unescape,  # there are recursive entities. It sucks
    #     preprocessing.strip_numeric,
    #     preprocessing.strip_punctuation,
    preprocessing.remove_stopwords,
    #     preprocessing.strip_short,
]

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

stopwords = []
# with open("results_monthly/stopwords.json") as f:
    # stopwords = set(json.load(f))


def process(path):
    # import data
    with open(path) as f:
        text = f.read()

    # process entire string
    string_processed = preprocessing.preprocess_string(text, custom_filters)

    # process individual tokens
    string_processed = [i.strip(punctuation) for i in string_processed]

    # get dictionary
    count = Counter(string_processed)
    return string_processed, count, str(path)


def filter_single(tokens, dictionary):
    return [i for i in tokens if (dictionary[i] > 10 and i not in stopwords)]


def load_data_vectors(path):
    # First pass processing
    with Parallel(n_jobs=-1, verbose=5) as p:
        generator = (delayed(process)(i) for i in pathlib.Path(path).glob("*.txt"))
        data = p(generator)

        counts = sum((i[1] for i in data), Counter())
        names = [i[2] for i in data]

        # filter out words that appear only once in the entire corpus.
        gen = (delayed(filter_single)(i, counts) for i, d, n in data)
        data = p(gen)

    return {k: v for k, v in zip(names, data)}


if __name__ == "__main__":
    mallet_path = "/home/ryan/local/bin/Mallet/bin/mallet"

    output_path = "covid/mallet"
    input_path = "covid/data/weekly_text"

    # Tokenization
    word_tokens = load_data_vectors(input_path)

    # Save as new text format
    for key, tokens in word_tokens.items():
        name = key.split("/")[-1]
        with open(f"{output_path}/data/{name}", "w+") as f:
            f.write(" ".join(tokens))

    # convert to mallet format
    os.system(f"{mallet_path} import-dir --input {output_path}/data "
              f"--output {output_path}/topic-input.mallet --keep-sequence --remove-stopwords")

    # train topic model
    hyper = {
        "num_topics": 16,
        "num_iterations": 2000,
        "optimize-interval": 10,
        "num-top-words": 100,
    }

    os.system(f"{mallet_path} train-topics --input {output_path}/topic-input.mallet "
              f"--output-state {output_path}/topic-state.gz "
              f"--output-doc-topics {output_path}/doc_topics.txt "
              f"--output-topic-keys {output_path}/topic_keys.txt "
              f"--inferencer-filename {output_path}/model.mallet "
              f"--num-topics {hyper['num_topics']} --num-iterations {hyper['num_iterations']} --optimize-interval 10 "
              f"--num-top-words {hyper['num-top-words']}")
