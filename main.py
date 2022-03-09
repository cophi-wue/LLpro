import itertools
import logging
import warnings

from tqdm import tqdm

from llppipeline.pipeline import *
from glob import glob


if __name__ == "__main__":
    print(subprocess.run(['which', 'python'], shell=False, text=True, capture_output=True).stdout)

    logging.getLogger().setLevel(logging.DEBUG)
    logging.info('Loading modules')
    tokenizer = NLTKPunktTokenizer()
    pos_tagger = SoMeWeTaTagger()
    morph_tagger = RNNTagger()


    def files():
        for fname in glob('/mnt/data/kallimachos/Romankorpus/Heftromane/txt/*762'):
            fobj = open(fname)
            yield fobj, fname
            fobj.close()

    # token_stream = itertools.chain.from_iterable(tokenizer.tokenize(file, filename) for file, filename in files())
    # tagger.process(token_stream)
    for file, filename in files():
        logging.info(f'Start tokenization for {filename}')
        tokens = list(tokenizer.tokenize(file, filename))
        logging.info(f'Start tagging for {filename}')
        pos_tagger.process(tokens)
        morph_tagger.process(tokens)
        for tok in tokens:
            print(tok)

