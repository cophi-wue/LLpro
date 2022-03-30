from glob import glob

from tqdm import tqdm

from llppipeline.pipeline import *

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info('Loading modules')

    tokenizer = NLTKPunktTokenizer()
    # pos_tagger = SoMeWeTaTagger()
    morph_tagger = RNNTagger()
    lemmatizer = RNNLemmatizer()
    parzu = ParallelizedModule(ParzuParser, num_processes=20, chunks_per_process=10)


    def files():
        for fname in glob('/mnt/data/kallimachos/Romankorpus/Heftromane/txt/*762'):
            fobj = open(fname)
            yield fobj, fname
            fobj.close()

    for file, filename in files():
        logging.info(f'Start tokenization for {filename}')
        tokens = list(tokenizer.tokenize(file, filename))
        logging.info(f'Start tagging for {filename}')
        morph_tagger.run(tokens)
        lemmatizer.run(tokens)
        parzu.run(tokens)
        for tok in tokens:
            print(tok.to_output_line(modules={'pos': 'rnntagger', 'morph': 'rnntagger', 'lemma': 'rnnlemmatizer'}))

