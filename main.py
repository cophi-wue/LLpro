import argparse
import collections
import json
import math
import flair

from llppipeline.pipeline import *


def get_cpu_limit():
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    cpus = multiprocessing.cpu_count() if container_cpus < 1 else container_cpus
    return cpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLP Pipeline for literary texts written in German.')
    parser.add_argument('-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('--format', choices=['json', 'conll'], default='json')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stdout', default=True, help='Write all processed tokens to stdout',
                       action='store_const', dest='outtype', const='stdout')
    group.add_argument('--writefiles', metavar='DIR',
                       help='For each input file, write processed tokens to a separate file in DIR', default=None)
    parser.add_argument('infiles', metavar='FILE', type=str, nargs='+', help='Input files, or directories')
    parser.set_defaults(outtype='stdout')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    for hdl in logging.getLogger('flair').handlers:
        logging.getLogger('flair').removeHandler(hdl)
    logging.getLogger('flair').propagate = True

    filenames = []
    for f in args.infiles:
        if os.path.isfile(f):
            filenames.append(f)
        else:
            for root, dirs, members in os.walk(f, followlinks=True):
                filenames.extend([os.path.join(root, m) for m in members])

    logging.info('Loading modules')
    tokenizer = NLTKPunktTokenizer()
    pos_tagger = SoMeWeTaTagger()
    morph_tagger = RNNTagger()
    lemmatizer = RNNLemmatizer()
    parzu = ParallelizedModule(lambda: ParzuParser(pos_source=pos_tagger.name),
                               num_processes=math.floor(get_cpu_limit()),
                               tokens_per_process=1000, name='ParzuParser')
    rw_tagger = RedewiedergabeTagger()
    ner_tagger = FLERTNERTagger()

    for filename, processed_tokens in pipeline_process(tokenizer,
                                                       [pos_tagger, morph_tagger, lemmatizer, parzu, rw_tagger, ner_tagger],
                                                       list(filenames)):
        output = []
        if args.format == 'conll':
            for sent in Token.get_sentences(processed_tokens):
                for tok in sent:
                    misc_items = [f'STWR{rw_type}={d["value"]}' for rw_type, d in
                                  tok.get_field('redewiedergabe', rw_tagger.name, default={}).items()
                                  if d["value"] == 'yes']
                    if tok.get_field('ner', ner_tagger.name, None) is not None:
                        misc_items.append('NER=' + tok.get_field('ner', ner_tagger.name, None))
                    field_strings = [tok.id, tok.word,
                                     tok.get_field('lemma', lemmatizer.name, default='_'),
                                     '_',  # UPOS
                                     tok.get_field('pos', pos_tagger.name, default='_'),
                                     tok.get_field('morph', morph_tagger.name, default='_'),
                                     tok.get_field('head', parzu.name, default='_'),
                                     tok.get_field('deprel', parzu.name, default='_'),
                                     '_',  # DEPS
                                     '|'.join(misc_items)
                                     ]
                    output.append('\t'.join([str(x) for x in field_strings]))
                output.append('')
            output = '\n'.join(output)
        else:
            for tok in processed_tokens:
                obj = collections.defaultdict(lambda: {})
                for (field, module), value in tok.fields.items():
                    obj[field][module] = value
                output.append(json.dumps(obj))
            output = '\n'.join(output)

        if args.writefiles is not None:
            with open(os.path.join(args.writefiles, os.path.basename(filename)), 'w') as out:
                print(output, file=out)
        else:
            print(output)
