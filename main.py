import argparse
import collections
import json
import math

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
    rw_tagger = RedewiedergabeTagger(device_on_run=True)
    ner_tagger = FLERTNERTagger(device_on_run=True)
    coref_tagger = CorefIncrementalTagger(device_on_run=True)
    modules = [pos_tagger, morph_tagger, lemmatizer, parzu, rw_tagger, ner_tagger, coref_tagger]

    srl_tagger = None
    if os.path.exists('resources/inveroxl/resources/model/weights.pt'):
        srl_tagger = InVeRoXL(device_on_run=True)
        modules.append(srl_tagger)

    for filename, processed_tokens in pipeline_process(tokenizer, modules, list(filenames)):
        output = []
        if args.format == 'conll':
            output = Token.to_conll(processed_tokens, pos=pos_tagger.name, morph=morph_tagger.name,
                                    lemmatizer=lemmatizer.name)
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
