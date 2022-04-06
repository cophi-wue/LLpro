import argparse
import collections
import json

from llppipeline.pipeline import *

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
    parzu = ParallelizedModule(lambda: ParzuParser(pos_source=pos_tagger.name), num_processes=20, tokens_per_process=1000, name='ParzuParser')

    for filename, processed_tokens in pipeline_process(tokenizer, [pos_tagger, morph_tagger, lemmatizer, parzu], list(filenames)):
        if args.format == 'conll':
            output = Token.to_conll(processed_tokens,
                                    modules={'pos': pos_tagger.name, 'morph': morph_tagger.name, 'lemma': lemmatizer.name})
        else:
            output = []
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
