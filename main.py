import argparse
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
    print(args)

    logging.info('Loading modules')
    tokenizer = NLTKPunktTokenizer()
    pos_tagger = SoMeWeTaTagger()
    morph_tagger = RNNTagger()
    lemmatizer = RNNLemmatizer()
    # parzu = ParallelizedModule(ParzuParser, num_processes=20, tokens_per_process=1000)
    parzu = ParzuParser(pos_source=pos_tagger.name)

    filenames = more_itertools.collapse(
        [next(os.walk(f), (None, None, []))[2] if os.path.isdir(f) else f for f in args.infiles])
    for filename, processed_tokens in pipeline_process(tokenizer, [pos_tagger, morph_tagger, lemmatizer, parzu], list(filenames)):
        if args.format == 'conll':
            output = Token.to_conll(processed_tokens,
                                    modules={'pos': pos_tagger.name, 'morph': morph_tagger.name, 'lemma': lemmatizer.name})
        else:
            output = '\n'.join(
                [json.dumps(dict((field + '_' + module, value) for (field, module), value in tok.fields.items())) for
                 tok in processed_tokens])

        if args.writefiles is not None:
            with open(os.path.join(args.writefiles, os.path.basename(filename)), 'w') as out:
                print(output, file=out)
        else:
            print(output)
