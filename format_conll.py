import argparse
import json
import os
import sys

from llppipeline.common import Token


def read_tokens(infile):
    for line in infile:
        obj = json.loads(line)
        fields = {}
        for field, v in obj.items():
            for module, value in v.items():
                fields[(field, module)] = value
        yield Token(fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLP Pipeline for literary texts written in German.')
    parser.add_argument('--stdout', help='Write output to stdout', action='store_const', const=True, default=False)
    parser.add_argument('infiles', metavar='FILE', type=str, nargs='+', help='Input files')
    args = parser.parse_args()

    for f in args.infiles:
        if f == '-':
            infile = sys.stdin
        else:
            infile = open(f)

        tokens = read_tokens(infile)

        # hardcoded, cf. with main.py
        if args.stdout:
            out = sys.stdout
        else:
            out = open(os.path.join(os.path.dirname(f), os.path.basename(f) + '.conll'), 'w')

        for line in Token.to_conll(tokens, pos='SoMeWeTaTagger', morph='RNNTagger', lemmatizer='RNNLemmatizer'):
            print(line, file=out)

        if not args.stdout:
            print(f'writing to {out.name}', file=sys.stderr)
            out.close()
        infile.close()
