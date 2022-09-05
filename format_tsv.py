import argparse
import sys

from llpro.common import Token


def read_tokens(infile):
    return [Token.from_json(line) for line in infile]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts raw JSON output file of the LLPro pipeline into a tab-separated table')
    parser.add_argument('infile', metavar='INFILE', type=str, nargs='?', default='-', help='Input file, or - for stdin')
    parser.add_argument('outfile', metavar='OUTFILE', type=str, nargs='?', default='-', help='Output file, or - for stdout')
    args = parser.parse_args()

    if args.infile == '-':
        infile = sys.stdin
    else:
        infile = open(args.infile)

    tokens = read_tokens(infile)
    df = Token.to_dataframe(tokens)

    if args.outfile == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.outfile)
    df.to_csv(outfile, sep='\t', index=False)
