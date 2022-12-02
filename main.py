import argparse
import logging
import multiprocessing
import os

import spacy

import llpro.components
from llpro.components.tokenizer_somajo import SoMaJoTokenizer


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
    parser.add_argument('--paragraph-pattern', metavar='PAT', type=str, default=None,
                        help='Optional paragraph separator pattern. Paragraph separators are removed, and sentences '
                             'always terminate on paragraph boundaries.')
    parser.add_argument('--section-pattern', metavar='PAT', type=str, default=None,
                        help='Optional sectioning paragraph pattern. Paragraphs fully matching the pattern are '
                             'removed, and increment the section id counter for tokens in intermediate paragraphs.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stdout', default=True, help='Write all processed tokens to stdout.',
                       action='store_const', dest='outtype', const='stdout')
    group.add_argument('--writefiles', metavar='DIR',
                       help='For each input file, write processed tokens to a separate file in DIR.', default=None)
    parser.add_argument('--format', choices=['tsv', 'jsonl'], action='append',
                        help='Output format to write. If --writefiles is given, then --format can be specified multiple '
                             'times, writing the respective output format for each output file. Defaults to \'jsonl\'.')
    parser.add_argument('infiles', metavar='FILE', type=str, nargs='+', help='Input files, or directories.')
    parser.set_defaults(outtype='stdout')
    args = parser.parse_args()
    # if args.writefiles is not None:
    #     args.outtype = 'files'
    # if not args.format or len(args.format) == 0:
    #     args.format = ['jsonl']
    # logging.basicConfig(level=args.loglevel)
    # for hdl in logging.getLogger('flair').handlers:
    #     logging.getLogger('flair').removeHandler(hdl)
    # logging.getLogger('flair').propagate = True
    # logging.info('Picked up following arguments: ' + repr(vars(args)))
    #
    # if len(set(args.format)) > 1 and not args.writefiles:
    #     logging.error('Can only specify a single output format unless --writefiles is given.')
    #     sys.exit(1)
    #
    # if torch.cuda.is_available():
    #     logging.info(f'torch: CUDA available, version {torch.version.cuda}, architectures {torch.cuda.get_arch_list()}')
    # else:
    #     logging.info('torch: CUDA not available')
    #
    # filenames = []
    # for f in args.infiles:
    #     if os.path.isfile(f):
    #         filenames.append(f)
    #     else:
    #         for root, dirs, members in os.walk(f, followlinks=True):
    #             filenames.extend([os.path.join(root, m) for m in members])

    logging.info('Loading pipeline')
    nlp = spacy.blank("de")
    nlp.tokenizer = SoMaJoTokenizer()
    # nlp.add_pipe('tagger_someweta')
    # nlp.add_pipe('tagger_rnntagger')
    # nlp.add_pipe('lemma_rnntagger')
    # nlp.add_pipe('parser_parzu_parallelized', config={'num_processes': 1})
    # nlp.add_pipe('speech_redewiedergabe')
    # nlp.add_pipe('scenes_stss_se')
    nlp.add_pipe('ner_flair')
    nlp.analyze_pipes(pretty=True)

    text = open('./files/in/testfile1').read()
    doc = nlp(text)
    for tok in doc:
        print(tok.text, tok.ent_iob_, tok.ent_type_)

