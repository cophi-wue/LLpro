import argparse
import logging
import multiprocessing
import os
import sys

import spacy
import torch
from spacy.tokens import Doc
from tqdm import tqdm

import llpro.components
from llpro.common import spacy_doc_to_dataframe
from llpro.components.tokenizer_somajo import SoMaJoTokenizer

from tqdm.contrib.logging import logging_redirect_tqdm


def get_cpu_limit():
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    cpus = multiprocessing.cpu_count() if container_cpus < 1 else container_cpus
    return cpus


def run_pipeline_on_files(filenames, nlp, tokenizer=None):
    file_sizes = [os.path.getsize(f) for f in filenames]

    if tokenizer is None:
        tokenizer = nlp.tokenizer

    if not Doc.has_extension('filename'):
        Doc.set_extension('filename', default=None)

    with logging_redirect_tqdm():
        file_pbar = tqdm(total=sum(file_sizes), position=1, unit='B', unit_scale=True, dynamic_ncols=True)
        file_pbar.set_description_str(f'0/{len(filenames)}')
        for i, (filename, size) in enumerate(zip(filenames, file_sizes)):
            with open(filename) as f:
                content = f.read()
                doc = tokenizer(content)
                doc._.filename = filename

                nlp(doc)

            file_pbar.update(size)
            file_pbar.set_description_str(f'{i + 1}/{len(filenames)}')
            yield filename, doc
    file_pbar.close()


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
    parser.add_argument('--infiles', metavar='FILE', type=str, nargs='+', help='Input files, or directories.')
    parser.set_defaults(outtype='stdout')
    args = parser.parse_args()
    if args.writefiles is not None:
        args.outtype = 'files'
    logging.basicConfig(level=args.loglevel)
    for hdl in logging.getLogger('flair').handlers:
        logging.getLogger('flair').removeHandler(hdl)
    logging.getLogger('flair').propagate = True
    logging.info('Picked up following arguments: ' + repr(vars(args)))

    if torch.cuda.is_available():
        logging.info(f'torch: CUDA available, version {torch.version.cuda}, architectures {torch.cuda.get_arch_list()}')
    else:
        logging.info('torch: CUDA not available')

    filenames = []
    for f in args.infiles:
        if os.path.isfile(f):
            filenames.append(f)
        else:
            for root, dirs, members in os.walk(f, followlinks=True):
                filenames.extend([os.path.join(root, m) for m in members])

    logging.info('Loading pipeline')
    nlp = spacy.blank("de")
    # nlp = spacy.load('de_dep_news_trf', exclude=['ner', 'lemmatizer', 'textcat', 'morphologizer', 'attribute_ruler', 'parser'])
    nlp.add_pipe('tagger_someweta')
    nlp.add_pipe('tagger_rnntagger')
    nlp.add_pipe('lemma_rnntagger')
    nlp.add_pipe('parser_parzu_parallelized', config={'num_processes': 4})
    nlp.add_pipe('speech_redewiedergabe')
    nlp.add_pipe('scenes_stss_se')
    nlp.add_pipe('coref_uhhlt')
    nlp.add_pipe('ner_flair')
    nlp.add_pipe('events_uhhlt')
    nlp.analyze_pipes(pretty=True)

    tokenizer = SoMaJoTokenizer(nlp.vocab)
    for filename, tagged_doc in run_pipeline_on_files(filenames, nlp, tokenizer):
        if args.writefiles:
            with open(os.path.join(args.writefiles, os.path.basename(filename) + '.tsv'), 'w') as output_file:
                logging.info(f'writing processed tokens of {filename} to {output_file.name}')
                print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=output_file)
        elif args.outtype == 'stdout':
            print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=sys.stdout)
