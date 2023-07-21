import argparse
import logging
import multiprocessing
import os
import sys
import time

import spacy
import torch
from spacy.tokens import Doc
from tqdm import tqdm

from tqdm.contrib.logging import logging_redirect_tqdm

import llpro.components
from llpro.common import spacy_doc_to_dataframe
from llpro.components.tokenizer_somajo import SoMaJoTokenizer

logger = logging.getLogger('llpro_cli')


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
        file_pbar = tqdm(total=sum(file_sizes), position=1, unit='B', unit_scale=True, ncols=80)
        file_pbar.set_description_str(f'0/{len(filenames)}')
        for i, (filename, size) in enumerate(zip(filenames, file_sizes)):
            start_time = time.monotonic()
            with open(filename) as f:
                content = f.read()
                # tokenize outside of pipeline to set filename on resulting tokenized Doc
                tokenization_start_time = time.monotonic()
                doc = tokenizer(content)
                tokenization_end_time = time.monotonic()
                logger.info(
                    f'Finished tokenization for {filename} in {tokenization_end_time - tokenization_start_time:.0f}s '
                    f'({len(doc) / (tokenization_end_time - tokenization_start_time):.0f}tok/s)')

                doc._.filename = filename
                nlp(doc)

            end_time = time.monotonic()
            logger.info(f'Finished processing {filename} in {(end_time - start_time):.0f}s')

            file_pbar.update(size)
            file_pbar.set_description_str(f'{i + 1}/{len(filenames)}')
            yield filename, doc
    file_pbar.close()


def create_pipe():
    nlp = spacy.blank("de")
    nlp.add_pipe('tagger_someweta')
    nlp.add_pipe('tagger_rnntagger')
    nlp.add_pipe('lemma_rnntagger')
    nlp.add_pipe('parser_parzu_parallelized', config={'num_processes': torch.get_num_threads()})
    nlp.add_pipe('speech_redewiedergabe')
    nlp.add_pipe('scene_segmenter')
    nlp.add_pipe('coref_uhhlt')
    nlp.add_pipe('ner_flair')
    nlp.add_pipe('events_uhhlt')
    nlp.add_pipe('character_recognizer')

    return nlp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLP Pipeline for literary texts written in German.')
    parser.add_argument('-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('--version', action='version', version=llpro.__version__)
    parser.add_argument('--no-normalize-tokens', action='store_false', dest='normalize_tokens',
                        help='Do not normalize tokens.')
    parser.add_argument('--tokenized', action='store_true',
                        help='Skip tokenization, and assume that tokens are separated by whitespace.')
    parser.add_argument('--sentencized', action='store_true',
                        help='Skip sentence splitting, and assume that sentences are separated by newline characters.')
    parser.add_argument('--paragraph-pattern', metavar='PAT', type=str, default=None,
                        help='Optional paragraph separator pattern. Paragraph separators are removed, and sentences '
                             'always terminate on paragraph boundaries. Performed before tokenization/sentence '
                             'splitting.')
    parser.add_argument('--section-pattern', metavar='PAT', type=str, default=None,
                        help='Optional sectioning paragraph pattern. Paragraphs fully matching the pattern are '
                             'removed. Performed before tokenization/sentence splitting.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stdout', default=True, help='Write all processed tokens to stdout.',
                       action='store_const', dest='outtype', const='stdout')
    group.add_argument('--writefiles', metavar='DIR',
                       help='For each input file, write processed tokens to a separate file in DIR.', default=None)
    parser.add_argument('--infiles', metavar='FILE', type=str, nargs='+', help='Input files, or directories.', required=True)
    parser.set_defaults(outtype='stdout')
    args = parser.parse_args()
    if args.writefiles is not None:
        args.outtype = 'files'
    logging.basicConfig(level=args.loglevel)
    logging.captureWarnings(True)
    try:
        import flair
        for hdl in logging.getLogger('flair').handlers:
            logging.getLogger('flair').removeHandler(hdl)
        logging.getLogger('flair').propagate = True
    except:
        pass
    logger.info('Picked up following arguments: ' + repr(vars(args)))
    logger.info('LLpro version: ' + llpro.__version__)
    logger.info('LLpro resources root: ' + llpro.LLPRO_RESOURCES_ROOT)
    logger.info('LLpro temporary directory: ' + llpro.LLPRO_TEMPDIR)

    if args.version:
        print(llpro.__version__)
        sys.exit(0)

    if 'OMP_NUM_THREADS' not in os.environ:
        torch.set_num_threads(get_cpu_limit())

    if torch.cuda.is_available():
        logger.info(f'torch: CUDA available, version {torch.version.cuda}, architectures {torch.cuda.get_arch_list()}')
    else:
        logger.info('torch: CUDA not available')

    logger.info(f'torch: num_threads is {torch.get_num_threads()}')

    filenames = []
    for f in args.infiles:
        if not os.path.exists(f):
            logger.error(f'file {f} does not exist, aborting!')
            sys.exit(1)

        if os.path.isfile(f):
            filenames.append(f)
        else:
            for root, dirs, members in os.walk(f, followlinks=True):
                filenames.extend([os.path.join(root, m) for m in members])

    logger.info('Loading pipeline')
    nlp = create_pipe()

    tokenizer = SoMaJoTokenizer(nlp.vocab, normalize=args.normalize_tokens, is_pretokenized=args.tokenized,
                                is_presentencized=args.sentencized, paragraph_separator=args.paragraph_pattern,
                                section_pattern=args.section_pattern)
    for filename, tagged_doc in run_pipeline_on_files(filenames, nlp, tokenizer):
        if args.writefiles:
            with open(os.path.join(args.writefiles, os.path.basename(filename) + '.tsv'), 'w') as output_file:
                logger.info(f'writing processed tokens of {filename} to {output_file.name}')
                print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=output_file)
        elif args.outtype == 'stdout':
            print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=sys.stdout)
