import argparse
import logging
import multiprocessing
import os
import sys
import time
from omegaconf import OmegaConf

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
            logger.info(f'Start processing {filename}')
            try:
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
            except Exception as e:
                logger.exception('Failed to process %s: %s', filename, e)
                file_pbar.update(size)
                file_pbar.set_description_str(f'{i + 1}/{len(filenames)}')
    file_pbar.close()


def create_pipe(component_config):
    nlp = spacy.blank("de")
    components = [
        'tagger_someweta',
        'tagger_rnntagger',
        'lemma_rnntagger',
        'parser_parzu',
        'speech_redewiedergabe',
        'scene_segmenter',
        'coref_uhhlt',
        'ner_flair',
        'events_uhhlt',
        'character_recognizer',
    ]
    if os.getenv('LLPRO_EXPERIMENTAL', 'no').lower() in {'true', '1', 'y', 'yes'}:
        components.extend([
            'emotion_classifier'
        ])
    
    for component in components:
        config = dict(component_config.get(component, {}))
        if config.get('disable', False):
            continue
        nlp.add_pipe(component, config=config)

    return nlp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLP Pipeline for literary texts written in German.')
    parser.add_argument('-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('--version', action='version', version=llpro.__version__)
    parser.add_argument('-X', '--component-config', metavar='OPT', type=str, help='Component parameters of the form component_name.opt=value', 
                        required=False, action='append')
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

    component_config = OmegaConf.from_dotlist(args.component_config or [])
    logger.info('Picked up following component config overwrites:\n\n' +  OmegaConf.to_yaml(component_config))

    if 'OMP_NUM_THREADS' not in os.environ:
        try:
            torch.set_num_threads(get_cpu_limit())
        except FileNotFoundError:
            logger.warning(f'cgroup: could not retrieve cpu limit')

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
    nlp = create_pipe(component_config)

    tokenizer = SoMaJoTokenizer(nlp.vocab, pbar=True, **component_config.get('somajo_tokenizer', {}))
    for filename, tagged_doc in run_pipeline_on_files(filenames, nlp, tokenizer):
        if args.writefiles:
            with open(os.path.join(args.writefiles, os.path.basename(filename) + '.tsv'), 'w') as output_file:
                logger.info(f'writing processed tokens of {filename} to {output_file.name}')
                print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=output_file)
        elif args.outtype == 'stdout':
            print(spacy_doc_to_dataframe(tagged_doc).to_csv(None, sep='\t', index=True), file=sys.stdout)
