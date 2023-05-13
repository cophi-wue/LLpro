import os

__version__ = '0.1.0'
LLPRO_RESOURCES_ROOT = os.getenv('LLPRO_RESOURCES_ROOT',
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources')))
LLPRO_TEMPDIR = os.getenv('LLPRO_TEMPDIR',
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'tmp')))

from .components.tokenizer_somajo import SoMaJoTokenizer
from .components.tagger_someweta import tagger_someweta
from .components.tagger_rnntagger import tagger_rnntagger
from .components.lemma_rnntagger import lemma_rnntagger
from .components.parser_parzu import parser_parzu_parallelized
from .components.speech_redewiedergabe import speech_redewiedergabe
from .components.su_scene_segmenter import su_scene_segmenter
from .components.coref_uhhlt import coref_uhhlt
from .components.ner_flair import ner_flair
from .components.events_uhhlt import events_uhhlt
from .components.character_recognizer import character_recognizer
