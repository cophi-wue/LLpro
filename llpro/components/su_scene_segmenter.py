import logging
import sys
from pathlib import Path

import torch
from spacy import Language
from spacy.tokens import Doc, Span, Token
from typing import Callable

from ..common import Module
from .. import LLPRO_RESOURCES_ROOT

logger = logging.getLogger(__name__)

@Language.factory("su_scene_segmenter", assigns=['doc._.scenes', 'token._.scene'], default_config={
    'stss_se_home': LLPRO_RESOURCES_ROOT + '/su-scene-segmenter',
    'model_path': LLPRO_RESOURCES_ROOT + '/extracted-scene-segmenter-model', 'use_cuda': True, 'device_on_run': True,
    'pbar_opts': None
})
def su_scene_segmenter(nlp, name, stss_se_home, model_path, use_cuda, device_on_run, pbar_opts):
    if not Doc.has_extension('scenes'):
        Doc.set_extension('scenes', default=list())
    if not Token.has_extension('scene'):
        Token.set_extension('scene', default=None)
    return SceneSegmenter(name, stss_se_home=stss_se_home, model_path=model_path, use_cuda=use_cuda, device_on_run=device_on_run, pbar_opts=pbar_opts)

class SceneSegmenter(Module):

    def __init__(self, name, stss_se_home=LLPRO_RESOURCES_ROOT + '/su-scene-segmenter', model_path=LLPRO_RESOURCES_ROOT + '/extracted-scene-segementer-model', use_cuda=True,
                 device_on_run=True, pbar_opts=None):
        super().__init__(name, pbar_opts=pbar_opts)
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else "cpu")
        self.device_on_run = device_on_run
        self.stss_se_home = Path(stss_se_home)
        sys.path.insert(0, str(self.stss_se_home))
        from su_scene_segmenter_code.sequential_sentence_classification.model import SeqClassificationModel
        from su_scene_segmenter_code.sequential_sentence_classification.dataset_reader import SeqClassificationReader
        import allennlp.models.archival
        self.archive = allennlp.models.archival.load_archive(str(model_path))
        self.archive.model.eval()
        if not self.device_on_run:
            self.archive.model.to(self.device)

    def before_run(self):
        self.archive.model.to(self.device)
        logger.info(f"{self.name} using device {self.archive.model._get_prediction_device()}")

    def after_run(self):
        if self.device_on_run:
            self.archive.model.to('cpu')
            torch.cuda.empty_cache()

    def process(self, doc: Doc, progress_fn: Callable[[int], None]) -> Doc:
        sentences = list(doc.sents)
        # Joining the words with ' ' should be a sufficiently good approximation of the original text,
        # as the text is further tokenized by BERT in the scene segmenter anyway, which always splits at whitespace.
        prepared_sentences = [' '.join(tok.text for tok in sent) for sent in sentences]

        # cf. resources/su-scene-segmenter/su_scene_segmenter_code/sequential_sentence_classification/predictor.py
        sentence_counter = 0
        pred_labels = []
        with torch.no_grad():
            for sentences_loop, _, _, _ in self.archive.dataset_reader.enforce_max_sent_per_example(prepared_sentences):
                instance = self.archive.dataset_reader.text_to_instance(sentences_loop, predict=True)
                self.archive.dataset_reader.apply_token_indexers(instance)
                output = self.archive.model.forward_on_instance(instance)
                idx = output['action_probs'].argmax(axis=1).tolist()
                labels = [self.archive.model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
                for l in labels:
                    pred_labels.append((l, sentence_counter))
                    progress_fn(len(sentences[sentence_counter]))
                    sentence_counter = sentence_counter + 1

        scenes = self.postprocess(pred_labels)
        for segment_counter, scene in enumerate(scenes):
            scene_obj = Span(doc=doc, start=sentences[scene['begin']][0].i, end=sentences[scene['end']-1][-1].i+1, span_id=segment_counter, label=scene['type'])
            doc._.scenes.append(scene_obj)

            for tok in scene_obj:
                tok._.scene = scene_obj

        return doc

    def postprocess(self, pred_labels):
        # cf. resources/su-scene-segmenter/su_scene_segmenter_code/utils/postprocess.py
        scenes = []
        group = {}
        last_border = 0
        for i, label_offset in enumerate(pred_labels):
            label, offset = label_offset[0].replace("_label", ""), (label_offset[1], label_offset[1] + 1)
            if i == 0:
                prev_l = label.replace("-B", "")
                group = [offset]
            else:
                if "-B" in label:
                    # Non-scene to non-scene change is not allowed so continue expanding last non-scene despite
                    # prediction of Nonscene-B label
                    if label == "Nonscene-B" and prev_l == "Nonscene":
                        group.append(offset)
                    else:  # scene change due to prediction of -B label
                        scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})
                        group = [offset]
                        last_border = scenes[-1]["end"]
                        prev_l = label.replace("-B", "")
                else:
                    if label == prev_l:
                        group.append(offset)
                    else:  # scene change despite lack of -B label
                        scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})
                        group = [offset]
                        last_border = scenes[-1]["end"]
                        prev_l = label.replace("-B", "")
        if group:
            scenes.append({"begin": last_border, "end": group[-1][-1], "type": prev_l})

        return scenes