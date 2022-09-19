from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from sklearn.metrics import f1_score

@Predictor.register('SeqClassificationPredictor')
class SeqClassificationPredictor(Predictor):
    """
    Predictor for the abstruct model
    """
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        print("************************P*************")
        pred_labels = []
        sentences = json_dict['sentences']
        gold_labels = json_dict["labels"]
        for sentences_loop, _, _, _ in self._dataset_reader.enforce_max_sent_per_example(sentences):
            instance = self._dataset_reader.text_to_instance(sentences=sentences_loop)
            output = self._model.forward_on_instance(instance)
            idx = output['action_probs'].argmax(axis=1).tolist()
            labels = [self._model.vocab.get_token_from_index(i, namespace='labels') for i in idx]
            pred_labels.extend(labels)
        assert len(pred_labels) == len(sentences)
        print("*** F-score: ", f1_score(gold_labels, pred_labels, average="macro"))
        preds = list(zip(sentences, pred_labels))
        print("xxxx", preds)
        return preds
