import conllu
import spacy
from spacy.language import Language
from spacy.pipeline import DependencyParser
from spacy.tokens import Token


@Language.factory("parzu_parser")
class ParZuParser(DependencyParser):
    def __init__(self, nlp, name, timeout=1000, **kwargs):
        import parzu_class as parzu

        options = parzu.process_arguments()
        self.parzu = parzu.Parser(options, timeout)

    def __call__(self, doc):
        input_data = ""
        input_tokens = []
        for sent in doc.sents:
            for t in sent:
                if t.tag_ != "_SP":
                    input_data += f"{t.text}\n"
                    input_tokens.append(t)
            input_data += "\n"
        input_data += "\n"
        out = self.parzu.main(input_data)
        i = 0
        for sent in out:
            start_of_sent = i
            for parsed in conllu.parse(sent):
                for token in parsed:
                    input_tokens[i]._.custom_dep = token["deprel"]
                    if token["head"] is not None and token["head"] > 0:
                        input_tokens[i].head = input_tokens[
                            int(token["head"]) - 1 + start_of_sent
                        ]
                    i += 1
        return doc
