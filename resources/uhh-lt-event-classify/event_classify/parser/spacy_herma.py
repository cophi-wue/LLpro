import os
import subprocess
import tempfile

import conllu
from spacy.language import Language
from spacy.pipeline import DependencyParser

HERMA_PATH = os.environ.get("HERMA_PATH")


@Language.factory("herma_parser")
class HermaParser(DependencyParser):
    def __init__(self, nlp, name):
        pass

    def __call__(self, doc):
        input_data = ""
        input_tokens = []
        for sent in doc.sents:
            for t in sent:
                if t.tag_ != "_SP":
                    input_data += f"{t.text}\n"
                    input_tokens.append(t)
            input_data += "\n"
        if HERMA_PATH is None:
            raise ValueError(
                "'HERMA_PATH' environment variable must be set to use the parser!"
            )
        tokens = open(HERMA_PATH + "/Pipeline/01_tokens/text.txt", "w")
        for sent in doc.sents:
            for t in sent:
                if t.tag_ != "_SP":
                    tokens.write(f"{t.text}\n")
                    input_tokens.append(t)
            tokens.write("\n")
        tokens.close()
        subprocess.run(
            [
                "sh",
                "tag.sh",
                HERMA_PATH + "/Pipeline/01_tokens",
                HERMA_PATH + "/Pipeline/02_tags/",
            ],
            cwd=HERMA_PATH + "/Tools/Tagger",
        )
        subprocess.run(
            [
                "python",
                "germalemma_lemmatize_conllx_fallback_dir.py",
                HERMA_PATH + "/Pipeline/02_tags/Ensemble",
                HERMA_PATH + "/Pipeline/03_lemmata",
                "Vollformen_geschlossene_Wortklassen_final.txt",
            ],
            cwd=HERMA_PATH + "/Tools/Lemmatizer",
        )
        subprocess.run(
            [
                "sh",
                "parse.sh",
                HERMA_PATH + "/Pipeline/03_lemmata",
                HERMA_PATH + "/Pipeline/04_parse",
            ],
            cwd=HERMA_PATH + "/Tools/Parser",
        )
        i = 0
        for sent in conllu.parse_incr(open(HERMA_PATH + "/Pipeline/04_parse/text.txt")):
            start_of_sent = i
            for token in sent:
                input_tokens[i]._.custom_dep = token["deprel"]
                if token["head"] is not None and token["head"] > 0:
                    input_tokens[i].head = input_tokens[
                        int(token["head"]) - 1 + start_of_sent
                    ]
                i += 1
        # subprocess.run(
        #     ["sh", "clear.sh"],
        #     cwd=HERMA_PATH,
        # )
        return doc
