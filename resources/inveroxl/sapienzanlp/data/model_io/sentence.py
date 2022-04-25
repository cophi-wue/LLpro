from typing import List

from sapienzanlp.data.model_io.word import Word, Predicate


class Sentence:
    """
    A sentence class, containing a list of `Word`.

    # Parameters

    _words: `List[Word]`
        List of `Word` objects

    """

    def __init__(self, words: List[Word] = None):
        self._words = words or []

    def __len__(self):
        return len(self._words)

    def __getitem__(self, item):
        return self._words[item]

    def __repr__(self):
        return " ".join(w.text for w in self._words)

    def __str__(self):
        return self.__repr__()

    def append(self, word: Word):
        self._words.append(word)


class SrlSentence(Sentence):
    """
    A Semantic Role Labeling Sentence class, used to built the output.

    # Parameters

    _words: `List[Word]`
        List of `Word` objects

    predicates: `List[Predicate]`
        List of `Word` that are `Predicate`

    """

    def __init__(self, words: List[Word] = None, predicates: List[Predicate] = None):
        super(SrlSentence, self).__init__(words)
        self.predicates: List[Predicate] = predicates or []

    def get_predicates(self) -> List[Predicate]:
        return self.predicates

    def add_predicate(self, predicate: Predicate):
        self.predicates.append(predicate)
