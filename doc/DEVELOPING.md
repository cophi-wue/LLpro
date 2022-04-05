# Developing

This document contains information on how the LLP-Pipeline is constructed, and how it can be extended. In general, a
pipeline consists of:

1. a `Tokenizer` that generates `Token` objects, and
2. multiple `Module`s that annotate the tokens.

The
function
```python
llpipeline.common.pipeline_process(tokenizer: Tokenizer,
                                   modules: Iterable[Module],
                                   filenames: Sequence[str]) -> Tuple[str, Sequence[Token]]
```
implements the pipeline by processing each file, performing file reading, tokenization using `tokenizer`, running each module
from `modules` to annotate, and yielding the resulting annotated token sequence. See `main.py` on an example usage
of `llpipeline.common.pipeline_process`.

## The Class `Token`

The object `Token` holds all annotations for some specific token from a document. It stores annotations in a
dictionary `fields: Dict[Tuple[str, str], Any]` with the interpretation that keys of `fields` hold the type of field in
the first component, and the name of the module that annotated this field in the second component,
e.g., `(('pos', 'SoMeWeTa'), 'NN') in tok.fields` means that the module SoMeWeTa annotated the POS tag of that token
with NN.

Access to the fields is implemented in multiple ways:

- `tok.get_field(field)`: returns the value of field; only applicable if this field was set by precisely one module.
- `tok.get_field(field, module_name)`: returns the value of field set by module with name `module_name`.
- `tok.id`, `tok.doc`, ...: shorthand for `tok.get_field('id')`, ... Applicable for
  fields `id, doc, word, sentence, lemma, pos, morph, head, deprel`.

Adding an annotation to a field is implemented by `tok.set_field(field, module_name, value)`.

By the contract of the `Tokenizer` (see below), a token is uniquely indexed by the
tuple `(tok.doc, tok.id, tok.sentence)`.

## The Class `Tokenizer`

Implementations of the abstract class `Tokenizer` implement the method

```python
def tokenize(self, content: str, filename: str = None) -> Iterable[Token]
# [...]
# for [...]:
#     yield Token({
#       ('word', self.name): tok_word,
#       ('doc', self.name): filename,
#       ('id', self.name): i,
#       ('sentence', self.name): s
#     })
```

The general contract is as follows:

- When the implementing subclass is initialized, then the tokenizer should load any required models.
- When ``tokenize(self, content, filename)`` is called, then implementations are expected to tokenize `content`
  initialize `Token` objects, and set their `word`, `doc`, `id` and `sentence` field, and finally yield these tokens.
- For compatibility with the CONLL-U format, field `id` should be an integer index, incrementing with each token, but
  starting at 1 for each new sentence. Similarly, `sentence` starts at 1, increases with each new sentence, and all
  tokens of the same sentence have the same index value.

## The Class `Module`

Implementations of the abstract class `Module` implement the method

```python
def process(self, tokens: Sequence[Token], update_fn: Callable[[int], None], **kwargs) -> None:
    # [...]
    return
```

The general contract is as follows:

- When the implementing subclass is initialized, then the module should load any required models.
- When `process(self, tokens, update_fn)` is called, implementations are expected to modify the tokens in-place, i.e.
  invoke
  ``set_field(field, my_val, self.name)`` on each token of the supplied sequence of tokens.
- Implementations are expected to report progress by calling ``update_fn(x)`` whenever ``x`` new tokens were processed.
- Implementations can expect that the supplied sequence of tokens forms precisely one document.
