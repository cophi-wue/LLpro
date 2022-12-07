# import itertools
# import sys
#
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
#
# from event_classify.parser import HermaParser, ParZuParser

# Doc.set_extension("events", default=[])
# Token.set_extension("custom_dep", default=None)


def to_ranges(tokens):
    indexes = sorted(tokens)
    start = None
    end = None
    for pos, index in enumerate(indexes):
        if start is None:
            start = index
        if len(indexes) == pos + 1:
            end = index + 1
        elif indexes[pos + 1] != index + 1:
            end = index + 1
        if start is not None and end is not None:
            yield range(start, end)
            start = None
            end = None


def is_german_verb(tag):
    return tag.startswith("V") and tag.endswith("FIN")


def is_english_verb(tag):
    return tag.startswith("VB") and len(tag) != 2


# @Language.component("event_segmentation")
def event_segmentation(doc):
    processed = set()
    for token in doc:
        out_ranges = []
        if (is_english_verb(token.tag_) or is_german_verb(token.tag_)) and (
            token not in processed
        ):
            recursed = recurse_children(
                token, blacklist=list(processed), whitelist=[token]
            )
            processed |= set(recursed)
            processed.add(token)
            span_indexes = [t.i for t in recursed] + [token.i]
            if token.head.tag_ == "KON":
                span_indexes.append(token.head.i)
            # Ensure uniqueness:
            span_indexes = list(set(span_indexes))
            ranges = list(to_ranges(span_indexes))
            for i, r in enumerate(ranges):
                # if doc[r.stop].text == "Augen":
                start_offset, end_offset = (0, 0)
                try:
                    if i == len(ranges) - 1 and doc[r.stop].tag_ == "$.":
                        if doc[r.start].sent_start == 1 or doc[r.start].i == 0:
                            end_offset += 1
                    if doc[r.stop - 1].tag_ == "$,":
                        end_offset -= 1
                    if doc[r.start].tag_ == "$,":
                        start_offset += 1
                except IndexError:
                    # Reached end of document, that's fine
                    pass
                out_ranges.append(doc[r.start + start_offset : r.stop + end_offset])
            # TODO: add trailing fullstop if its a whole sentence
            # TODO: remove leading commas
            # if token.text == "flimmerten":
            #     breakpoint()
            doc._.events.append(out_ranges)
    return doc


def recurse_children(token, blacklist=(), whitelist=()):
    # We need to whitelist the initial token as we want to stop at any new finite verb
    return list(
        helper_recurse_children(
            list(token.rights) + list(token.lefts),
            blacklist,
            whitelist=list(whitelist) + [token],
        )
    )


def helper_recurse_children(tokens, blacklist=(), whitelist=()):
    """
    Recursively iterates dependency tree, stopping at finite verbs.
    """
    for t in tokens:
        if t in whitelist:
            yield from helper_recurse_children(t.lefts, blacklist=blacklist)
            yield from helper_recurse_children(t.rights, blacklist=blacklist)
            return
        children = t.children
        child_tags = [c.tag_ for c in children]
        has_verb_child = any(
            is_german_verb(t) or is_english_verb(t) for t in child_tags
        )
        is_verb = is_german_verb(t.tag_) or is_english_verb(t.tag_)
        # Skip any empty tokens
        if t.text.strip() == "":
            continue
        # Skip periods at the end of sentences (this could probably break in edge cases)
        if t.tag_ == "$.":
            continue
        if t.tag_.startswith("$"):
            direct_children = list(t.rights) + list(t.lefts)
            if len(direct_children) == 0:
                yield t
            if sum(len(list(recurse_children(t))) for t in direct_children) == 0:
                yield t
                continue
        # Don't recures into relative clauses
        if t.dep_ == "rc":
            continue
        # Don't recurse into modifiers with their own finite verb, they will be picked up independently
        if t.dep_ == "mo" and is_verb:
            continue
        if t.dep_ == "cd" and (has_verb_child or is_verb):
            continue
        if t.dep_ == "cj" and (has_verb_child or is_verb):
            continue
        # if t._.custom_dep.lower() == "gmod" and t.tag_.startswith("V") and t.tag_.endswith("FIN"):
        #     continue
        if t.tag_.lower() == "kon" and has_verb_child:
            continue
        if t.dep_.lower() == "rel" and has_verb_child:
            continue
        if t.dep_.lower() == "neb" and has_verb_child:
            continue
        if t.dep_.lower() == "par" and has_verb_child:
            continue
        if t not in blacklist:
            yield t
        yield from helper_recurse_children(t.lefts, blacklist=blacklist)
        yield from helper_recurse_children(t.rights, blacklist=blacklist)
