import torch
from collections import Counter

import enum


class GoldLabelStrategy(enum.Enum):
    """
    Describes the strategy used to, for training purposes, assign gold labels to each cluster.

    MOST_RECENT: gold class of the most recently added mention, may bias towards locally correct decisions
    ORIGINAL: gold class of the first mention in the cluster, encourages global consistency
    """
    MOST_RECENT = "most_recent"
    ORIGINAL = "original"


class IncrementalEntities:
    def __init__(self, conf, device, gold_strategy=GoldLabelStrategy.MOST_RECENT):
        self.conf = conf
        self.device = device

        self.emb = torch.tensor([]).to(device)
        self.count = torch.tensor([]).to(device)
        # From class to correct entity
        self.class_gold_entity = {} # [class: gold_entity]
        self.mention_distance = torch.tensor([]).to(device)
        self.sentence_distance = torch.tensor([]).to(device)
        self.mention_to_cluster_id = {}
        self.gold_strategy = gold_strategy

    def _check_integrity(self):
        size = self.emb.shape[0]
        assert self.count.shape[0] == size
        assert max(self.class_gold_entity.values()) < size
        assert self.mention_distance.shape[0] == size
        assert self.sentence_distance.shape[0] == size
        assert max(self.mention_to_cluster_id.values()) < size

    def __len__(self):
        return len(self.emb)

    def evict(self, evict_to=None):
        """
        Evicts entities that are older than the specified thresholds.
        """
        offset = 0
        for i, distance in enumerate(self.mention_distance.clone()):
            distance = distance.item()
            if (
                distance > self.conf["unconditional_eviction_limit"]
                or (distance > self.conf["singleton_eviction_limit"] and self.count[i - offset] == 1)
            ) and len(self) > 1:
                if evict_to is not None:
                    evict_to._add_entity(
                        self.emb[i - offset].to(evict_to.device),
                        None,  # We don't retain this as the evicted copy is not used for loss computation
                        [span for span, entity in self.mention_to_cluster_id.items() if entity == (i - offset)],
                        0,
                        count=self.count[i - offset].to(evict_to.device),
                        sentence_distance=self.sentence_distance[i - offset].to(evict_to.device),
                        mention_distance=self.mention_distance[i - offset].to(evict_to.device),
                    )
                self.emb = torch.cat(
                    [self.emb[: i - offset], self.emb[i + 1 - offset :]], 0
                ).to(self.device)
                self.sentence_distance = torch.cat(
                    [
                        self.sentence_distance[: i - offset],
                        self.sentence_distance[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                self.count = torch.cat(
                    [
                        self.count[: i - offset],
                        self.count[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                self.mention_distance = torch.cat(
                    [
                        self.mention_distance[: i - offset],
                        self.mention_distance[i + 1 - offset :],
                    ],
                    0,
                ).to(self.device)
                new_classes = {}
                for class_, entity_index in self.class_gold_entity.items():
                    if entity_index == i - offset:
                        pass
                    elif entity_index > i - offset:
                        new_classes[class_] = entity_index - 1
                    else:
                        new_classes[class_] = entity_index
                self.class_gold_entity = new_classes
                new_cluster_ids = {}
                for span, cluster in self.mention_to_cluster_id.items():
                    if cluster == i - offset:
                        pass
                    elif cluster > i - offset:
                        new_cluster_ids[span] = cluster - 1
                    elif cluster < i - offset:
                        new_cluster_ids[span] = cluster
                self.mention_to_cluster_id = new_cluster_ids
                offset += 1

    def _add_entity(self, emb, gold_class, spans, offset, sentence_distance=None,
                    count=None, mention_distance=None):
        if sentence_distance is None:
            sentence_distance = torch.zeros(1).unsqueeze(0).to(self.device)
        else:
            sentence_distance = sentence_distance.unsqueeze(0)
        if count is None:
            count = torch.ones(1).unsqueeze(0).type(torch.long).to(self.device)
        else:
            count = count.unsqueeze(0)
        if mention_distance is None:
            mention_distance = torch.ones(1).to(self.device).unsqueeze(0)
        else:
            mention_distance = mention_distance.unsqueeze(0)
        if len(self) == 0:
            self.emb = emb.unsqueeze(0).to(self.device)
            self.count = count
            if gold_class is not None:
                self.class_gold_entity[gold_class] = 0
            self.sentence_distance = sentence_distance
            self.mention_distance = (
                mention_distance
            )
            for (span_start, span_end) in spans:
                span_start += offset
                span_end += offset
                self.mention_to_cluster_id[(span_start, span_end)] = 0
        else:
            self.emb = torch.cat([self.emb, emb.unsqueeze(0)])
            self.count = torch.cat([self.count, count])
            self.sentence_distance = torch.cat(
                [self.sentence_distance, sentence_distance]
            )
            self.mention_distance = torch.cat(
                [self.mention_distance, mention_distance]
            )
            if gold_class is not None:
                entity_gold_class = self.class_gold_entity.get(gold_class)
                if self.gold_strategy != GoldLabelStrategy.ORIGINAL or entity_gold_class is None:
                    self.class_gold_entity[gold_class] = self.emb.shape[0] - 1
            for (span_start, span_end) in spans:
                span_start += offset
                span_end += offset
                self.mention_to_cluster_id[(span_start, span_end)] = (
                    self.emb.shape[0] - 1
                )

    def add_entity(self, emb, gold_class, span_start, span_end, offset):
        self._add_entity(emb, gold_class, [(span_start.item(), span_end.item())], offset)
        self.mention_distance += 1

    def update_entity(
        self,
        cluster_to_update,
        emb,
        gold_class,
        span_start,
        span_end,
        update_gate,
        offset=0,
    ):
        span_start += offset
        span_end += offset
        if gold_class is not None:
            if self.gold_strategy == GoldLabelStrategy.MOST_RECENT:
                self.class_gold_entity[gold_class] = cluster_to_update.item()
        self.mention_to_cluster_id[
            (span_start.item(), span_end.item())
        ] = cluster_to_update.item()
        # High values in update gate mean the old representation is mostly replaced
        self.emb = self.emb.clone()
        self.emb[cluster_to_update] = (
            update_gate * emb
            + (torch.tensor(1) - update_gate) * self.emb[cluster_to_update].clone()
        )
        self.count[cluster_to_update] += 1
        self.mention_distance[cluster_to_update] = 0
        self.mention_distance += 1

    def extend(self, other):
        """
        Extend entity collection with existing one.

        In the process the information required for loss computation is lost.
        """
        for i in range(len(other.emb)):
            # TODO: this can be optimized by building a dictionary outside the for loop
            cluster_ids = [span for span, entity in other.mention_to_cluster_id.items() if entity == i]
            self._add_entity(
                other.emb[i].to(self.device),
                None,
                cluster_ids,
                0,
                count=other.count[i].to(self.device),
                sentence_distance=other.sentence_distance[i].to(self.device),
                mention_distance=other.mention_distance[i].to(self.device),
            )

    def get_result(self, remove_singletons=True):
        """
        Returns 4-tuple (span_starts, span_ends, mention_to_cluster_id, predicted_clusters) representing the entities in the cluster.
        """
        span_starts = []
        span_ends = []
        if remove_singletons:
            counter = Counter(self.mention_to_cluster_id.values())
            non_singletons = set(value for value, count in counter.items() if count > 1)
            cluster_mapping = {v:k for k,v in enumerate(non_singletons)}
            mention_to_cluster_id = {k: cluster_mapping[v] for k, v in self.mention_to_cluster_id.items() if v in non_singletons}
        else:
            mention_to_cluster_id = self.mention_to_cluster_id
        if len(mention_to_cluster_id) == 0:
            predicted_clusters_list = []
        else:
            predicted_clusters_list = [list() for _ in range((max(mention_to_cluster_id.values()) + 1))]
        for (start, end), cluster_id in mention_to_cluster_id.items():
            span_starts.append(start)
            span_ends.append(end)
            predicted_clusters_list[cluster_id].append((start, end))
        if len(mention_to_cluster_id) != 0:
            assert len(predicted_clusters_list) == (max(mention_to_cluster_id.values()) + 1)
        return span_starts, span_ends, mention_to_cluster_id, predicted_clusters_list
