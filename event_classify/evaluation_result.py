from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class EvaluationResult:
    weighted_f1: Optional[float]
    macro_f1: Optional[float]
    predictions: torch.tensor
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    extra_predictions: Dict[str, torch.tensor] = field(default_factory=dict)
    extra_labels: Dict[str, torch.tensor] = field(default_factory=dict)

    def get_prediction_lists(self):
        from .datasets import EventType, SpeechType

        event_types = [EventType(p.item()) for p in self.predictions]
        iterative = []
        mental = []
        i = 0
        for et in event_types:
            if et != EventType.NON_EVENT:
                iterative.append(bool(self.extra_predictions["iterative"][i].item()))
                mental.append(bool(self.extra_predictions["mental"][i].item))
                i += 1
            else:
                iterative.append(None)
                mental.append(None)
        return {
            "event_types": event_types,
            "speech_type": [
                SpeechType(p.item()) for p in self.extra_predictions["speech_type"]
            ],
            "thought_representation": [
                bool(p.item()) for p in self.extra_predictions["thought_representation"]
            ],
            "iterative": iterative,
            "mental": mental,
        }
