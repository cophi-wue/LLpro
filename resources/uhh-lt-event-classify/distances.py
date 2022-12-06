"""
Calculate distances for d-prose jsonlines file
"""
import os
from itertools import combinations_with_replacement
from typing import List

import numpy as np
import pandas as pd
import typer
import ujson
from dtw import dtw
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm

from event_classify.datasets import EventType
from event_classify.util import smooth_bins

app = typer.Typer()


def get_narrativity_scores(data, smoothing_span=30):
    scores = []
    for span in data["annotations"]:
        scores.append(
            EventType.from_tag_name(span["predicted"]).get_narrativity_score()
        )
    series = (
        pd.Series(scores).rolling(smoothing_span, center=True, win_type="cosine").mean()
    )
    series = series[series.notnull()]
    out = int(data["dprose_id"]), [
        np.mean(e) for e in np.array_split(series.to_numpy(), 50)
    ]
    return out


@app.command()
def calculate_distances(dprose_json: str, out_path: str, display: List[int] = []):
    series = {}
    jsonlines = open(dprose_json, "r")
    for i, line in tqdm(enumerate(jsonlines)):
        doc = ujson.loads(line)
        dprose_id, scores = get_narrativity_scores(doc)
        series[dprose_id] = list(scores)
        if dprose_id in display:
            plt.plot(scores, label=doc["title"])
    plt.legend()
    plt.show()
    df = pd.DataFrame(None, columns=sorted(series.keys()), index=sorted(series.keys()))
    # Diagonal is zeros
    for el in sorted(series.keys()):
        df[el][el] = 0
    for (outer_id, outer_scores), (inner_id, inner_scores) in tqdm(
        combinations_with_replacement(series.items(), 2)
    ):
        if outer_id != inner_id:
            dist = dtw(outer_scores, inner_scores).normalizedDistance
            # dist = np.mean(np.abs(np.array(outer_scores) - np.array(inner_scores)))
            df[outer_id][inner_id] = dist
            df[inner_id][outer_id] = dist
    df.to_pickle(out_path)
    print(df)


if __name__ == "__main__":
    app()
