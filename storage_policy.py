from abc import ABC, abstractmethod
from typing import List
import torch
from numpy import inf
from torch import cat, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from avalanche.benchmarks.utils import (
    AvalancheDataset,
)
from avalanche.models import FeatureExtractorBackbone

class ExemplarsSelectionStrategy(ABC):
    """
    Base class to define how to select a subset of exemplars from a dataset.
    """

    @abstractmethod
    def make_sorted_indices(
        self, strategy, data: AvalancheDataset
    ) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """
        ...

class FeatureBasedExemplarsSelectionStrategy(ExemplarsSelectionStrategy, ABC):
    """Base class to select exemplars from their features"""
    def __init__(self, model: Module, layer_name: str):
        self.feature_extractor = FeatureExtractorBackbone(model, layer_name)

    @torch.no_grad()
    def make_sorted_indices(self, strategy, data: AvalancheDataset) -> List[int]:
        self.feature_extractor.eval()
        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
        features = cat(
            [
                self.feature_extractor(x.to(strategy.device))
                for x, *_ in DataLoader(
                    data,
                    collate_fn=collate_fn,
                    batch_size=strategy.eval_mb_size,
                )
            ]
        )
        return self.make_sorted_indices_from_features(features)

    @abstractmethod
    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        """
        Should return the sorted list of indices to keep as exemplars.

        The last indices will be the first to be removed when cutoff memory.
        """


class HerdingSelectionStrategy(FeatureBasedExemplarsSelectionStrategy):
    """The herding strategy as described in iCaRL.

    It is a greedy algorithm, that select the remaining exemplar that get
    the center of already selected exemplars as close as possible as the
    center of all elements (in the feature space).
    """

    def make_sorted_indices_from_features(self, features: Tensor) -> List[int]:
        # print(features.shape) # [n, 160, 4, 4]
        features = features.reshape(features.shape[0], -1) # [n, 160*4*4]
        # print(features.shape) # [n, 2560]

        selected_indices = []

        center = features.mean(dim=0) # [160, 4, 4]
        current_center = center * 0 # [160, 4, 4]

        for i in range(len(features)):
            # Compute distances with real center
            candidate_centers = current_center * i / (i + 1) + features / ( i + 1) # [3, 160, 4, 4]
            distances = pow(candidate_centers - center, 2).sum(dim=1)
            distances[selected_indices] = inf # [3, 4, 4]
            # Select best candidate
            new_index = distances.argmin().tolist() # int
            selected_indices.append(new_index) # list
            current_center = candidate_centers[new_index]

        print(selected_indices)
        return selected_indices