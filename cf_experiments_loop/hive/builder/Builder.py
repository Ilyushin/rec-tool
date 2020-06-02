import turicreate as tc
import os
from turicreate.toolkits._model import Model

class Builder:
    def __init__(self, target: str, **kwargs):
        self._target              = target
        self._extraModelArguments = kwargs

    def build(self, data: tc.SFrame) -> Model:
        return tc.ranking_factorization_recommender.create(
            data,
            target=self._target,
            solver='ials',
            **self._extraModelArguments
        )

