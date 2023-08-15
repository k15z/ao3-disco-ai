import logging
import os
import pickle
from typing import List

import torch
import yaml

from ao3_disco_ai.arch import SparseNNV0
from ao3_disco_ai.feature import FeatureExtractor
from ao3_disco_ai.model import DiscoModel
from ao3_disco_ai.structs import Work


class ModelWrapper:
    def __init__(
        self, feature_extractor: FeatureExtractor, model_arch: SparseNNV0
    ) -> None:
        self.feature_extractor = feature_extractor
        self.model_arch = model_arch

    def embed(self, works: List[Work]):
        dense, sparse = self.feature_extractor.transform(works)
        pt_dense = torch.FloatTensor(dense)
        pt_sparse = {}
        for name in sparse:
            id_list, offsets = [], []
            for row in sparse[name]:
                offsets.append(len(id_list))
                id_list.extend(row)
            id_list, offsets = torch.IntTensor(id_list), torch.IntTensor(offsets)
            pt_sparse[name] = (id_list, offsets)
        return self.model_arch(pt_dense, pt_sparse)

    @classmethod
    def build(cls, model_dir, checkpoint):
        with open(os.path.join(model_dir, "hparams.yaml"), "rt") as fin:
            hparams = yaml.safe_load(fin)
            data_dir = hparams["dataset"]
        with open(os.path.join(data_dir, "features.pkl"), "rb") as fin:
            feature_store = pickle.load(fin)
        logging.info("Preparing model %s for bundling...", model_dir)
        logging.info("Preparing dataset %s for bundling...", data_dir)

        model = DiscoModel.load_from_checkpoint(
            os.path.join(model_dir, "checkpoints", checkpoint),
            feature_store=feature_store,
        )
        wrapper = ModelWrapper(feature_store._extractor, model.arch)

        with open(os.path.join(data_dir, "works.pkl"), "rb") as fin:
            work_to_json = pickle.load(fin)
            example_works = list(work_to_json.values())[:10]
            result = wrapper.embed([Work(**x) for x in example_works])
            logging.info("Example result: %s", result.shape)

        return wrapper
