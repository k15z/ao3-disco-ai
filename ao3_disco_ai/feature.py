import logging
from random import shuffle
from typing import Dict, List

import numpy as np
from pydantic import validate_call
from sklearn.preprocessing import LabelEncoder, RobustScaler

from ao3_disco_ai.structs import Work


class FeatureExtractor:
    def __init__(self):
        self._fitted = False
        self._dense_fe = DenseFeatureExtractor()
        self._sparse_fe = SparseFeatureExtractor()

    def metadata(self):
        assert self._fitted
        return self._dense_fe.metadata(), self._sparse_fe.metadata()

    @validate_call
    def fit(self, works: List[Work]):
        self._fitted = True
        self._dense_fe.fit(works)
        self._sparse_fe.fit(works)

    @validate_call
    def transform(self, works: List[Work]):
        assert self._fitted
        return self._dense_fe.transform(works), self._sparse_fe.transform(works)


class DenseFeatureExtractor:
    def __init__(self):
        self._dense_features = [
            "hits",
            "kudos",
            "words",
            "comments",
            "bookmarks",
            "complete_chapters",
            "total_chapters",
            "is_complete",
            "words_per_chapter",
            "publish_year",
            "update_year",
            "days_between_publish_update",
            "days_per_chapter",
            "characters_in_summary",
            "words_in_summary",
            "5times_in_title",
            "hiatus_in_summary",
        ]
        self._dense_dims = len(self._dense_features)

    def metadata(self):
        return self._dense_features

    @validate_call
    def fit(self, works: List[Work]):
        self._scaler = RobustScaler()
        self._scaler.fit(self._raw_transform(works))
        logging.info("Dense features: %s", self.metadata())

    @validate_call
    def transform(self, works: List[Work]) -> np.ndarray:
        return self._scaler.transform(self._raw_transform(works))

    def _raw_transform(self, works: List[Work]):
        dense = np.zeros((len(works), self._dense_dims))
        dense[:, 0] = [x.statistics.hits for x in works]
        dense[:, 1] = [x.statistics.kudos for x in works]
        dense[:, 2] = [x.statistics.words for x in works]
        dense[:, 3] = [x.statistics.comments for x in works]
        dense[:, 4] = [x.statistics.bookmarks for x in works]

        def _parse_chapters(x):
            x = x.replace(",", "")
            if "/" not in x:
                complete_chapters = int(x)
                total_chapters = complete_chapters
            else:
                complete_chapters, total_chapters = x.split("/")
                complete_chapters = int(complete_chapters)
                total_chapters = int(total_chapters) if total_chapters != "?" else 0
            is_complete = float(complete_chapters == total_chapters)
            return [complete_chapters, total_chapters, is_complete]

        dense[:, 5:8] = np.array(
            [_parse_chapters(x.statistics.chapters) for x in works]
        )
        dense[:, 8] = dense[:, 2] / dense[:, 5]  # words-per-chapter

        for x in works:
            if not x.statistics.status:
                x.statistics.status = x.statistics.published
        dense[:, 9] = [x.statistics.published.year for x in works]
        dense[:, 10] = [x.statistics.status.year for x in works]
        dense[:, 11] = [
            (x.statistics.status - x.statistics.published).days for x in works
        ]
        dense[:, 12] = dense[:, 11] / dense[:, 5]  # days-per-chapter

        dense[:, 13] = [len(x.summary) for x in works]
        dense[:, 14] = [len(x.summary.split()) for x in works]

        dense[:, 15] = ["5 times" in x.title.lower() for x in works]
        dense[:, 16] = [
            ("hiatus" in x.summary.lower() or "abandoned" in x.summary.lower())
            for x in works
        ]

        return dense


class SparseFeatureExtractor:
    def __init__(self):
        self._label_encoders = {}

    def metadata(self):
        return {k: len(v.classes_) for k, v in self._label_encoders.items()}

    @validate_call
    def fit(self, works: List[Work]):
        self._build_label_encoder("fandom", works)
        self._build_label_encoder("rating", works)
        self._build_label_encoder("category", works)
        self._build_label_encoder("relationship", works)
        self._build_label_encoder("category", works)
        self._build_label_encoder("freeform", works)
        self._build_label_encoder("author", works)
        logging.info("Sparse features: %s", self.metadata())

    @validate_call
    def transform(self, works: List[Work]) -> Dict[str, List[List[int]]]:
        sparse_feats = {}
        for key, le in self._label_encoders.items():
            if key == "author":  # Special handling!
                tag_values = [x for work in works for x in work.authors]
                tag_counts = [len(work.authors) for work in works]
            else:
                tag_values = [x for work in works for x in work.tags.__dict__[key]]
                tag_counts = [len(work.tags.__dict__[key]) for work in works]
            tag_ids = le.transform(self._normalize_tag_values(tag_values))
            sparse_feats[key] = []
            for tag_count in tag_counts:
                sparse_feats[key].append(tag_ids[:tag_count])
                tag_ids = tag_ids[tag_count:]
        return sparse_feats

    def _build_label_encoder(self, key, works):
        le = LabelEncoder()
        if key == "author":  # Special handling!
            x = [x for work in works for x in work.authors]
        else:
            x = [x for work in works for x in work.tags.__dict__[key]]
        shuffle(x)
        le.fit(self._normalize_tag_values(x))
        self._label_encoders[key] = le

    def _normalize_tag_values(self, values):
        return [x.lower().replace(" ", "_") for x in values]


class FeatureStore:
    def __init__(self, work_to_json, extractor=None):
        self._work_ids = list(work_to_json.keys())
        self._work_id_to_i = {k: v for v, k in enumerate(self._work_ids)}
        work_jsons = [work_to_json[x] for x in self._work_ids]

        self._extractor = extractor or FeatureExtractor()
        logging.info(f"Fitting feature store to {len(work_to_json)} works.")
        self._extractor.fit(work_jsons)
        logging.info("Precomputing feature store.")
        self._dense_features, self._sparse_features = self._extractor.transform(
            work_jsons
        )
        logging.info("Feature store ready.")

    def metadata(self):
        return self._extractor.metadata()

    def get_features(self, work_ids):
        idx = [self._work_id_to_i[x] for x in work_ids]
        dense = self._dense_features[idx]
        sparse = {}
        for k, v in self._sparse_features.items():
            sparse[k] = [v[i] for i in idx]
        return dense, sparse
