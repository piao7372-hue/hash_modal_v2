"""Stage 1 dataset preprocessing interfaces and adapter registry."""

from .base_dataset import BaseDatasetAdapter, DatasetProtocol, ProtocolCounts, SampleRecord
from .mirflickr25k_adapter import MIRFlickr25KAdapter
from .mscoco_adapter import MSCOCOAdapter
from .nuswide_adapter import NUSWIDEAdapter

DATASET_ADAPTERS = {
    "mirflickr25k": MIRFlickr25KAdapter,
    "nuswide": NUSWIDEAdapter,
    "mscoco": MSCOCOAdapter,
}

SUPPORTED_DATASETS = tuple(DATASET_ADAPTERS.keys())

__all__ = [
    "BaseDatasetAdapter",
    "DatasetProtocol",
    "ProtocolCounts",
    "SampleRecord",
    "DATASET_ADAPTERS",
    "SUPPORTED_DATASETS",
    "MIRFlickr25KAdapter",
    "NUSWIDEAdapter",
    "MSCOCOAdapter",
]
