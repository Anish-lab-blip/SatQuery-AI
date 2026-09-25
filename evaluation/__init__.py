"""SatQuery AI evaluation: manifests, leakage controls, metrics, runners."""

from evaluation.leakage import (
    PUBLIC_TEST_DIRNAME,
    LeakageReport,
    PublicTestFirewall,
    assert_no_hidden_access,
    assert_no_scene_overlap,
    assign_splits_by_scene,
    audit_manifest,
    deduplicate,
)
from evaluation.manifests import (
    DatasetManifest,
    RunManifest,
    SampleRecord,
    geographic_hash,
)

__all__ = [
    "DatasetManifest",
    "RunManifest",
    "SampleRecord",
    "geographic_hash",
    "LeakageReport",
    "PublicTestFirewall",
    "PUBLIC_TEST_DIRNAME",
    "assign_splits_by_scene",
    "audit_manifest",
    "assert_no_scene_overlap",
    "assert_no_hidden_access",
    "deduplicate",
]