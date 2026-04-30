"""Unit tests for the in process job store."""

from __future__ import annotations

import pytest

from atop_web.api import jobs as jobs_mod
from atop_web.api.jobs import (
    STAGE_BUILDING_SAMPLES,
    STAGE_DECODING_SSTAT,
    STAGE_DECODING_TSTAT,
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_HEADER,
    STAGE_PENDING,
    STAGE_SCANNING,
    STAGE_UPLOAD_SAVED,
    JobStore,
    stage_label,
)


@pytest.fixture()
def store() -> JobStore:
    return JobStore()


def test_stage_progress_mapping_covers_all_stages():
    mapping = jobs_mod._STAGE_PROGRESS
    # Every exposed stage constant must have an associated progress value.
    for key in (
        STAGE_PENDING,
        STAGE_UPLOAD_SAVED,
        STAGE_HEADER,
        STAGE_SCANNING,
        STAGE_DECODING_SSTAT,
        STAGE_DECODING_TSTAT,
        STAGE_BUILDING_SAMPLES,
        STAGE_DONE,
        STAGE_ERROR,
    ):
        assert key in mapping, key
    # Progress numbers are monotonic up through ``building_samples``; done
    # and error both sit at 100.
    assert mapping[STAGE_PENDING] <= mapping[STAGE_UPLOAD_SAVED]
    assert mapping[STAGE_UPLOAD_SAVED] <= mapping[STAGE_HEADER]
    assert mapping[STAGE_HEADER] <= mapping[STAGE_SCANNING]
    assert mapping[STAGE_SCANNING] <= mapping[STAGE_DECODING_SSTAT]
    assert mapping[STAGE_DECODING_SSTAT] < mapping[STAGE_DECODING_TSTAT]
    assert mapping[STAGE_DECODING_TSTAT] < mapping[STAGE_BUILDING_SAMPLES]
    assert mapping[STAGE_BUILDING_SAMPLES] < mapping[STAGE_DONE]
    assert mapping[STAGE_DONE] == 100
    assert mapping[STAGE_ERROR] == 100


def test_stage_labels_are_english_and_non_empty():
    # Spot check a few stages to guarantee the user facing string is not
    # the raw machine key.
    assert stage_label(STAGE_HEADER) != STAGE_HEADER
    assert stage_label(STAGE_DECODING_SSTAT) != STAGE_DECODING_SSTAT
    assert stage_label(STAGE_DONE) == "Done"
    assert stage_label(STAGE_ERROR) == "Error"
    assert stage_label("unknown-key") == "unknown-key"  # pass through fallback


def test_to_dict_includes_stage_label_and_detail(store: JobStore):
    job = store.create(source="upload", filename="atop_test")
    store.update(job.job_id, stage=STAGE_DECODING_SSTAT, detail="5 / 10 samples")

    data = store.get(job.job_id).to_dict()
    assert data["stage"] == STAGE_DECODING_SSTAT
    assert data["stage_label"] == stage_label(STAGE_DECODING_SSTAT)
    assert data["detail"] == "5 / 10 samples"
    # Schema must carry every required key.
    for key in (
        "job_id",
        "source",
        "status",
        "stage",
        "stage_label",
        "progress",
        "detail",
        "filename",
        "result",
        "error",
    ):
        assert key in data


def test_update_detail_can_be_cleared(store: JobStore):
    job = store.create(source="upload")
    store.update(job.job_id, stage=STAGE_DECODING_SSTAT, detail="1 / 2 samples")
    assert store.get(job.job_id).detail == "1 / 2 samples"
    store.update(job.job_id, detail=None)
    assert store.get(job.job_id).detail is None


def test_progress_never_decreases(store: JobStore):
    job = store.create(source="upload")
    store.update(job.job_id, progress=30)
    store.update(job.job_id, progress=20)  # late / stale update
    assert store.get(job.job_id).progress == 30
    store.update(job.job_id, progress=55)
    assert store.get(job.job_id).progress == 55


def test_mark_done_clears_detail(store: JobStore):
    job = store.create(source="upload")
    store.update(job.job_id, stage=STAGE_DECODING_TSTAT, detail="9 / 10 samples")
    store.mark_done(job.job_id, {"session": "abc"})
    snap = store.get(job.job_id).to_dict()
    assert snap["status"] == "done"
    assert snap["stage"] == STAGE_DONE
    assert snap["progress"] == 100
    assert snap["detail"] is None


def test_stage_advances_progress_to_floor(store: JobStore):
    # Advancing to a named stage without an explicit progress should snap
    # progress to the stage floor (but never backwards).
    job = store.create(source="upload")
    store.update(job.job_id, stage=STAGE_SCANNING)
    assert store.get(job.job_id).progress == jobs_mod._STAGE_PROGRESS[STAGE_SCANNING]
