"""What an adapter has to record, and what it must refuse.

No test here downloads anything. A dataset large enough to be worth adapting is
too large to commit, so the fixtures are small files written in the test, and
what is checked is the contract rather than the data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kb_arena.adapters import ADAPTERS, ChecksumMismatchError, LicenseRefusalError, sha256_of
from kb_arena.adapters.base import DatasetAdapter
from kb_arena.models.adapter import NON_REDISTRIBUTABLE, DatasetManifest


def _complete(**overrides) -> dict:
    base = {
        "name": "demo",
        "attribution": "Somebody, cited as they ask",
        "source_url": "https://example.org/demo",
        "revision": "v1.2.0",
        "license": "CC-BY-4.0",
        "checksum_sha256": "a" * 64,
        "preprocessing_version": "1",
        "cache_path": "/tmp/demo",
        "documents": 3,
        "questions": 9,
    }
    return {**base, **overrides}


def test_a_manifest_records_what_a_reader_needs_to_check_the_corpus():
    manifest = DatasetManifest(**_complete())

    assert manifest.attribution
    assert manifest.source_url
    assert manifest.revision
    assert manifest.license
    assert manifest.checksum_sha256
    assert manifest.preprocessing_version
    assert manifest.downloaded_at, "a corpus without a date cannot be placed in time"


@pytest.mark.parametrize("revision", ["latest", "HEAD", "main", "master", " "])
def test_a_moving_revision_is_refused(revision):
    """A run against `latest` cannot be repeated, so it cannot be cited."""
    with pytest.raises(ValidationError, match="moving target"):
        DatasetManifest(**_complete(revision=revision))


@pytest.mark.parametrize("checksum", ["", "abc", "A" * 64, "z" * 64, "a" * 63])
def test_a_checksum_that_is_not_a_digest_is_refused(checksum):
    with pytest.raises(ValidationError):
        DatasetManifest(**_complete(checksum_sha256=checksum))


def test_a_manifest_that_counts_data_must_say_where_it_is():
    """A corpus nobody can locate is a corpus nobody can check."""
    with pytest.raises(ValidationError, match="must say where they"):
        DatasetManifest(**_complete(cache_path=""))

    # A template counts nothing, so it needs no path yet.
    template = DatasetManifest(**_complete(cache_path="", documents=0, questions=0))
    assert template.cache_path == ""


@pytest.mark.parametrize("license_id", sorted(NON_REDISTRIBUTABLE))
def test_a_non_commercial_licence_is_not_redistributable(license_id):
    assert DatasetManifest(**_complete(license=license_id)).redistributable is False


@pytest.mark.parametrize("license_id", ["CC-BY-4.0", "Apache-2.0", "MIT", "ODC-BY-1.0"])
def test_an_open_licence_is_redistributable(license_id):
    assert DatasetManifest(**_complete(license=license_id)).redistributable is True


def test_the_digest_reads_a_file_in_chunks(tmp_path):
    """A dataset worth adapting is too large to hold in memory."""
    path = tmp_path / "corpus.jsonl"
    payload = b'{"id": "one"}\n' * 5000
    path.write_bytes(payload)

    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_data_that_does_not_match_its_manifest_is_refused(tmp_path):
    class _Demo(DatasetAdapter):
        name = "demo"

        def manifest_template(self):  # pragma: no cover - not used here
            raise NotImplementedError

        def build(self, destination):  # pragma: no cover - not used here
            raise NotImplementedError

    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "one"}\n')

    adapter = _Demo()
    adapter.verify(path, sha256_of(path))

    with pytest.raises(ChecksumMismatchError, match="cannot be cited"):
        adapter.verify(path, "b" * 64)


class _DownloadOnly(DatasetAdapter):
    download_only = True
    name = "restricted"

    def manifest_template(self):  # pragma: no cover - not used here
        raise NotImplementedError

    def build(self, destination):  # pragma: no cover - not used here
        raise NotImplementedError


def test_download_only_data_never_lands_inside_the_repository(tmp_path):
    """The licence difference is where the file lands, so that is what is checked."""
    repo = tmp_path / "repo"
    (repo / "datasets").mkdir(parents=True)
    adapter = _DownloadOnly()

    with pytest.raises(LicenseRefusalError, match="inside the repository"):
        adapter.check_destination(repo / "datasets" / "restricted", repo)
    with pytest.raises(LicenseRefusalError):
        adapter.check_destination(repo, repo)

    # Outside the checkout is the user's own storage, which the licence allows.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert adapter.check_destination(outside, repo) is None


def test_a_redistributable_adapter_may_write_anywhere(tmp_path):
    class _Open(DatasetAdapter):
        name = "open"

        def manifest_template(self):  # pragma: no cover - not used here
            raise NotImplementedError

        def build(self, destination):  # pragma: no cover - not used here
            raise NotImplementedError

    repo = tmp_path / "repo"
    repo.mkdir()
    assert _Open().check_destination(repo / "datasets" / "open", repo) is None


def test_crag_is_download_only_and_says_the_terms():
    """CC BY-NC 4.0 lets a user fetch it and does not let this project ship it."""
    adapter = ADAPTERS["crag"]()
    template = adapter.manifest_template()

    assert adapter.download_only is True
    assert template.license == "CC-BY-NC-4.0"
    assert template.redistributable is False
    assert "creativecommons.org" in template.license_url
    assert template.attribution, "the dataset card asks for a citation"


def test_crag_refuses_to_fetch_and_says_what_to_do_instead():
    """Fetching it is the user's act under the licence they accepted, not ours."""
    with pytest.raises(NotImplementedError) as refused:
        ADAPTERS["crag"]().build(Path("/tmp/crag"))

    message = str(refused.value)
    assert "download-only" in message
    assert "CC BY-NC 4.0" in message
    assert "github.com/facebookresearch/CRAG" in message


def test_no_adapter_data_is_committed():
    """A dataset large enough to be worth adapting is too large to commit."""
    root = Path(__file__).resolve().parents[2]
    for adapter_name in ADAPTERS:
        bundled = root / "datasets" / adapter_name
        assert not bundled.exists(), (
            f"datasets/{adapter_name} is committed. An adapter fetches its data; it "
            "does not ship it."
        )


def test_every_adapter_emits_a_manifest_the_model_accepts():
    """The template is the promise. A broken one fails here, not at download time."""
    for name, cls in ADAPTERS.items():
        template = cls().manifest_template()
        assert isinstance(template, DatasetManifest)
        assert template.name == name
        # A template round-trips, so it can be written beside the data.
        assert json.loads(template.model_dump_json())["name"] == name


def test_the_dataset_doc_states_the_three_rules_the_code_enforces():
    """A doc that describes a protection the code does not have is worse than none."""
    root = Path(__file__).resolve().parents[2]
    doc = " ".join((root / "docs" / "datasets.md").read_text().split())

    assert "A moving revision is refused" in doc
    assert "A checksum mismatch stops the run" in doc
    assert "A non-commercial licence is never vendored" in doc
    # Each rule has a test above it in this file, so the doc is not the only
    # place the promise lives.
    assert "CC BY-NC 4.0" in doc
    assert "enforced by where the file lands, not by a comment" in doc
