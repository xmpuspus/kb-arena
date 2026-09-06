# What to do when a released surface is wrong

A KB Arena release writes to five surfaces. Each one rolls back a different
way, and two of them do not roll back at all. Read the surface you need before
you touch it.

The named version below is 0.11.0. Replace it with the version you must undo.

## PyPI does not let you replace a version

PyPI refuses a re-upload of a filename it already holds. You cannot fix 0.11.0
in place.

These are the two options, best first.

1. Ship 0.11.1. Fix the defect, bump the version, and publish again. Readers who
   pinned 0.11.0 keep a working install, and `pip install kb-arena` gives them
   the fix.
2. Yank 0.11.0. Use this only when the release is unsafe to install, for example
   when it leaks a credential or corrupts data. Run `python3 -m twine yank` or
   use the PyPI web form. A yank hides the version from resolution, but a pin to
   `kb-arena==0.11.0` still installs it. A yank is not a delete.

Never delete a PyPI release to re-upload the same number. The number is then
permanently unusable.

## The git tag moves only before anybody fetches it

If nobody fetched `v0.11.0` and no artifact points at it, delete and recreate
it with these two commands.

```
gh api -X DELETE repos/xmpuspus/kb-arena/git/refs/tags/v0.11.0
git tag -d v0.11.0
```

After PyPI holds a release built from that tag, the tag is a record and not a
pointer. Leave it. Cut a new tag instead.

## A GitHub release is editable, and its assets are replaceable

The release body, the title and the assets all change after publication.

```
gh release edit v0.11.0 --notes-file <path>
gh release delete-asset v0.11.0 <asset-name>
gh release upload v0.11.0 <path>
```

Deleting the release itself removes the Zenodo trigger for that version but not
the Zenodo record it already made.

## Zenodo keeps every version it archived

A Zenodo record is a citable archive, so Zenodo does not delete one on request
from the repository. The concept DOI `10.5281/zenodo.20319678` always resolves
to the newest version, so the fix is to publish a newer version and let the
concept DOI move.

To correct the metadata of one version, edit that record in the Zenodo web
interface. To retract one, open a support request with Zenodo and expect a
tombstone rather than a delete.

## Repository metadata rolls back in one command

The description, the homepage and the topics are free to change.

```
gh repo edit --description "<previous text>" --homepage "<previous url>"
gh repo edit --add-topic <name> --remove-topic <name>
```

The 20-topic cap means an add can fail while the description edit already
succeeded. Read the result back with `gh repo view --json description,homepageUrl,repositoryTopics`.

## Before you roll anything back, read the surface

Every surface here has a read command. Run it first, so you know the current
state rather than the state you expect.

```
curl -s https://pypi.org/pypi/kb-arena/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
gh release view v0.11.0 --json tagName,isDraft,assets
gh repo view --json description,homepageUrl,repositoryTopics
curl -s "https://zenodo.org/api/records?q=conceptdoi:%2210.5281/zenodo.20319678%22&sort=mostrecent&size=1"
```
