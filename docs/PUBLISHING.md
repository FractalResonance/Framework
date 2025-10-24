# FRC Publishing Policy and Workflow

## DOI Policy (Best Practice)

- One concept DOI per paper ID.
  - Example: `FRC 566.001` → its own concept DOI; subsequent revisions are versions.
  - `FRC 566.010`, `566.020`, … each get their own concept DOI.
- Use “versions” only for revisions of the same paper (typo fixes, minor figure additions, etc.).
- Keep the `FRC 100.003` family under a single concept (your chosen 100.003 concept), and publish new 100.003 versions there.
- Papers in different series (e.g., `566.001`, `567.901`) must have independent concept DOIs.
- Connect papers with `related_identifiers` rather than sharing a concept.
  - `references`, `isReferencedBy`, `isSupplementTo`, `isPartOf`/`hasPart` as appropriate.

## What GitHub Releases Are Used For

- Use Releases only to build and attach PDFs via CI.
- Do not rely on Release→Zenodo auto-archiving to mint concepts, because it groups all releases from one repo into one Zenodo concept.

## How To Create a Zenodo Record (Per Paper)

1. Build the PDF locally or via CI.
2. Prepare a paper-specific `zenodo.json` (metadata): title, abstract, creators (Hadi Servat), keywords, license (CC BY 4.0), publication_date, version, related_identifiers.
3. Create a deposition via API (preferred) using `scripts/zenodo_create_deposition.sh` (requires `ZENODO_TOKEN`).
4. Upload the PDF to that deposition, then publish.
5. Add the DOI badge to README and insert the DOI into the paper front matter.

## 100.003 Family

- Publish new `100.003` versions under your existing 100.003 concept.
- You can treat `100.003.566` as a separate methods note (independent concept), or publish it as a version in the 100.003 concept (your choice). This repo defaults to independent concepts unless you specify otherwise.

## Templates

- See `papers/templates/zenodo_metadata.example.json` for a ready-to-edit JSON template.

