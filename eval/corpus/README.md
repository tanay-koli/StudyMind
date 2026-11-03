# Eval corpus

*Open Data Structures* (Python edition) by Pat Morin, released under a Creative Commons Attribution
licence (see the book's "Why This Book?" section): <https://opendatastructures.org/ods-python.pdf>.

`python -m eval.ingest` downloads the book here and splits out three chapters (PDF page ranges of the full book):

| File | Chapter | Pages |
|---|---|---|
| `ods-03-linked-lists.pdf` | 3 Linked Lists | 73–94 |
| `ods-04-skiplists.pdf` | 4 Skiplists | 95–112 |
| `ods-06-binary-trees.pdf` | 6 Binary Trees | 139–156 |

Page numbers in `eval/dataset.jsonl` gold anchors are pages of these chapter files (1-based), not of the book.
The PDFs are git-ignored; this file and `eval/ingest.py` are the record of exactly what was used.
