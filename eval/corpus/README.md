# Eval corpus

*Open Data Structures* (Python edition) by Pat Morin, released under a Creative Commons Attribution
licence (see the book's "Why This Book?" section): <https://opendatastructures.org/ods-python.pdf>.

The three chapter PDFs below are excerpts of that book, redistributed unchanged under its licence, with
credit to Pat Morin. They are committed so CI and fresh clones never depend on the website; `python -m eval.ingest`
downloads the full book only if one of them is missing (page ranges of the full book):

| File | Chapter | Pages |
|---|---|---|
| `ods-03-linked-lists.pdf` | 3 Linked Lists | 73–94 |
| `ods-04-skiplists.pdf` | 4 Skiplists | 95–112 |
| `ods-06-binary-trees.pdf` | 6 Binary Trees | 139–156 |

Page numbers in `eval/dataset.jsonl` gold anchors are pages of these chapter files (1-based), not of the book.
The full book (`ods-python.pdf`) is git-ignored; this file and `eval/ingest.py` record exactly what was used.
