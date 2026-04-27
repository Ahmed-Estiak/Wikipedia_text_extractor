# Wikipedia Text Extractor

Clean plain-text extractor for Wikipedia pages.

## Usage

Create and activate the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Extract by page title:

```powershell
python .\wiki_text_extractor.py --title "Bangladesh" --lang en --output output\bangladesh.txt
```

Extract by URL:

```powershell
python .\wiki_text_extractor.py --url "https://en.wikipedia.org/wiki/Bangladesh" --output output\bangladesh.txt
```

Run the extracts API method as its own Python file:

```powershell
python .\extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Saturn" --output output\saturn.txt
```

Run the HTML parser method as its own Python file:

```powershell
python .\extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Saturn" --output output\saturn.txt
```

Choose how math equations are handled:

```powershell
python .\extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Kuiper_belt" --math remove --output output\kuiper_belt.txt
python .\extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Kuiper_belt" --math latex --output output\kuiper_belt.txt
python .\extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Kuiper_belt" --math keep --output output\kuiper_belt.txt
```

Run both methods, save both outputs, compare mismatches, and save runtime:

```powershell
python .\compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Saturn" --output output\saturn.txt
```

Save raw Wikipedia API responses for debugging:

```powershell
python .\compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Kuiper_belt" --output output\kuiper_belt.txt --save-raw
```

This writes:

```text
output\Saturn\English\saturn_extracts_remove.txt
output\Saturn\English\saturn_html_remove.txt
output\Saturn\English\saturn_extracts_latex.txt
output\Saturn\English\saturn_html_latex.txt
output\Saturn\English\saturn_extracts_keep.txt
output\Saturn\English\saturn_html_keep.txt
output\Saturn\English\saturn_comparison.txt
output\Saturn\English\saturn_runtime.txt
```

With `--save-raw`, this also writes:

```text
output\Saturn\English\saturn_raw_extracts.txt
output\Saturn\English\saturn_raw_html.txt
```

Outputs are grouped as `output\<Topic>\<Language>\...`, for example
`output\Kuiper_belt\English\...`, `output\বাংলাদেশ\Bangla\...`, or
`output\Kuiperin_vyöhyke\Suomi\...`.

Write output to a file:

```powershell
python .\wiki_text_extractor.py --title "Bangladesh" --output bangladesh.txt
```

## What It Cleans

- Infoboxes, tables, navigation boxes, sidebars, and table of contents
- Reference markers and reference lists
- Edit links and metadata blocks
- References, external links, further reading, notes, and see also tail sections
- Math equations can be removed, kept as LaTeX, or kept raw
- Extra whitespace and noisy blank lines

## Scripts

- `extract_with_extracts_api.py`: uses Wikipedia `prop=extracts&explaintext=1`
- `extract_with_html_parser.py`: uses Wikipedia parsed HTML, then removes noisy HTML sections
- `compare_extraction_methods.py`: runs both methods, saves both `.txt` files, writes mismatch diff and runtime report
- `wiki_text_extractor.py`: shared core logic and backwards-compatible CLI

## Tests

```powershell
python -m unittest discover -s tests
```
