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

Choose an extraction method:

```powershell
python .\wiki_text_extractor.py --url "https://en.wikipedia.org/wiki/Saturn" --method extracts --output output\saturn.txt
python .\wiki_text_extractor.py --url "https://en.wikipedia.org/wiki/Saturn" --method html --output output\saturn.txt
```

Save both extraction methods for comparison:

```powershell
python .\wiki_text_extractor.py --url "https://en.wikipedia.org/wiki/Saturn" --method both --output output\saturn.txt
```

This writes:

```text
output\saturn_extracts.txt
output\saturn_html.txt
```

Write output to a file:

```powershell
python .\wiki_text_extractor.py --title "Bangladesh" --output bangladesh.txt
```

## What It Cleans

- Infoboxes, tables, navigation boxes, sidebars, and table of contents
- Reference markers and reference lists
- Edit links and metadata blocks
- References, external links, further reading, notes, and see also tail sections
- Extra whitespace and noisy blank lines

## Tests

```powershell
python -m unittest discover -s tests
```
