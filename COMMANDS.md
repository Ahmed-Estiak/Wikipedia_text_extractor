# Wikipedia Text Extractor Command Reference

Copy these commands in **Command Prompt** from the project folder:

```cmd
cd /d "C:\Users\Lenovo Legion\Desktop\Wikipedia_text_extractor"
```

Replace the URL and output file name when testing another page.

## Full Extraction: Extracts API

Creates one cleaned `.txt` file using the Wikipedia extracts API.

### Math Removed

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math remove --output output\large_language_model.txt
```

### Math as LaTeX

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

### Math Kept Raw

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math keep --output output\large_language_model.txt
```

## Full Extraction: HTML Parser

Creates one cleaned `.txt` file using parsed Wikipedia HTML.

### Math Removed

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math remove --output output\large_language_model.txt
```

### Math as LaTeX

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

### Math Kept Raw

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math keep --output output\large_language_model.txt
```

## HTML With References

Creates the normal HTML parser output plus a separate references file.

### Keep Inline Citation Numbers and References at End

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt --save-references
```

### Keep References Only at End, No Inline Citation Numbers

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt --save-references --references-end-only
```

## Run Both Methods

Runs extracts API and HTML parser together.

### Generate All Six Clean Text Files

Creates:

- extracts API: `remove`, `latex`, `keep`
- HTML parser: `remove`, `latex`, `keep`
- remove-mode comparison report
- runtime report

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --output output\large_language_model.txt
```

### Run Both Methods for Only One Math Mode

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

## Raw Debug Files

Creates raw API/debug files beside the clean outputs.

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --output output\large_language_model.txt --save-raw
```

## Raw Debug Plus References

Creates clean outputs, raw files, references file, comparison, and runtime.

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --output output\large_language_model.txt --save-raw --save-references
```

## References End Only in Combined Run

Creates references file with numbered sources at the end, but without inline citation markers.

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --output output\large_language_model.txt --save-references --references-end-only
```

## Partial Extraction: Original Fuzzy Method

Before running, paste copied Wikipedia text into:

```text
input_text\partial_input.txt
```

Then run:

```cmd
.venv\Scripts\python.exe extract_partial_html.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

Optional custom input file:

```cmd
.venv\Scripts\python.exe extract_partial_html.py --url "https://en.wikipedia.org/wiki/Large_language_model" --input input_text\partial_input.txt
```

## Partial Extraction: Hybrid Heading/Citation Method

This is the newer partial extraction method. It uses headings, citation numbers, and sentence-window matching.

Before running, paste copied Wikipedia text into:

```text
input_text\partial_input.txt
```

Then run:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

Default output removes inline citation numbers and does not add a References section.

Optional custom input file:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --input input_text\partial_input.txt
```

Keep only copied/used references with original Wikipedia numbers:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --references original
```

Keep only copied/used references, sort them by original number, and renumber from 1:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --references smart
```

## Different Languages

Use a direct Wikipedia URL for the language.

### Bangla Wikipedia

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://bn.wikipedia.org/wiki/বাংলাদেশ" --output output\bangladesh.txt
```

### Finnish Wikipedia

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://fi.wikipedia.org/wiki/Suomi" --output output\suomi.txt
```

## Title Instead of URL

Use `--title` with `--lang`.

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --title "Large language model" --lang en --math latex --output output\large_language_model.txt
```

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --title "Large language model" --lang en --output output\large_language_model.txt
```

## Output Folder Pattern

Outputs are saved under:

```text
output\<Topic>\<Language>\
```

Example for Large language model:

```text
output\Large_language_model\English\
```

Common generated files:

```text
large_language_model_extracts_remove.txt
large_language_model_extracts_latex.txt
large_language_model_extracts_keep.txt
large_language_model_html_remove.txt
large_language_model_html_latex.txt
large_language_model_html_keep.txt
large_language_model_comparison.txt
large_language_model_runtime.txt
large_language_model_raw_extracts.txt
large_language_model_raw_html.txt
large_language_model_html_references.txt
partial_text.txt
partial_match_report.txt
partial_hybrid_text.txt
partial_hybrid_match_report.txt
```

## Run Tests

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests
```
