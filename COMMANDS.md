# Wikipedia Text Extractor Command Reference

Copy these commands in **Command Prompt** from the project folder.

```cmd
cd /d "C:\Users\Lenovo Legion\Desktop\Wikipedia_text_extractor"
```

This launch command list focuses on English Wikipedia pages and the two primary products:

- Full text extraction with HTML parser and Extracts API
- Partial text extraction with the hybrid heading/citation matcher

Replace the URL and output file name when testing another page.

## Full Text: HTML Parser

Recommended clean output with LaTeX math:

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

Math removed:

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math remove --output output\large_language_model.txt
```

Math kept raw:

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math keep --output output\large_language_model.txt
```

## Full Text: HTML With References

Save normal clean HTML output plus a separate references output. The references output keeps inline citation numbers and places the rebuilt `== References ==` section in the original page order.

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt --save-references
```

Save references at the end only, without inline citation numbers:

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt --save-references --references-end-only
```

## Full Text: Extracts API

Recommended Extracts API output with LaTeX math:

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

Math removed:

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math remove --output output\large_language_model.txt
```

Math kept raw:

```cmd
.venv\Scripts\python.exe extract_with_extracts_api.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math keep --output output\large_language_model.txt
```

## Full Text: Compare HTML and API

Generate all six clean text files, the remove-mode comparison report, and the runtime report:

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --output output\large_language_model.txt
```

Run both methods for only one math mode:

```cmd
.venv\Scripts\python.exe compare_extraction_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --math latex --output output\large_language_model.txt
```

## Partial Text: Hybrid Method

Before running, paste copied Wikipedia text into:

```text
input_text\partial_input.txt
```

Create/open the input file from Command Prompt:

```cmd
if not exist input_text mkdir input_text
notepad input_text\partial_input.txt
```

Default partial clean output:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

Default behavior:

- Uses headings, citation numbers, sentence windows, and staged boundary ranges.
- Reads image captions from the Wikipedia HTML and strips matching copied captions from the pasted input.
- Ignores copied `References`, `See also`, and `External links` sections for matching.
- Removes inline citation numbers from the final output.
- Does not add a References list.

Use a custom copied input file:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --input input_text\partial_input.txt
```

Keep only copied/used references with original Wikipedia numbers:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --references original
```

Keep only copied/used references, sort by original reference number, and renumber from 1:

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --url "https://en.wikipedia.org/wiki/Large_language_model" --references smart
```

## Partial Text: Token Method (Experimental)

This is a faster experimental matcher. It builds an inverted token index from the cleaned article, searches candidate windows around shared rare tokens, then scores them with weighted token overlap and ordered-token coverage. Fuzzy matching is off by default.

Default token partial clean output:

```cmd
.venv\Scripts\python.exe extract_partial_token.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

Use optional fuzzy confirmation only when several token candidates are very close:

```cmd
.venv\Scripts\python.exe extract_partial_token.py --url "https://en.wikipedia.org/wiki/Large_language_model" --confirm fuzzy
```

Tune the token anchor size and minimum score:

```cmd
.venv\Scripts\python.exe extract_partial_token.py --url "https://en.wikipedia.org/wiki/Large_language_model" --window-tokens 60 --min-score 0.72
```

The token method scans copied start/end text in non-overlapping token chunks, using `60` tokens by default.

Token method outputs:

```text
output\Large_language_model\English\partial_token_text.txt
output\Large_language_model\English\partial_token_match_report.txt
```

## Partial Text: Diff Match Patch Method (Experimental)

This method is only for timing comparison. It uses Google's `diff-match-patch` package with short start/end anchor chunks after the same caption/noise cleanup. It does not run a full article-wide DMP diff by default, because that is too slow for large pages.

Install the optional package once if it is not already installed:

```cmd
.venv\Scripts\python.exe -m pip install diff-match-patch
```

Default DMP partial clean output:

```cmd
.venv\Scripts\python.exe extract_partial_dmp.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

Tune DMP coverage, anchor size, and timeout:

```cmd
.venv\Scripts\python.exe extract_partial_dmp.py --url "https://en.wikipedia.org/wiki/Large_language_model" --min-coverage 0.72 --anchor-chars 600 --timeout 1.0
```

DMP method outputs:

```text
output\Large_language_model\English\partial_dmp_text.txt
output\Large_language_model\English\partial_dmp_match_report.txt
```

## Partial Method Runtime Benchmark CSV

Run hybrid, token, and DMP consecutively against the same pasted input. The three clean text files are overwritten fresh on each run, while three timing rows are appended to the CSV file. Fetch and clean are shared once; each CSV row records the method-specific match runtime and estimated full runtime.

```cmd
.venv\Scripts\python.exe benchmark_partial_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model"
```

CSV output:

```text
output\Large_language_model\English\large_language_model_partial_benchmark.csv
```

Clean text outputs overwritten on every run:

```text
output\Large_language_model\English\partial_benchmark_hybrid_text.txt
output\Large_language_model\English\partial_benchmark_token_text.txt
output\Large_language_model\English\partial_benchmark_dmp_text.txt
```

If one method fails, its text file is still overwritten with a readable error report instead of being left missing.

Run it again after changing `input_text\partial_input.txt`; the next benchmark rows will be appended below the old rows.

Tune benchmark settings:

```cmd
.venv\Scripts\python.exe benchmark_partial_methods.py --url "https://en.wikipedia.org/wiki/Large_language_model" --token-window 60 --token-min-score 0.72 --dmp-min-coverage 0.72 --dmp-anchor-chars 600
```

## Debug Reports To Keep

The partial hybrid match report is important for debugging and should be kept:

```text
output\Large_language_model\English\partial_hybrid_match_report.txt
```

Open it from Command Prompt:

```cmd
type output\Large_language_model\English\partial_hybrid_match_report.txt
```

It records matched headings, copied citations, stripped caption count, start/end search ranges, scores, offsets, and reference mode.

Runtime report:

```text
output\Large_language_model\English\large_language_model_runtime.txt
```

Open it:

```cmd
type output\Large_language_model\English\large_language_model_runtime.txt
```

## Output Folder Pattern

Outputs are saved under:

```text
output\<Topic>\<Language>\
```

Example:

```text
output\Large_language_model\English\
```

Common primary files:

```text
large_language_model_html_latex.txt
large_language_model_html_remove.txt
large_language_model_html_keep.txt
large_language_model_html_references.txt
large_language_model_extracts_latex.txt
large_language_model_extracts_remove.txt
large_language_model_extracts_keep.txt
large_language_model_comparison.txt
large_language_model_runtime.txt
partial_hybrid_text.txt
partial_hybrid_match_report.txt
partial_token_text.txt
partial_token_match_report.txt
partial_dmp_text.txt
partial_dmp_match_report.txt
```

## Title Instead of URL

Use `--title` with `--lang` if you do not want to pass a URL.

```cmd
.venv\Scripts\python.exe extract_with_html_parser.py --title "Large language model" --lang en --math latex --output output\large_language_model.txt
```

```cmd
.venv\Scripts\python.exe extract_partial_hybrid.py --title "Large language model" --lang en
```

```cmd
.venv\Scripts\python.exe extract_partial_token.py --title "Large language model" --lang en
```

```cmd
.venv\Scripts\python.exe extract_partial_dmp.py --title "Large language model" --lang en
```

## Run Tests

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests
```
