"""Run the Python half of every R/Python pair in the articles and write what it printed under it.

An article shows a block in both languages by wrapping the R chunk and its Python twin in a
``.lang-pair`` div:

    ::: {.lang-pair}
    ```{r}
    grain_matrix(series, plot, t, temp, grain = "month")
    ```

    ```python
    ts.grain_matrix(series, "plot", "t", "temp", grain="month")
    ```
    :::

The R chunk is knitted when the site is built. The Python block is run here, every block of an
article in one namespace in the order it stands, after ``resolve_core.manual_seed(42)`` as the R
chunks run after ``resolve.manual_seed(42)`` in the setup chunk, and its output is written back
under each statement, the way a knitted R chunk shows it. A block's info string takes ``error=true`` (the block
must raise, and the error is its output) and ``fig-alt="..."``. A figure a block leaves open is
saved as SVG beside the article and drawn under the block; the ``<img>`` line is this tool's.

A number in the prose that comes from the Python run is a span beside the R number:

    The weekly candidate scores [0.838]{.lang-r-only}[0.800]{.lang-py-only py="f'{mean_auc:.3f}'"}.

The expression in ``py`` is evaluated in the article's namespace at the span's place in the text,
after the blocks above it have run, and its value is written back as the span's content. An
expression uses single quotes inside, because the attribute is double-quoted.

The article records the digest of the Python it was run from, so ``--check`` says whether every
article is current without running anything and without the compiled core:

    python tools/python_twins.py            run every article that holds a pair
    python tools/python_twins.py NAME       run vignettes/articles/NAME.Rmd alone
    python tools/python_twins.py --check    exit 1 if an article is older than its Python
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import html
import io
import os
import re
import shlex
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROMPT = "#> "
ARTICLES = ROOT / "vignettes" / "articles"
BANNER = re.compile(r"^<!-- Python blocks run by tools/python_twins.py \(md5 ([0-9a-f]{32})\)\. -->$")
BANNER_TEXT = "<!-- Python blocks run by tools/python_twins.py (md5 {digest}). -->"
FIGURE = re.compile(r'^<img src="[^"]*py-figure-\d+\.svg" alt="[^"]*" />$')
FIGURE_NAME = "py-figure-"
SPAN = re.compile(r'\[([^\[\]\n]*)\]\{\.lang-py-only py="([^"]*)"\}')


def digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def run_statement(stmt: ast.stmt, shown_value: bool, namespace: dict) -> None:
    """Run one top-level statement, printing the value of an expression the way a console does."""
    if isinstance(stmt, ast.Expr):
        value = eval(compile(ast.Expression(stmt.value), "<article>", "eval"), namespace)
        if shown_value and value is not None:
            print(value if isinstance(value, str) else repr(value))
    else:
        exec(compile(ast.Module([stmt], type_ignores=[]), "<article>", "exec"), namespace)


def run_block(source: str, namespace: dict, expect_error: bool) -> list[str]:
    """Run one block statement by statement and return its lines with each statement's output
    under it, as a knitted chunk shows them.

    An expression's value is shown unless its line ends in a semicolon. Lines between two
    statements, comments and blank lines, travel with the statement after them.
    """
    lines = source.rstrip("\n").splitlines()
    out, start, raised = [], 0, False
    for stmt in ast.parse(source).body:
        end = stmt.end_lineno
        out += lines[start:end]
        shown = io.StringIO()
        with contextlib.redirect_stdout(shown):
            try:
                run_statement(stmt, not lines[end - 1].rstrip().endswith(";"), namespace)
            except Exception as error:  # noqa: BLE001 -- the error is the output being shown
                if not expect_error:
                    sys.stdout = sys.__stdout__
                    traceback.print_exc()
                    raise SystemExit(f"a block raised:\n{source}")
                print(f"{type(error).__name__}: {error}")
                raised = True
        out += [(PROMPT + line).rstrip() for line in shown.getvalue().rstrip("\n").splitlines()]
        start = end
        if raised:
            break
    if expect_error and not raised:
        raise SystemExit(f"a block marked error=true ran without raising:\n{source}")
    return out + lines[start:]


def take_figures(name: str, path: str, count: int) -> dict[str, str]:
    """Every figure left open, as SVG text keyed by its path beside the article, closed after.

    `path` is the figure's path with `{n}` standing for its number, counted on from `count`.
    """
    if "matplotlib.pyplot" not in sys.modules:
        return {}
    import matplotlib.pyplot as plt
    # The SVG's element ids are drawn from a salt, a fresh random one unless it is set, and a
    # figure that changes on every build would change the site's committed build with it.
    plt.rcParams["svg.hashsalt"] = name
    out = {}
    for number in plt.get_fignums():
        count += 1
        figure = plt.figure(number)
        recolour(figure)
        drawn = io.StringIO()
        figure.savefig(drawn, format="svg", metadata={"Date": None}, transparent=True)
        out[path.format(n=count)] = drawn.getvalue().replace(INK, "currentColor")
    plt.close("all")
    return out


# The text, spines and ticks of a figure are drawn in this colour and written out as
# `currentColor`, so a figure reads in the page's own text colour under the light and the dark
# theme alike, as the site's R figures do. No figure draws in it on purpose.
INK = "#010203"


def recolour(figure) -> None:
    """Put a figure's text, spines, ticks and legend frame in the page's colour, not black."""
    for text in figure.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "set_color")):
        if text.get_color() in ("black", "#000000", "k", (0.0, 0.0, 0.0, 1.0)):
            text.set_color(INK)
    for ax in figure.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(INK)
        ax.tick_params(colors=INK, which="both")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_alpha(0)


def check_install() -> None:
    """Stop unless the resolve_core being imported is the version this tree carries.

    The articles show what the engine prints, so they are run against the engine as it stands
    here, never against an older build: the compiled module's version must be the one
    `DESCRIPTION` carries.
    """
    try:
        import resolve_core
    except ImportError as error:
        raise SystemExit(f"{sys.executable} cannot import resolve_core ({error}). Build this tree "
                         "or name an interpreter that has it in PYTHON.")
    wanted = next(line.split(":", 1)[1].strip()
                  for line in (ROOT / "DESCRIPTION").read_text(encoding="utf-8").splitlines()
                  if line.startswith("Version:"))
    if resolve_core.__version__ != wanted:
        raise SystemExit(f"the resolve_core {sys.executable} imports is "
                         f"{resolve_core.__version__} at {resolve_core.__file__}, and this tree "
                         f"is {wanted}. Build the engine first.")


def sources() -> dict[str, Path]:
    """Every article that holds a pair, by name: the vignette and the articles written by hand."""
    paths = [ROOT / "vignettes" / "resolveR.Rmd", *sorted(ARTICLES.glob("*.Rmd"))]
    found = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if ".lang-pair" in text:
            found[path.stem] = path
    return found


class Block:
    """One Python fence of a pair: where it stands, its options, its source without output."""

    def __init__(self, first: int, last: int, options: dict, source: list[str]):
        self.first, self.last = first, last   # the fence's lines, [first, last] inclusive
        self.options, self.source = options, source
        self.figures_end = last                # past the figure lines under the fence


def blocks(lines: list[str]) -> list[Block]:
    """The Python fences inside `.lang-pair` divs, in order."""
    found, inside, i = [], False, 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.startswith(":::"):
            inside = ".lang-pair" in line if not inside else False
        elif inside and line.startswith("```"):
            words = shlex.split(line[3:].strip().strip("{}"))
            end = next((j for j in range(i + 1, len(lines)) if lines[j].rstrip("\n") == "```"), None)
            if end is None:
                raise SystemExit(f"the fence opening on line {i + 1} is not closed")
            if words and words[0] in ("python", ".python"):
                body = [l.rstrip("\n") for l in lines[i + 1:end] if not l.startswith("#>")]
                options = dict(w.split("=", 1) for w in words[1:])
                found.append(Block(i, end, options, body))
            i = end
        i += 1
    for block in found:
        j = block.last + 1
        while j < len(lines) and (not lines[j].strip() or FIGURE.match(lines[j].rstrip("\n"))):
            j += 1
        # Blank lines between the fence and what follows are the author's; only figure lines go.
        block.figures_end = j
    return found


def split_front_matter(lines: list[str]) -> int:
    """The index of the line closing the YAML header."""
    if not lines or lines[0].strip() != "---":
        raise SystemExit("an article opens with a YAML header between --- lines")
    return next(i for i in range(1, len(lines)) if lines[i].strip() == "---")


def article_digest(found: list[Block], lines: list[str]) -> str:
    spans = [" ".join(m.group(2).split()) for m in SPAN.finditer("".join(lines))]
    return digest("\n\x00\n".join("\n".join(b.source) + repr(sorted(b.options.items()))
                                  for b in found) + "\n\x01\n" + "\n".join(spans))


def fill_spans(segment: list[str], namespace: dict) -> list[str]:
    """The prose lines with each Python span's content replaced by its expression's value."""
    def value(match: re.Match) -> str:
        expression = " ".join(match.group(2).split())
        try:
            shown = str(eval(expression, namespace))
        except Exception as error:  # noqa: BLE001 -- named with its expression below
            raise SystemExit(f'the span py="{expression}" raised {type(error).__name__}: {error}')
        return f'[{shown}]{{.lang-py-only py="{match.group(2)}"}}'
    return SPAN.sub(value, "".join(segment)).splitlines(keepends=True)


def recorded(lines: list[str]) -> str | None:
    for line in lines[:split_front_matter(lines) + 4]:
        match = BANNER.match(line.rstrip("\n"))
        if match:
            return match.group(1)
    return None


def with_resources(header: list[str], paths: list[str]) -> list[str]:
    """The YAML header with `resource_files` listing the figures drawn and the entries it had."""
    out, kept, in_list = [], [], False
    for line in header[:-1]:
        if line.startswith("resource_files:"):
            in_list = True
        elif in_list and line.startswith("  - "):
            if FIGURE_NAME not in line:
                kept.append(line)
        else:
            in_list = False
            out.append(line)
    entries = kept + [f"  - {path}\n" for path in paths]
    if entries:
        out += ["resource_files:\n", *entries]
    return out + [header[-1]]


def render(name: str, path: Path) -> tuple[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = blocks(lines)
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace: dict = {"__name__": "__article__"}
    import resolve_core
    resolve_core.manual_seed(42)
    figures: dict[str, str] = {}
    out: list[str] = []
    cursor = 0
    for block in found:
        out += fill_spans(lines[cursor:block.first], namespace)
        # Pandoc reads options on a fence only inside braces, so a fence that has any is written
        # as {.python key=value}.
        options = " ".join(f'{k}="{v}"' if " " in v else f"{k}={v}" for k, v in block.options.items())
        info = "```{.python " + options + "}" if options else "```python"
        result = run_block("\n".join(block.source) + "\n", namespace,
                           block.options.get("error") == "true")
        out += [info + "\n", *[line + "\n" for line in result], "```\n"]
        drawn = take_figures(name, f"{name}_files/{FIGURE_NAME}{{n}}.svg", len(figures))
        figures.update(drawn)
        if drawn:
            alt = html.escape(block.options.get("fig-alt", ""), quote=True)
            out += ["\n"] + [f'<img src="{p}" alt="{alt}" />\n' for p in drawn]
        region = lines[block.last + 1:block.figures_end]
        trailing = 0
        while trailing < len(region) and not region[-1 - trailing].strip():
            trailing += 1
        out += ["\n"] * trailing
        cursor = block.figures_end
    out += fill_spans(lines[cursor:], namespace)

    close = split_front_matter(out)
    banner = BANNER_TEXT.format(digest=article_digest(found, lines)) + "\n"
    body = [l for l in out[close + 1:] if not BANNER.match(l.rstrip("\n"))]
    while body and not body[0].strip():
        body.pop(0)
    header = with_resources(out[:close + 1], list(figures))
    return "".join(header) + "\n" + banner + "\n" + "".join(body), figures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    available = sources()
    names = args.names or list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit("no pair in: " + ", ".join(unknown))

    if args.check:
        stale = []
        for name in names:
            lines = available[name].read_text(encoding="utf-8").splitlines(keepends=True)
            if recorded(lines) != article_digest(blocks(lines), lines):
                stale.append(name)
        if stale:
            print("Older than its Python, rerun tools/python_twins.py: " + ", ".join(stale),
                  file=sys.stderr)
            return 1
        print("The articles' Python blocks match their output.")
        return 0

    check_install()
    for name in names:
        path = available[name]
        article, figures = render(name, path)
        folder = path.parent
        for old in (folder / f"{name}_files").glob(f"{FIGURE_NAME}*.svg"):
            if f"{name}_files/{old.name}" not in figures:
                old.unlink()
        for rel, svg in figures.items():
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text(svg, encoding="utf-8", newline="\n")
        path.write_text(article, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
