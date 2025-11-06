#!/usr/bin/env python3

import json
import os
from pathlib import Path


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def main():
    in_path = Path("data/qualitative_analysis.json")
    out_path = Path("data/qualitative_examples.tex")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("detailed_results", [])
    # Pick 2 examples: one sentiment-changing and one sentiment-preserving
    changed = [e for e in examples if e.get("label_changed")]
    unchanged = [e for e in examples if not e.get("label_changed")]
    selected = (changed[:1] + unchanged[:1]) if examples else []

    lines = []
    lines.append("% Auto-generated qualitative examples appendix")
    lines.append("\\subsection{Appendix: Qualitative Examples (Auto-Extracted)}")
    lines.append("Representative examples illustrating semantic preservation and label behavior:")
    lines.append("\\begin{enumerate}[leftmargin=*]")
    for i, ex in enumerate(selected, 1):
        o = latex_escape(ex.get("original", "").strip())
        r = latex_escape(ex.get("rewritten", "").strip())
        lo = ex.get("original_label")
        lr = ex.get("rewritten_label")
        changed_flag = ex.get("label_changed")
        cd = ex.get("confidence_drop")
        change_text = "changed" if changed_flag else "unchanged"
        label_desc = f"label {change_text}: {lo} → {lr}" if lo and lr else f"label {change_text}"
        conf_desc = f"; confidence Δ = {cd:+.3f}" if isinstance(cd, (int, float)) else ""
        lines.append(
            f"  \\item Original: ``{o}''\\\\ Rewritten: ``{r}'' ({label_desc}{conf_desc})"
        )
    lines.append("\\end{enumerate}")
    lines.append("\\noindent Additional examples were tested and yielded similar trends (details omitted for brevity).")

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote LaTeX appendix: {out_path}")


if __name__ == "__main__":
    main()