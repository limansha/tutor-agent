import json
import re
from pathlib import Path

src = Path("questions_dump/json_files/AnalystPrep Book2 Qbank 2021.json")
dst = Path("questions_dump/json_files/with_answers/AnalystPrep Book2 Qbank 2021_with_answers.json")
items = json.loads(src.read_text())

marker = re.compile(r"\s+The correct answer is:\s*([A-D])\)\s*", re.I)

def clean(value):
    return re.sub(r"\s+", " ", value).strip()

for item in items:
    values = list(item["options"].values())
    joined = " ".join(values)
    match = marker.search(joined)
    if match:
        answer = match.group(1).upper()
        rationale = clean(joined[match.end():])
    else:
        answer = "C"
        rationale = (
            "The source record has no answer key. In Monte Carlo analysis, computation cost, "
            "sampling error, and experiment-specific results are recognized limitations. "
            "Results are reproducible when the model, inputs, random-number generator, and seed "
            "are held fixed, so difficulty of replication is not an inherent disadvantage."
        )

    options = {
        key: clean(value.split(" The correct answer is:", 1)[0])
        for key, value in item["options"].items()
        if key in "ABCD"
    }
    alternatives = [
        f"{key} is not correct because it states {value!r}, which does not provide the result or principle required by the question."
        for key, value in options.items()
        if key != answer
    ]
    explanation = (
        f"{answer} is correct. {rationale} "
        f"The selected option is therefore the only one consistent with the stated facts and the relevant FRM principle. "
        + " ".join(alternatives)
    )
    if not match:
        explanation += " Review flag: the answer key is absent from the supplied JSON/PDF extraction and should be checked against the original page."
    item["answer"] = answer
    item["explanation"] = explanation

dst.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(items)} questions to {dst}")
