import csv
from pathlib import Path

from recognition import identify

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
OUT_DIR = ROOT / "outputs"

SCENARIOS = {
    "enrolled": "Usuário cadastrado",
    "unknown": "Usuário não cadastrado",
    "invalid": "Entrada inválida",
}
SUMMARY_HEADERS = ["Cenário", "Testes", "Acertos", "Erros", "Acurácia"]
DETAIL_HEADERS = ["Imagem", "Cenário", "Esperado", "Obtido", "Score", "Resultado"]


def expected_of(photo):
    label = photo.stem.rsplit("_", 1)[0]
    if label in ("unknown", "invalid"):
        return label, label
    return "enrolled", label


def describe(status, user):
    if status == "recognized":
        return user
    return "não reconhecido" if status == "unknown" else "sem rosto"


def evaluate():
    details = []
    for photo in sorted(TEST_DIR.glob("*.jpg")):
        scenario, label = expected_of(photo)
        r = identify(str(photo))
        if scenario == "enrolled":
            correct = r["status"] == "recognized" and r["user"] == label
            expected = label
        else:
            correct = r["status"] == scenario
            expected = describe(scenario, None)
        details.append([
            photo.name,
            SCENARIOS[scenario],
            expected,
            describe(r["status"], r["user"]),
            "-" if r["score"] is None else f'{r["score"]:.3f}',
            "Acerto" if correct else "Erro",
        ])

    summary = []
    for name in SCENARIOS.values():
        rows = [d for d in details if d[1] == name]
        hits = sum(d[5] == "Acerto" for d in rows)
        summary.append([name, len(rows), hits, len(rows) - hits,
                        f"{hits / len(rows):.1%}" if rows else "-"])
    total = len(details)
    hits = sum(d[5] == "Acerto" for d in details)
    accuracy = hits / total if total else 0.0
    summary.append(["Total", total, hits, total - hits, f"{accuracy:.1%}"])
    return summary, details, accuracy


def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line.format(*[str(x) for x in row]))


if __name__ == "__main__":
    summary, details, accuracy = evaluate()
    print_table(DETAIL_HEADERS, details)
    print()
    print_table(SUMMARY_HEADERS, summary)
    print(f"\nAcurácia global: {accuracy:.1%}")
    OUT_DIR.mkdir(exist_ok=True)
    save_csv(OUT_DIR / "evaluation_summary.csv", SUMMARY_HEADERS, summary)
    save_csv(OUT_DIR / "evaluation_details.csv", DETAIL_HEADERS, details)
    print("Tabelas salvas em outputs/evaluation_summary.csv e outputs/evaluation_details.csv")
