from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
DATA_PATH = ROOT / "dataset_labeled_wsn.xlsx"
MODEL_PATH = ROOT / "rf_wsn_multiclass_best_pipeline.pkl"
PDF_PATH = ROOT / "Referensi_Model_WSN_Detail.pdf"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def _parse_training_summary(log_text: str) -> list[list[str]]:
    rows = [["Model", "Accuracy", "F1_macro", "Precision_macro", "Recall_macro"]]
    line_pattern = re.compile(
        r"^\s*\d+\s+([A-Za-z ]+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        flags=re.MULTILINE,
    )
    for m in line_pattern.finditer(log_text):
        rows.append(
            [
                m.group(1).strip(),
                m.group(2),
                m.group(3),
                m.group(4),
                m.group(5),
            ]
        )
    return rows


def _parse_feature_importance(log_text: str) -> list[list[str]]:
    rows = [["Feature", "Importance"]]
    in_block = False
    for line in log_text.splitlines():
        if "=== FEATURE IMPORTANCE - RANDOM FOREST ===" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip() == "":
            continue
        if line.startswith(".venv") or line.startswith("cleaning.ipynb#"):
            break
        m = re.match(r"^\s*\d+\s+([a-zA-Z0-9_]+)\s+([0-9.]+)\s*$", line)
        if m:
            rows.append([m.group(1), m.group(2)])
    return rows


def _parse_permutation_importance(log_text: str) -> list[list[str]]:
    rows = [["Feature", "Importance_mean"]]
    in_block = False
    for line in log_text.splitlines():
        if "=== PERMUTATION IMPORTANCE ===" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if line.strip() == "":
            continue
        if "=== PDP - MODEL TUNED ===" in line:
            break
        m = re.match(r"^\s*\d+\s+([a-zA-Z0-9_]+)\s+([0-9.]+)\s*$", line)
        if m:
            rows.append([m.group(1), m.group(2)])
    return rows


def _parse_noise_table(log_text: str) -> list[list[str]]:
    rows = [["Noise_level", "F1_macro"]]
    in_block = False
    for line in log_text.splitlines():
        if "=== HASIL NOISE ROBUSTNESS ===" in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(r"^\s*\d+\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m:
            rows.append([m.group(1), m.group(2)])
            continue
        if rows and len(rows) > 1 and line.strip() == "":
            break
    return rows


def _parse_ablation_table(log_text: str) -> list[list[str]]:
    rows = [["Removed_feature", "F1_after_removal", "F1_drop"]]
    in_block = False
    for line in log_text.splitlines():
        if "=== HASIL FEATURE ABLATION ===" in line:
            in_block = True
            continue
        if not in_block:
            continue
        m = re.match(
            r"^\s*\d+\s+([a-zA-Z0-9_]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
            line,
        )
        if m:
            rows.append([m.group(1), m.group(2), m.group(3)])
            continue
        if rows and len(rows) > 1 and line.strip() == "":
            break
    return rows


def _make_table(data: list[list[str]], col_widths=None) -> Table:
    table = Table(data, hAlign="LEFT", colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_text = safe_text.replace("\n", "<br/>")
    return Paragraph(safe_text, style)


def build_pdf() -> Path:
    training_log = _read_text(LOG_DIR / "run_training_cleaning.log")
    testing_log = _read_text(LOG_DIR / "run_model_dev.log")
    experiment_log = _read_text(LOG_DIR / "run_eksperimen.log")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    df = pd.read_excel(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    label_counts = Counter(y)

    pipe = joblib.load(MODEL_PATH)
    rf = pipe.named_steps.get("rf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        spaceAfter=10,
    )
    h_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )
    n_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=5,
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    story = []
    story.append(_p("Dokumentasi Referensi Model WSN", title_style))
    story.append(
        _p(
            "Dokumen ini menjelaskan alur pemrosesan model, training, testing, log yang dicatat, "
            "dan komponen XAI yang sudah dipakai (tanpa SHAP).",
            n_style,
        )
    )
    story.append(_p(f"Tanggal generate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", n_style))
    story.append(Spacer(1, 8))

    story.append(_p("1) Artefak Dan Sumber Log", h_style))
    story.append(
        _p(
            f"- Dataset utama: {DATA_PATH.name}\n"
            f"- Model final: {MODEL_PATH.name}\n"
            "- Log training: logs/run_training_cleaning.log\n"
            "- Log testing final + XAI: logs/run_model_dev.log\n"
            "- Log eksperimen tambahan: logs/run_eksperimen.log",
            n_style,
        )
    )

    story.append(_p("2) Profil Dataset Dan Target", h_style))
    story.append(
        _p(
            f"- Jumlah baris: {len(df)}\n"
            f"- Jumlah fitur: {X.shape[1]}\n"
            f"- Nama fitur: {', '.join(X.columns)}\n"
            f"- Label target: {', '.join(sorted(label_counts.keys()))}",
            n_style,
        )
    )
    class_rows = [["Label", "Jumlah"]] + [[k, str(v)] for k, v in sorted(label_counts.items())]
    story.append(_make_table(class_rows, col_widths=[220, 100]))
    story.append(
        _p(
            f"Imbalance ratio total (kelas terbesar/terkecil): "
            f"{max(label_counts.values()) / min(label_counts.values()):.2f}",
            n_style,
        )
    )

    story.append(_p("3) Konfigurasi Model Final", h_style))
    steps = ", ".join(pipe.named_steps.keys())
    story.append(_p(f"Pipeline model: {steps}", n_style))
    param_rows = [
        ["Parameter RF", "Nilai"],
        ["n_estimators", str(getattr(rf, "n_estimators", ""))],
        ["max_depth", str(getattr(rf, "max_depth", ""))],
        ["min_samples_leaf", str(getattr(rf, "min_samples_leaf", ""))],
        ["min_samples_split", str(getattr(rf, "min_samples_split", ""))],
        ["max_features", str(getattr(rf, "max_features", ""))],
        ["random_state", str(getattr(rf, "random_state", ""))],
        ["n_jobs", str(getattr(rf, "n_jobs", ""))],
    ]
    story.append(_make_table(param_rows, col_widths=[180, 140]))

    story.append(_p("4) Alur Training Dan Log Yang Dicatat", h_style))
    story.append(
        _p(
            "Training baseline dilakukan dengan alur:\n"
            "1. load dataset berlabel,\n"
            "2. stratified train-test split (80:20, random_state=42),\n"
            "3. standardisasi fitur (fit hanya pada train),\n"
            "4. balancing train dengan SMOTE,\n"
            "5. training 4 model tree-based (Decision Tree, Random Forest, Gradient Boosting, Extra Trees),\n"
            "6. evaluasi per model pada test,\n"
            "7. rangkum metrik dan feature importance RF.",
            n_style,
        )
    )
    story.append(
        _p(
            "Log yang dicatat saat training:\n"
            "- distribusi label total/train/test,\n"
            "- distribusi sebelum dan sesudah SMOTE,\n"
            "- akurasi dan classification report per model,\n"
            "- confusion matrix raw per model,\n"
            "- tabel ringkasan metrik antar model,\n"
            "- model terbaik sementara,\n"
            "- feature importance Random Forest.",
            n_style,
        )
    )
    training_table = _parse_training_summary(training_log)
    if len(training_table) > 1:
        story.append(_make_table(training_table, col_widths=[120, 70, 70, 90, 90]))

    rf_fi = _parse_feature_importance(training_log)
    if len(rf_fi) > 1:
        story.append(Spacer(1, 4))
        story.append(_p("Feature importance Random Forest (log training):", n_style))
        story.append(_make_table(rf_fi, col_widths=[180, 120]))

    story.append(_p("5) Alur Testing Final Dan Validasi", h_style))
    story.append(
        _p(
            "Testing final (model_dev) mencakup:\n"
            "1. evaluasi test set,\n"
            "2. confusion matrix,\n"
            "3. recall per kelas,\n"
            "4. confidence score dari predict_proba,\n"
            "5. 5-fold cross-validation (scoring=f1_macro),\n"
            "6. cek overfitting train vs test,\n"
            "7. XAI global (Permutation Importance),\n"
            "8. XAI lokal-global (Partial Dependence Plot),\n"
            "9. feature ablation,\n"
            "10. noise robustness test.",
            n_style,
        )
    )

    summary_rows = [
        ["Metrik", "Nilai"],
        ["Test Accuracy", str(_extract_float(r"Test Accuracy:\s*([0-9.]+)", testing_log) or "-")],
        ["Test F1-macro", str(_extract_float(r"Test F1-macro:\s*([0-9.]+)", testing_log) or "-")],
        ["CV Mean F1-macro", str(_extract_float(r"Mean F1-macro:\s*([0-9.]+)", testing_log) or "-")],
        ["CV Std F1-macro", str(_extract_float(r"Std\s+F1-macro:\s*([0-9.]+)", testing_log) or "-")],
        ["Train Accuracy", str(_extract_float(r"Train Accuracy:\s*([0-9.]+)", testing_log) or "-")],
        ["Train-Test Gap", str(_extract_float(r"Gap\s*:\s*([0-9.]+)", testing_log) or "-")],
    ]
    story.append(_make_table(summary_rows, col_widths=[180, 180]))

    perm_rows = _parse_permutation_importance(testing_log)
    if len(perm_rows) > 1:
        story.append(Spacer(1, 4))
        story.append(_p("Permutation importance (global XAI):", n_style))
        story.append(_make_table(perm_rows, col_widths=[200, 120]))

    ablation_rows = _parse_ablation_table(testing_log)
    if len(ablation_rows) > 1:
        story.append(Spacer(1, 4))
        story.append(_p("Feature ablation (impact ke F1-macro):", n_style))
        story.append(_make_table(ablation_rows, col_widths=[140, 110, 90]))

    noise_rows = _parse_noise_table(testing_log)
    if len(noise_rows) > 1:
        story.append(Spacer(1, 4))
        story.append(_p("Noise robustness (f1_macro vs noise level):", n_style))
        story.append(_make_table(noise_rows, col_widths=[120, 120]))

    story.append(_p("6) Pertimbangan Utama Saat Menentukan Skema Tes", h_style))
    story.append(
        _p(
            "Pertimbangan yang dipakai:\n"
            "- Data imbalance tinggi (IR ~40.50), sehingga evaluasi tidak hanya accuracy.\n"
            "- Metrik utama untuk fairness antar kelas: F1_macro, precision_macro, recall_macro.\n"
            "- Recall per kelas dipantau agar kelas fault minor tidak terlewat.\n"
            "- Overfitting dicek lewat gap train-test + stabilitas CV (mean/std).\n"
            "- Robustness diuji dengan noise injection dan feature ablation.\n"
            "- Real-time feasibility diuji lewat inference time test 10.000 sampel.\n"
            "- Reproducibility dijaga dengan random_state=42 pada split, SMOTE, dan model.",
            n_style,
        )
    )

    inference_total = _extract_float(
        r"Total waktu prediksi untuk 10000 sampel:\s*([0-9.]+)", experiment_log
    )
    inference_ms = _extract_float(r"Per sampel:\s*([0-9.]+) ms", experiment_log)
    if inference_total is not None and inference_ms is not None:
        story.append(
            _p(
                f"Hasil benchmark inference dari log eksperimen: "
                f"{inference_total:.4f} detik per 10.000 sampel "
                f"(~{inference_ms:.4f} ms per sampel).",
                n_style,
            )
        )

    story.append(_p("7) Ringkasan XAI Yang Sudah Digunakan", h_style))
    story.append(
        _p(
            "XAI yang aktif saat ini:\n"
            "- Permutation Importance: mengukur penurunan performa saat satu fitur diacak.\n"
            "- Partial Dependence Plot (PDP): menunjukkan hubungan nilai fitur terhadap output kelas target.\n"
            "- Feature Ablation: mengukur drop F1 saat fitur dihapus total dari proses training ulang.\n"
            "Catatan: SHAP belum diaktifkan pada alur saat ini.",
            n_style,
        )
    )

    story.append(_p("8) Catatan Operasional", h_style))
    story.append(
        _p(
            "Untuk menjalankan ulang log referensi:\n"
            "- python pythonver/model_dev-readonly.py\n"
            "- python pythonver/eksperimen_readonlynotrun..py\n"
            "- jalankan cell training utama pada cleaning.ipynb (cell pipeline training final)\n"
            "Saat mode headless, warning FigureCanvasAgg dapat muncul karena plt.show(). "
            "Warning ini tidak mengubah hasil metrik numerik.",
            n_style,
        )
    )

    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    output = build_pdf()
    print(f"PDF generated: {output}")
