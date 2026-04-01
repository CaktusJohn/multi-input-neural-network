import zipfile
from pathlib import Path
import sys

import mlflow
import pandas as pd


RUN_ID = "7b6e91acc03043159be5835f1b43f9e5"
OUTPUT_DIR = "shared_run_report_attention"


def main():
    if RUN_ID == "PASTE_RUN_ID_HERE":
        raise ValueError("Укажи RUN_ID в начале файла.")

    run_id = RUN_ID
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Экспорт run_id={run_id}")
    print(f"Папка отчета: {output_dir.resolve()}")

    run = mlflow.get_run(run_id)
    rows = []
    for key, value in run.data.params.items():
        rows.append({"section": "param", "key": key, "value": value})
    for key, value in run.data.metrics.items():
        rows.append({"section": "metric", "key": key, "value": value})
    for key, value in run.data.tags.items():
        rows.append({"section": "tag", "key": key, "value": value})

    summary_df = pd.DataFrame(rows, columns=["section", "key", "value"]).sort_values(
        ["section", "key"]
    )
    summary_path = output_dir / "run_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Сохранено: {summary_path}")

    # Нужен импорт модуля с классом модели для корректной десериализации pickle.
    import scripts.multi_input_model  # noqa: F401

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    with (output_dir / "model_structure.txt").open("w", encoding="utf-8") as f:
        f.write(repr(model))

    model_rows = []
    for name, parameter in model.named_parameters():
        model_rows.append(
            {
                "param_name": name,
                "shape": tuple(parameter.shape),
                "num_params": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "dtype": str(parameter.dtype),
            }
        )

    pd.DataFrame(model_rows).to_csv(
        output_dir / "model_parameters.csv", index=False, encoding="utf-8"
    )
    print("Сохранены: model_structure.txt, model_parameters.csv")

    zip_path = output_dir / "run_report.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path != zip_path:
                archive.write(file_path, arcname=file_path.relative_to(output_dir))

    print(f"Готово. Архив для отправки: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
