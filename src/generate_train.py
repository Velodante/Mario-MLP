import re
from pathlib import Path

# Carpeta donde está este script
SCRIPT_DIR = Path(__file__).resolve().parent

# Ajusta la cantidad de .parent según dónde ubiques el script
repo_root = SCRIPT_DIR.parent

data_dir = repo_root / "Super Mario Bros. (World)" / "Data"


def crear_train_vacio():
    existing_files = list(data_dir.glob("train_*.csv"))

    nums = []
    for file in existing_files:
        m = re.search(r"train_(\d+)\.csv", file.name)
        if m:
            nums.append(int(m.group(1)))

    siguiente = max(nums) + 1 if nums else 1

    csv_path = data_dir / f"train_{siguiente}.csv"
    config_path = repo_root / "LuaScriptData" / "dataset_path.txt"

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(str(csv_path) + "\n")
        f.write(str(data_dir) + "\n")

    print(f"Nuevo archivo CSV creado: {csv_path}")
    print(f"Configuración actualizada en: {config_path}")

    return csv_path, data_dir


def main():
    print("Definiendo nuevo train...")
    crear_train_vacio()


if __name__ == "__main__":
    main()