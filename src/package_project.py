
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT.parent / "AgriPredict_AI_Submission.zip"

EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}

EXCLUDE_FILES = {
    "package_project.py",
}

def main():
    added = 0

    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in PROJECT.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(PROJECT)

            if any(part in EXCLUDE_DIRS for part in relative.parts):
                continue

            if path.name in EXCLUDE_FILES:
                continue

            # Skip the raw PlantVillage dataset and other image datasets.
            if relative.parts[0] == "data":
                continue

            archive.write(path, Path(PROJECT.name) / relative)
            added += 1

    print("Project ZIP created successfully!")
    print("Files packaged:", added)
    print("ZIP location:", ZIP_PATH)
    print("Original project files were not changed.")

if __name__ == "__main__":
    main()
