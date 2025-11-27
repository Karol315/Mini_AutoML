import os
import re

IGNORE = {'.git', '__pycache__', '.DS_Store'}

def tree(dir_path, prefix=""):
    """Zwraca strukturę folderów i plików w formie drzewa, ignorując pliki systemowe."""
    items = sorted([item for item in os.listdir(dir_path) if item not in IGNORE])
    tree_str = ""
    for index, item in enumerate(items):
        path = os.path.join(dir_path, item)
        connector = "└─ " if index == len(items) - 1 else "├─ "
        tree_str += f"{prefix}{connector}{item}\n"
        if os.path.isdir(path):
            extension = "   " if index == len(items) - 1 else "│  "
            tree_str += tree(path, prefix + extension)
    return tree_str

# Ścieżka do katalogu projektu
project_dir = "."

# Generowanie drzewa
tree_structure = tree(project_dir)

# Treść do wstawienia w sekcję
tree_section = f"## Struktura plików:\n```\n{tree_structure}```\n"

readme_file = "README.md"

# Wczytaj istniejący README
try:
    with open(readme_file, "r", encoding="utf-8") as f:
        readme_content = f.read()
except FileNotFoundError:
    readme_content = "# Mini_AutoML\n\nProjekt został zrealizowany w ramach kursu AutoML na Politechnice Warszawskiej.\n\n## Autorzy:\n- Karol Kacprzak\n- Ludwik Madej\n- Mikołaj Bójski\n\n"

# Sprawdź, czy sekcja '## Struktura plików' już istnieje
pattern = r"## Struktura plików:\n```[\s\S]*?```"
if re.search(pattern, readme_content):
    # Zamień istniejącą sekcję
    readme_content = re.sub(pattern, tree_section, readme_content)
else:
    # Dodaj na końcu README
    if not readme_content.endswith("\n"):
        readme_content += "\n"
    readme_content += tree_section

# Zapisz README
with open(readme_file, "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README.md został zaktualizowany z aktualną strukturą plików (bez plików systemowych)!")
