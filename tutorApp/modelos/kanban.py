import subprocess
import pandas as pd
from datetime import datetime
import os

def get_commits(repo_path, source):
    command = ["git", "log", "--pretty=format:%h,%ad,%s", "--date=short"]
    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)

    lines = result.stdout.strip().split("\n")
    data = [line.split(",", 2) for line in lines if len(line.split(",", 2)) == 3]

    df = pd.DataFrame(data, columns=["hash", "date", "message"])
    df["source"] = source
    return df

def merge_commits(gitlab_path, github_path):
    gitlab_commits = get_commits(gitlab_path, "GitLab")
    github_commits = get_commits(github_path, "GitHub")

    commits = pd.concat([gitlab_commits, github_commits])
    commits.sort_values("date", inplace=True)
    return commits

def export_to_excel(commits, output_file="kanban_board.xlsx"):
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        commits.to_excel(writer, sheet_name="Kanban", index=False)

        # Estilo
        workbook = writer.book
        worksheet = writer.sheets["Kanban"]
        header_format = workbook.add_format({
            "bold": True,
            "bg_color": "#2196F3",
            "font_color": "white",
            "border": 1
        })

        for col_num, value in enumerate(commits.columns):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 30)

# 🔧 Cambia estas rutas a las de tus repos locales:
gitlab_repo = r"C:/Users/benas/Documents/Repositorios Clonados/tesis-tutor-inteligente"
github_repo = r"C:/Users/benas/Documents/Repo Tesis/tesis-doctutor-github"

# 🚀 Ejecutar todo
commits_df = merge_commits(gitlab_repo, github_repo)
export_to_excel(commits_df)
print("✅ Excel generado correctamente: kanban_board.xlsx")