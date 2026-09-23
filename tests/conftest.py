import os
import tempfile

import pytest
from git import Actor, Repo

# github_api.py chama logging.basicConfig(FileHandler('github_api.log')) ao ser importado,
# criando o arquivo no diretório atual. Importa aqui, dentro de um diretório temporário,
# para que os testes não deixem esse log na raiz do repo.
_cwd = os.getcwd()
_log_dir = tempfile.mkdtemp(prefix="terraform-miner-tests-")
os.chdir(_log_dir)
try:
    import src.miner.github_api  # noqa: F401
finally:
    os.chdir(_cwd)


AUTHOR = Actor("Test Author", "author@example.com")


def _commit(repo, files, message):
    for rel_path, content in files.items():
        abs_path = os.path.join(repo.working_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    repo.index.add(list(files))
    return repo.index.commit(message, author=AUTHOR, committer=AUTHOR).hexsha


@pytest.fixture
def git_repo(tmp_path):
    """
    Repo git real com 5 commits:
      c1 "initial"          main.tf
      c2 "fix: wrong cidr"  main.tf
      c3 "Fix bug in docs"  README.md
      c4 "fix: example"     examples/demo.tf
      c5 "add feature"      variables.tf
    Retorna (caminho, {"c1": hash, ...}).
    """
    path = tmp_path / "owner_repo"
    repo = Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", AUTHOR.name)
        cw.set_value("user", "email", AUTHOR.email)
        cw.set_value("commit", "gpgsign", "false")

    hashes = {
        "c1": _commit(repo, {"main.tf": 'resource "aws_vpc" "main" {\n  cidr_block = "10.0.0.0/8"\n}\n'}, "initial"),
        "c2": _commit(repo, {"main.tf": 'resource "aws_vpc" "main" {\n  cidr_block = "10.0.0.0/16"\n}\n'}, "fix: wrong cidr"),
        "c3": _commit(repo, {"README.md": "# docs\n"}, "Fix bug in docs"),
        "c4": _commit(repo, {"examples/demo.tf": 'module "demo" {\n  source = "../"\n}\n'}, "fix: example"),
        "c5": _commit(repo, {"variables.tf": 'variable "region" {}\n'}, "add feature"),
    }
    return str(path), hashes
