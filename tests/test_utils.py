import pytest

from src.miner.utils import is_real_infra_file, normalize_path


@pytest.mark.parametrize("raw, expected", [
    ("Modules\\VPC\\Main.tf", "modules/vpc/main.tf"),
    ("a//b///c.tf", "a/b/c.tf"),
    ("/a/b/", "a/b"),
    ("main.tf", "main.tf"),
])
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize("path, expected", [
    ("main.tf", True),
    ("modules/vpc/main.tf", True),
    ("Modules/VPC/Main.TF", True),
    ("examples/x.tf", False),
    ("example/x.tf", False),
    ("test/x.tf", False),
    ("tests/unit/x.tf", False),
    ("docs/x.tf", False),
    ("a/fixtures/b.tf", False),
    ("a/sample/b.tf", False),
    ("a/mock/b.tf", False),
    ("Examples\\X.TF", False),
    ("example.tf", False),
    ("modules/mock.tf", False),
    ("README.md", False),
    ("main.tf.json", False),
    ("examples.tf", True),  # só diretórios/nomes exatos são excluídos
])
def test_is_real_infra_file(path, expected):
    assert is_real_infra_file(path) is expected
