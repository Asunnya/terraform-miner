import os
import re


def normalize_path(path: str) -> str:
    """
    Normaliza separadores, remove barras duplicadas e converte para lowercase.
    """
    p = path.replace("\\", "/")
    p = re.sub(r"/+", "/", p)
    return p.lower().strip("/")


def is_real_infra_file(path: str) -> bool:
    """
    Retorna True se o arquivo .tf for parte da infraestrutura principal,
    excluindo exemplos, testes, docs, fixtures e snippets.
    """
    p = normalize_path(path)
    exclude_dirs = {
        'examples', 'example',
        'test', 'tests',
        'docs', 'fixtures', 'sample', 'mock',
    }
    exclude_files = {
        'example.tf', 'test.tf', 'tests.tf',
        'fixture.tf', 'sample.tf', 'mock.tf'
    }

    filename = os.path.basename(p)
    if filename in exclude_files:
        return False

    parts = p.split('/')
    for part in parts[:-1]:  
        if part in exclude_dirs:
            return False

    return filename.endswith('.tf')