"""agentcore must stay generic: no hospital imports, no hospital words."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "agentcore"
FORBIDDEN_IMPORTS = {"agent_service", "mock_hospital", "evals", "tests"}
FORBIDDEN_WORDS = ("hospital", "doctor", "appointment", "patient", "bill")


def _py_files():
    return sorted(ROOT.rglob("*.py"))


def test_agentcore_has_no_domain_imports():
    for path in _py_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in FORBIDDEN_IMPORTS, f"{path.name} imports {name}"


def test_agentcore_has_no_domain_words():
    for path in _py_files():
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{path.name} mentions '{word}'"
