from tuner.discovery import _public_symbols


def test_discovery_captures_signatures_config_exports_and_pairs(tmp_path):
    root = tmp_path / "example"
    root.mkdir()
    (root / "__init__.py").write_text('from .api import Client as Client\n__all__ = ["Client"]\n')
    path = root / "api.py"
    path.write_text("""class Client:
    def call(self, count: int = 1) -> str: pass
    async def call_async(self, count: int = 1) -> str: pass

class Config:
    limit: int = 10
""")
    entries = {item["symbol"]: item for item in _public_symbols(root, "test")}
    assert entries["example.api.Client.call"]["parameters"] == "self, count: int=1"
    assert entries["example.api.Client.call"]["paired_symbol"] == "example.api.Client.call_async"
    assert entries["example.api.Config"]["fields"]["limit"] == {
        "annotation": "int",
        "default": "10",
    }
    assert entries["example.Client"]["target"] == "example.api.Client"
    assert entries["example.Client"]["explicit_export"]
    assert all("\\" not in item["file"] for item in entries.values())
    path.write_text(path.read_text().replace("count: int = 1", "count: int = 2"))
    changed = {item["symbol"]: item for item in _public_symbols(root, "test")}
    assert entries.keys() == changed.keys()
    assert entries["example.api.Client.call"] != changed["example.api.Client.call"]


def test_identically_named_classes_in_different_modules_have_unique_ids(tmp_path):
    root = tmp_path / "example"
    root.mkdir()
    for name in ("first", "second"):
        (root / f"{name}.py").write_text("class Client:\n    def call(self): pass\n")
    entries = _public_symbols(root, "test")
    assert len({item["symbol"] for item in entries}) == len(entries)
