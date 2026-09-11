from pathlib import Path
import ast


APP_PATH = Path("src/flowcytometer_tool/app/flow_cytometer_tool.py")


def _app_source_text() -> str:
    return APP_PATH.read_text(encoding="utf-8")


def test_training_button_uses_train_handler():
    src = _app_source_text()
    assert 'text="Authenticate and train Model", command=self.handle_train_model' in src


def test_train_handler_forwards_bead_samples_to_train_model():
    src = _app_source_text()
    assert "def handle_train_model(self):" in src
    assert "bead_samples=self.bead_samples" in src


def test_bead_samples_contract_is_documented_at_invocation_point():
    src = _app_source_text()
    assert "bead_samples contract" in src
    assert '"packet": packet_dict' in src
    assert '"dataframe" (or "df")' in src


def test_selected_cyz_bead_builder_has_one_implementation():
    tree = ast.parse(_app_source_text())
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnifiedApp")
    builders = [
        node for node in app_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_training_bead_samples_from_selected_cyz_paths"
    ]
    assert len(builders) == 1
