import ast
from pathlib import Path


TRAINING_PATH = Path("src/flowcytometer_tool/misc/functions_training.py")
APP_PATH = Path("src/flowcytometer_tool/app/flow_cytometer_tool.py")


def _training_tree() -> ast.Module:
    return ast.parse(TRAINING_PATH.read_text(encoding="utf-8"))


def _train_model_fn() -> ast.FunctionDef:
    tree = _training_tree()
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_model"
    )


def test_bead_calibration_no_longer_depends_on_calibration_toggle():
    fn = _train_model_fn()
    if_nodes = [node for node in ast.walk(fn) if isinstance(node, ast.If)]
    prep_call_name = "prepare_training_dataframe_with_optional_bead_calibration"

    def _names_in_test(test_node):
        return {
            child.id
            for child in ast.walk(test_node)
            if isinstance(child, ast.Name)
        }

    def _contains_prepare_call(node):
        return any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == prep_call_name
            for child in ast.walk(node)
        )

    has_bead_samples_gate_for_prepare = any(
        "bead_samples" in _names_in_test(node.test)
        and "calibration_enabled" not in _names_in_test(node.test)
        and _contains_prepare_call(node)
        for node in if_nodes
    )
    assert has_bead_samples_gate_for_prepare

    has_calibration_enabled_gate_for_prepare = any(
        "calibration_enabled" in _names_in_test(node.test)
        and _contains_prepare_call(node)
        for node in if_nodes
    )
    assert not has_calibration_enabled_gate_for_prepare


def test_probabilistic_calibration_toggle_forwarded_to_train_classifier():
    fn = _train_model_fn()
    has_probabilistic_assignment = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            target_is_probabilistic = any(
                isinstance(target, ast.Name)
                and target.id == "probabilistic_calibration_enabled"
                for target in node.targets
            )
            if target_is_probabilistic:
                value_names = {child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)}
                if "calibration_enabled" in value_names:
                    has_probabilistic_assignment = True
                    break
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "probabilistic_calibration_enabled"
                and node.value is not None
            ):
                value_names = {child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)}
                if "calibration_enabled" in value_names:
                    has_probabilistic_assignment = True
                    break
    assert has_probabilistic_assignment

    train_classifier_calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "train_classifier"
    ]
    assert train_classifier_calls

    has_forwarded_keyword = any(
        any(
            kw.arg == "calibration_enabled"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "probabilistic_calibration_enabled"
            for kw in call.keywords
        )
        for call in train_classifier_calls
    )
    assert has_forwarded_keyword


def test_ui_checkbox_is_explicitly_probabilistic():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    checkbutton_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Checkbutton"
    ]
    assert checkbutton_calls

    has_probabilistic_label = any(
        any(
            kw.arg == "text"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "Enable Probabilistic Calibration"
            for kw in call.keywords
        )
        for call in checkbutton_calls
    )
    assert has_probabilistic_label
