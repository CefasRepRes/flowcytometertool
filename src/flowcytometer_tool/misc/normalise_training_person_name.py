
def _normalise_training_person_name(name):
    """Return a stable person key for expertise-matrix and XML upload matching."""
    if name is None:
        return ""
    return str(name).strip()
