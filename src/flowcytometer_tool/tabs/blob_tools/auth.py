from azure.identity import InteractiveBrowserCredential, TokenCachePersistenceOptions

_credential = None

def get_credential():
    global _credential
    if _credential is None:
        _credential = InteractiveBrowserCredential(
            cache_persistence_options=TokenCachePersistenceOptions(enabled=True)
        )
    return _credential