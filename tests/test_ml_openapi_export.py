"""The released contract is deterministic and network-free."""

def test_export_is_deterministic_and_does_not_enter_runtime(monkeypatch):
    import socket
    from embedding import runtime
    from scripts.export_ml_openapi import build_schema, serialized_schema, PRODUCT_PATHS

    def forbidden(*args, **kwargs):
        raise AssertionError('OpenAPI export must not start model or network work')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(runtime, '_load_sentence_transformer', forbidden)
    assert serialized_schema() == serialized_schema()
    schema = build_schema()
    assert set(schema['paths']) == set(PRODUCT_PATHS)
    assert schema['info']['x-online-product-database-access'] is False
    for path in PRODUCT_PATHS:
        assert schema['paths'][path]['post']['security'] == [{'InternalSecret': []}]
