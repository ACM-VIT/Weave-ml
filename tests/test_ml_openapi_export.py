"""The released contract is deterministic and network-free on a cold import."""
from pathlib import Path
import subprocess
import sys
import textwrap


def test_export_is_deterministic_and_does_not_enter_runtime():
    # Collection elsewhere imports api.main. A new interpreter is essential:
    # guards must precede every application import, including transitive imports.
    guarded_export = textwrap.dedent('''
        import socket
        import sys
        from types import ModuleType

        def forbidden(*args, **kwargs):
            raise AssertionError('OpenAPI export must not start model or network work')

        socket.create_connection = forbidden
        socket.socket.connect = forbidden
        socket.socket.connect_ex = forbidden
        transformer = ModuleType('sentence_transformers')
        transformer.SentenceTransformer = forbidden
        sys.modules['sentence_transformers'] = transformer
        assert 'api.main' not in sys.modules
        from embedding import runtime
        runtime._load_sentence_transformer = forbidden
        from scripts.export_ml_openapi import build_schema, serialized_schema, PRODUCT_PATHS

        assert serialized_schema() == serialized_schema()
        schema = build_schema()
        assert set(schema['paths']) == set(PRODUCT_PATHS)
        assert schema['info']['x-online-product-database-access'] is False
        for path in PRODUCT_PATHS:
            assert schema['paths'][path]['post']['security'] == [{'InternalSecret': []}]
        operation = schema['paths']['/api/v2/recommendations/generate']['post']
        capability = [p for p in operation['parameters'] if p['name'] == 'x-scoring-capture-capability']
        assert len(capability) == 1 and capability[0]['required'] is False
        assert operation['x-scoring-capture-security']['security'] == [
            {'InternalSecret': [], 'ScoringCaptureCapability': []}]
        assert set(schema['components']['schemas']['ScoringCaptureRequest']['required']) == {'schema_version', 'encoding'}
    ''')
    subprocess.run(
        [sys.executable, '-c', guarded_export], check=True,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=30,
    )
