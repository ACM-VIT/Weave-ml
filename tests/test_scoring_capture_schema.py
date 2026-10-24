"""Validate registered request JSON Schema against actual runtime boundaries."""
import json
from functools import lru_cache
from jsonschema import Draft202012Validator
from pydantic import ValidationError
import pytest
from api import main
from api.contracts import RecommendationRequest
from tests.test_scoring_provider import payload


@lru_cache(maxsize=1)
def _generated_request_validator():
    document = main.app.openapi()
    wire = document["paths"]["/api/v2/recommendations/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    schema = {**wire, "components": document["components"]}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("feed,mode,accepted", [
    (1, "absent", True), (2**53-1, "absent", True), (1, "null", True),
    (2**53, "absent", False), (0, "absent", False), (-1, "absent", False),
    (True, "absent", False), (1.5, "absent", False), ("1", "absent", False),
    ("1", "null", False), (1, "captured", False), (2**53-1, "captured", False),
    ("1", "captured", True), (str(2**53+1), "captured", True),
    (str(2**63-1), "captured", True), (str(2**63), "captured", False),
    ("9999999999999999999", "captured", False), ("10000000000000000000", "captured", False),
    ("0", "captured", False), ("01", "captured", False), ("+1", "captured", False),
    (" 1", "captured", False), ("1 ", "captured", False), ("1\n", "captured", False),
    ("1\r\n", "captured", False), ("١", "captured", False), (True, "captured", False),
])
def test_registered_generated_schema_and_runtime_match_negotiation_boundaries(feed, mode, accepted):
    body = payload(captured=False, feed_version=feed)
    if mode == "null":
        body["scoring_capture"] = None
    elif mode == "captured":
        body["scoring_capture"] = {"schema_version": 1, "encoding": "float32-le-v1"}
    assert _generated_request_validator().is_valid(body) is accepted
    try:
        RecommendationRequest.model_validate_json(json.dumps(body))
        runtime_valid = True
    except ValidationError:
        runtime_valid = False
    assert runtime_valid is accepted


def test_json_schema_integer_numeric_semantics_do_not_change_ordinary_strict_int():
    # JSON Schema defines 1.0 as an integer. It cannot distinguish the lexical
    # numeric representation; preserve the pre-existing strict runtime contract.
    body = payload(captured=False, feed_version=1.0)
    assert _generated_request_validator().is_valid(body)
    with pytest.raises(ValidationError):
        RecommendationRequest.model_validate_json(json.dumps(body))


def test_generated_decimal_maximum_pattern_at_every_prefix_boundary():
    maximum = 2**63-1
    values = {1, maximum, maximum+1}
    for power in range(19):
        scale = 10**power
        boundary = maximum // scale * scale
        values.update((max(1, boundary-1), max(1, boundary), boundary+1, 10**(power+1)-1))
    validator = _generated_request_validator()
    for value in sorted(values):
        body = payload(feed_version=str(value))
        expected = value <= maximum
        assert validator.is_valid(body) is expected, value
        if expected:
            assert RecommendationRequest.model_validate_json(json.dumps(body)).feed_version == str(value)
        else:
            with pytest.raises(ValidationError):
                RecommendationRequest.model_validate_json(json.dumps(body))


def test_capture_variant_keeps_the_complete_reusable_object_contract():
    # Code generators consume each oneOf arm independently. An anonymous
    # object loses its field types even when the common schema is strict.
    document = main.app.openapi()
    request = document["components"]["schemas"]["RecommendationRequest"]
    branch = next(row for row in request["oneOf"] if "scoring_capture" in row["required"])
    capture = branch["properties"]["scoring_capture"]
    validator = Draft202012Validator({**capture, "components": document["components"]})
    assert validator.is_valid({"schema_version": 1, "encoding": "float32-le-v1"})
    assert not validator.is_valid({})
    assert not validator.is_valid({"unexpected": "field"})
    assert not validator.is_valid({"schema_version": 1, "encoding": "wrong"})


@pytest.mark.parametrize('field,value,accepted',[
    ('epoch','1',True),('epoch','0',False),('epoch',1,False),('epoch','01',False),
    ('as_of_ms','0',True),('as_of_ms',str(2**63-1),True),('as_of_ms',str(2**63),False),
    ('minimum_applied_operation_version','1\n',False),('required_profile_version','+1',False),
])
def test_model_context_negotiation_schema_matches_actual_runtime(field,value,accepted):
    body=payload(captured=False,feed_version='1')
    body['model_context']={'schema_version':1,'session_id':'00000000-0000-0000-0000-000000000001',
        'epoch':'1','policy_sha256':'a'*64,'minimum_applied_operation_version':'1','required_profile_version':'1','as_of_ms':'1'}
    body['model_context'][field]=value
    assert _generated_request_validator().is_valid(body) is accepted
    try:RecommendationRequest.model_validate_json(json.dumps(body));valid=True
    except ValidationError:valid=False
    assert valid is accepted
