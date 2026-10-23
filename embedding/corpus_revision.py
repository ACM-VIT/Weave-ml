"""Opt-in managed corpus authority; never infer safety from a timer or local role.

Requires an enforced single authoritative endpoint, fenced former primaries,
managed admin/restore access, and complete writer participation. Redis INFO
cannot prove those external prerequisites. Defaults retain uncached serving.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit, unquote
from uuid import uuid4

T = TypeVar('T')


class CorpusAuthorityUnavailable(ConnectionError):
    """An enabled writer must retry before dispatch or preserve unknown DIRTY work."""


@dataclass(frozen=True)
class MaintenanceProof:
    proof_id: str
    old_writers_retired: bool
    qdrant_operations_completed: bool
    former_primaries_fenced: bool
    admin_boundary_managed: bool

    def __post_init__(self):
        if not isinstance(self.proof_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', self.proof_id):
            raise ValueError('maintenance proof ID must be bounded and non-sensitive')
        if not all(value is True for value in (
            self.old_writers_retired, self.qdrant_operations_completed,
            self.former_primaries_fenced, self.admin_boundary_managed,
        )):
            raise ValueError('all old writers/operations and topology/admin prerequisites require proof')


def operation_completed(result: Any) -> bool:
    status = getattr(result, 'status', None)
    return getattr(status, 'value', status) == 'completed'


# INFO and record access are in ONE atomic script. No ROLE (NOSCRIPT), expiry,
# WAIT consistency assumption, process-local generation or automatic bootstrap.
_PROTOCOL = r'''
local op,scope,maximum,token,nonce,proof=unpack(ARGV)
maximum=tonumber(maximum)
local server=redis.call('INFO','server')
local replication=redis.call('INFO','replication')
local run=string.match(server,'run_id:([^\r\n]+)')
local repl=string.match(replication,'master_replid:([^\r\n]+)')
local role=string.match(replication,'role:([^\r\n]+)')
if not run or not repl or role~='master' then return {'UNAVAILABLE'} end
local epoch=run..'|'..repl
local function get(field) return redis.call('HGET',KEYS[1],field) end
local function valid_nonce(value)
 return value and #value==32 and not string.find(value,'[^0-9a-f]')
end
if not maximum or maximum<1 or maximum>64 or #scope>128 then return {'UNAVAILABLE'} end
if op=='bootstrap' then
 if #proof<1 or #proof>128 or not valid_nonce(nonce) then return {'UNAVAILABLE'} end
 -- Only a controlled maintenance caller may attest retirement/completion. No
 -- amount of time or absence of writer records substitutes for that proof.
 redis.call('HSET',KEYS[1],'v','1','scope',scope,'epoch',epoch,'state','READY',
  'corpus',nonce,'eligibility',nonce,'writers','{}','proof',proof)
 return {'OK'}
end
if get('v')~='1' or get('scope')~=scope or get('epoch')~=epoch then return {'UNAVAILABLE'} end
local encoded=get('writers')
if not encoded or #encoded>4096 then return {'UNAVAILABLE'} end
local parsed,writers=pcall(cjson.decode,encoded)
if not parsed or type(writers)~='table' then return {'UNAVAILABLE'} end
local count=0
for id,value in pairs(writers) do
 if not valid_nonce(id) or value~=true then return {'UNAVAILABLE'} end
 count=count+1
 if count>64 then return {'UNAVAILABLE'} end
end
local state=get('state')
local corpus,eligibility=get('corpus'),get('eligibility')
if not valid_nonce(corpus) or not valid_nonce(eligibility) then return {'UNAVAILABLE'} end
if state~='READY' and state~='DIRTY' and state~='UNTRUSTED' then return {'UNAVAILABLE'} end
if state=='READY' and count~=0 then return {'UNAVAILABLE'} end
if state=='DIRTY' and count==0 then return {'UNAVAILABLE'} end
if op=='status' then return {'STATUS',state,tostring(count)} end
if op=='read' then
 if state=='READY' then return {'READY',corpus,eligibility} end
 return {'UNAVAILABLE'}
end
if state=='UNTRUSTED' or not valid_nonce(token) or not valid_nonce(nonce) then return {'UNAVAILABLE'} end
if op=='begin' then
 if count>=maximum or writers[token] then
  redis.call('HSET',KEYS[1],'state','UNTRUSTED')
  return {'UNAVAILABLE'}
 end
 writers[token]=true
 redis.call('HSET',KEYS[1],'state','DIRTY','writers',cjson.encode(writers),
  'corpus',nonce,'eligibility',nonce)
 return {'OK'}
end
if op=='finish' then
 if not writers[token] then return {'UNAVAILABLE'} end
 writers[token]=nil
 local next_state='DIRTY'
 if count==1 then next_state='READY' end
 redis.call('HSET',KEYS[1],'state',next_state,'writers',cjson.encode(writers),
  'corpus',nonce,'eligibility',nonce)
 return {'OK'}
end
return {'UNAVAILABLE'}
'''


class CorpusRevisionAuthority:
    def __init__(self, client: Any, *, key: str, scope: str,
                 managed_boundary_attested: bool, max_writers: int = 64):
        if managed_boundary_attested is not True:
            raise ValueError('managed single-endpoint/admin boundary must be attested explicitly')
        if not re.fullmatch(r'[A-Za-z0-9_:.{}-]{1,192}', key):
            raise ValueError('authority key must be bounded')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', scope):
            raise ValueError('authority scope must be bounded')
        if type(max_writers) is not int or not 1 <= max_writers <= 64:
            raise ValueError('authority writer bound must be between 1 and 64')
        self.client, self.key, self.scope, self.max_writers = client, key, scope, max_writers

    def _call(self, operation: str, *, token='', proof='') -> list[str]:
        try:
            result = self.client.eval(_PROTOCOL, 1, self.key, operation, self.scope,
                                      self.max_writers, token, uuid4().hex, proof)
            if not isinstance(result, (list, tuple)) or len(result) > 3:
                raise ValueError('invalid authority response')
            values = [item.decode('ascii') if isinstance(item, bytes) else str(item) for item in result]
            if any(len(item) > 128 for item in values):
                raise ValueError('unbounded authority response')
            return values
        except Exception as exc:
            raise CorpusAuthorityUnavailable(
                f'corpus revision authority unavailable ({type(exc).__name__})'
            ) from None

    def close(self) -> None:
        close = getattr(self.client, 'close', None)
        if callable(close):
            close()

    def read(self) -> tuple[str, str] | None:
        try:
            result = self._call('read')
        except CorpusAuthorityUnavailable:
            return None  # Serving stays genuinely uncached on authority failure.
        if len(result) == 3 and result[0] == 'READY':
            if all(re.fullmatch(r'[0-9a-f]{32}', value) for value in result[1:]):
                return result[1], result[2]
        return None

    def status(self) -> dict[str, Any]:
        result = self._call('status')
        if len(result) == 3 and result[0] == 'STATUS':
            return {'state': result[1], 'writers': int(result[2])}
        return {'state': 'UNAVAILABLE', 'writers': None}

    def bootstrap(self, proof: MaintenanceProof) -> None:
        if not isinstance(proof, MaintenanceProof):
            raise ValueError('explicit completed maintenance proof required')
        if self._call('bootstrap', proof=proof.proof_id) != ['OK']:
            raise CorpusAuthorityUnavailable('corpus authority recovery refused')

    def begin(self) -> str:
        token = uuid4().hex
        if self._call('begin', token=token) != ['OK']:
            raise CorpusAuthorityUnavailable('corpus authority refused write before dispatch')
        return token

    def finish(self, token: str, *, completed: bool) -> bool:
        if not re.fullmatch(r'[0-9a-f]{32}', token):
            raise ValueError('invalid writer token')
        if completed is not True:
            return False  # Unknown/acknowledged/failed operations never clear DIRTY.
        if self._call('finish', token=token) != ['OK']:
            raise CorpusAuthorityUnavailable('corpus authority could not finish writer')
        return True

    def mutate(self, operation: Callable[[], T], *, synchronous_boolean: bool = False) -> T:
        token = self.begin()  # A failure here MUST prevent dispatch to Qdrant.
        result = operation()  # Exceptions preserve the non-expiring token.
        completed = result is True if synchronous_boolean else operation_completed(result)
        if not completed:
            raise CorpusAuthorityUnavailable('Qdrant operation completion is not proven; authority remains DIRTY')
        self.finish(token, completed=True)
        return result


def authority_from_environment(*, collection: str, vector: str, endpoint: str | None = None) -> CorpusRevisionAuthority | None:
    enabled = os.getenv('CORPUS_AUTHORITY_ENABLED', 'false').strip().casefold()
    if enabled not in {'false','0','true','1'}:
        raise ValueError('CORPUS_AUTHORITY_ENABLED must be true or false')
    if enabled in {'false','0'}:
        return None
    if os.getenv('CORPUS_AUTHORITY_MANAGED_BOUNDARY') != 'single-endpoint-fenced-admin-v1':
        raise ValueError('enabled authority requires explicit managed topology/admin attestation')
    url = os.getenv('CORPUS_AUTHORITY_REDIS_URL') or os.getenv('REDIS_URL')
    if not url:
        raise ValueError('enabled corpus authority requires a scoped Redis URL')
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in {'redis', 'rediss'} and bool(parsed.hostname)
        if os.getenv('APP_ENV', 'development').casefold() == 'production':
            valid = valid and bool(parsed.username) and len(unquote(parsed.password or '')) >= 16 and not parsed.query
        if not valid:
            raise ValueError
    except ValueError:
        raise ValueError('corpus authority requires a valid scoped Redis URL and production ACL credentials') from None
    import redis
    scope = hashlib.sha256(json.dumps([
        endpoint or os.getenv('QDRANT_URL', 'http://localhost:6333'), collection, vector,
    ], separators=(',', ':')).encode()).hexdigest()
    try:
        client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2,
                                socket_timeout=2, health_check_interval=30)
    except Exception:
        raise ValueError('invalid scoped corpus authority Redis configuration') from None
    return CorpusRevisionAuthority(client, key='weave:corpus-authority:v1:' + scope,
                                   scope=scope, managed_boundary_attested=True)
