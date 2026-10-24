# Contributor evidence policy (ML #155)

The requester delegated this decision after reviewing the existing owner-presence
proxy. Remove the proxy, make no additional GitHub requests, and exclude the
legacy `mentionable_users_count` field from activity scoring. No observed
contributor count or reputation signal is currently available in canonical
enrichment. Missing evidence remains unknown; owner presence is identity evidence.

## Available sources and cost

| Source in this pipeline | Observation | Additional collection cost | Allowed use |
| --- | --- | --- | --- |
| Existing repository owner identity | Account/organization owning the repository | None beyond existing metadata query | Identity and lineage only; not community size or reputation |
| Existing default-branch commit sample | Already collected commit timestamps | None beyond existing bounded metadata query | Repository activity calculation; not distinct-contributor count |
| Existing push timestamp | Observed repository push age | None beyond existing metadata query | Thirty-day half-life activity fallback when commit samples are absent |
| Local `contributors=[]` compatibility argument | No contributor observation was collected | Zero extra calls | Unknown; never evidence of zero contributors |
| Legacy caller-supplied mentionable count | A deprecated unprovenanced numeric field | No new collection here | Wire compatibility only; no activity contribution or reputation claim |

A missing owner, sparse repository, unavailable community source, organization
owner and single-maintainer owner all leave contributor evidence absent. New
canonical enrichment omits `mentionable_users_count` from the payload and backend
handoff. It does not synthesize zero or one from an empty list or owner object.
The compatibility argument to `to_osiris_payload` remains accepted but does not
produce a contributor count from a caller's unprovenanced list.

## Scoring and compatibility

With no recent commit observations, activity now equals push recency:
`exp(-pushed_days_ago * ln(2) / 30)`, rounded to four decimals. Today is 1.0,
thirty days is 0.5, and the existing unknown-age default rounds to zero. Existing
commit-based activity calculations are unchanged. No ranking weights, training
artifacts or additional collection budgets are introduced. This changes fallback
activity values; quality improvement has not been measured.

Refresh recomputes activity when push age changes, using server-owned derivation.
A patch to the deprecated mentionable count alone cannot overwrite an activity
score previously derived from richer commit evidence. Old derived-score echoes
must match the new effective formula when its dependencies change; inconsistent
values are rejected by the existing validation contract.

The optional legacy API fields and integer Qdrant payload shape remain accepted.
Their default zero is a compatibility placeholder for missing evidence, not a
measured contributor population. Historical payloads are not automatically
rewritten. New indexing or a qualifying observation refresh applies the activity
policy; no backfill is run by this change.

Reserved embedding text/identities remain unchanged, including their legacy
`Contributor signal: mentionable_users_count=...` line. That frozen label does
not establish contributor provenance or reputation. New canonical enrichment
leaves the value absent, so the existing representation uses its neutral legacy
placeholder. Removing dynamic numeric metadata text is separately owned by #25;
this change neither adds semantic text nor silently versions/reindexes existing
vectors. Further contributor collection requires its own bounded source,
provenance/missingness contract, budget and utility validation before scoring.

## Validation scope

Local fixtures exercise real enrichment and backend projection without a network
client, legacy-counter-independent activity values, and real V2 refresh/Qdrant
behavior. The cross-repository handoff fixture keeps explicit caller-supplied
legacy counts to verify compatibility; it labels that injection separately from
canonical enrichment. No live GitHub data, real-person reputation experiment,
production migration or deployment is required or claimed.
