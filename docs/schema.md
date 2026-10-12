# Shared schema and interfaces (provisional)

## Statement record
| field | type | notes |
|---|---|---|
| statement_id | str | unique |
| event_id | str | e.g. ottawashooting |
| source_id | str | user/account that wrote it |
| text | str | cleaned text |
| raw_text | str | original text |
| timestamp | datetime | |
| thread_id | str, optional | |
| gold_* | optional | labels, never used by unsupervised steps |

## Scorer (Divit)
Input: two statements (ids or texts). Batch version takes a list of pairs.
Output per pair:
```json
{"label": "entailment|neutral|contradiction",
 "p_entail": 0.0, "p_neutral": 0.0, "p_contra": 0.0}
```
Probabilities sum to 1.

## Clusterer (Apoorv)
Input: list of statements.
Output: `{statement_id: cluster_id}`; cluster_id = -1 means noise/unassigned.