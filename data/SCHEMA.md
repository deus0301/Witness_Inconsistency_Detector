# Shared data schema

Every module downstream of `data/` (clustering, NLI scoring, pipeline, evaluation, demo)
consumes **only** the `Claim` record defined in [`schema.py`](schema.py). A claim is one
statement made by one account about one event.

## `Claim`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `claim_id` | `str` | yes | Unique id. PHEME: `<event>:<tweet_id>`. Free text: `<event>:<account_id>:<sentence_index>`. |
| `event` | `str` | yes | Event the claim is about, e.g. `ottawashooting`. The pipeline is run per event. |
| `thread_id` | `str` | yes | PHEME conversation thread (id of its source tweet). Free text: the account id. |
| `account_id` | `str` | yes | Who made the claim (Twitter screen name, or the name given in the demo). Pairs are only compared across different accounts. |
| `text` | `str` | yes | Cleaned text (see "Cleaning"). This is what gets embedded and scored. |
| `timestamp` | `str` or `null` | no | ISO-8601 UTC time of the tweet. |
| `proxy_label` | `"true"`, `"false"`, `"unverified"` or `null` | no | **Veracity of the rumour story the tweet reports** (PHEME annotation). `null` for non-rumour tweets, replies and free text. It is a proxy used only for evaluation. It is not an NLI label and it is never an input to the detector. |
| `metadata` | `dict` | yes (may be empty) | Extra fields, listed below. Downstream code must tolerate missing keys. |

### `metadata` keys written by the PHEME loader

| Key | Type | Meaning |
|---|---|---|
| `is_source` | `bool` | `true` for a thread's source tweet, `false` for a reply. |
| `is_rumour` | `bool` | Thread was annotated as a rumour. |
| `veracity` | `str` or `null` | Same value as `proxy_label`, also set on replies of a rumour thread. |
| `rumour_category` | `str` or `null` | The rumour *story* the thread belongs to (PHEME `category`), e.g. "There were 148 people onboard the Germanwings plane". Used as the weak reference for clustering and for the proxy labels. |
| `is_turnaround` | `bool` | PHEME `is_turnaround`: the tweet resolved the story's status (for a false story, a debunk). |
| `lang` | `str` | Twitter's language code for the tweet. |
| `parent_id` | `str` or `null` | Tweet id this tweet replies to. |
| `raw_text` | `str` | Text before cleaning. |
| `n_duplicates` | `int` | How many later tweets with identical normalised text were folded into this claim. |
| `duplicate_accounts` | `list[str]` | Accounts of those folded duplicates. |

Free-text accounts (demo, `--input`) set `is_source`, `sentence_index` and `raw_text` only.

## Serialisation

`save_claims(claims, path)` / `load_claims(path)` write and read UTF-8 JSON lines, one claim
per line, with exactly the fields above. `Claim.validate()` runs on construction and raises
`ValueError` for empty required strings or an unknown `proxy_label`.

## Cleaning (`cleaning.py`)

Applied to every tweet and every pasted sentence:

1. HTML entities unescaped (`&amp;` becomes `&`), curly quotes normalised.
2. URLs removed.
3. Retweet markers (`RT @user:`, `MT @user:`) and all `@mentions` removed.
4. Hashtags keep their words and lose the `#`; CamelCase is split (`#OttawaShooting` becomes `Ottawa Shooting`).
5. Emoji and pictographs removed; whitespace collapsed.

Then, per event:

- **Language:** tweets whose Twitter `lang` is not in `data.languages` (default `[en]`) are dropped. The NLI model is English-only, so scoring non-English text would be meaningless. All source tweets of the four main events are tagged `en`; the filter matters for replies (about 2 to 5 percent are French, Spanish, Arabic, German or undetermined). Setting `languages: []` keeps everything, and `models.embedding_multilingual` is provided for that case.
- **Length:** cleaned texts with fewer than `data.min_tokens` (4) tokens are dropped.
- **Duplicates:** claims with identical normalised text (lower-case alphanumerics) are folded into the earliest one. Identical copies cannot disagree with each other and would only inflate cluster sizes.

## What PHEME actually contains (verified on the local copy)

```
<event>-all-rnr-threads/{rumours,non-rumours}/<thread_id>/
    source-tweets/<thread_id>.json   reactions/<tweet_id>.json   structure.json   annotation.json
```

- Rumour `annotation.json` keys: `is_rumour`, `category`, `misinformation`, `true`, `links`, `is_turnaround`. Non-rumour threads have `is_rumour` only.
- **The thread releases have no per-reply stance labels (support / deny / query / comment) and no entailment / contradiction labels.** Pair labels come from the separate PHEME RTE release described below.
- Veracity is constant within a rumour story (`category`): no story of the four main events mixes true and false threads.
- PHEME v1 (`data/raw/pheme_v1`) has the same folders without `annotation.json`; the loader falls back to it and then yields claims without story or veracity.

## Labelled pairs: PHEME RTE (`pheme_rte.py`)

`pheme_rte_datasets_2016/pheme_rte_datasets_2016/per_event_splits/pheme_rte_<event>_en.xml` holds
tweet pairs of the four main events as `<pair entailment="ENTAILMENT|CONTRADICTION|UNKNOWN"
id="<tweetA>.json_<tweetB>.json" rumours="..."><t>..</t><h>..</h></pair>`. `load_pheme_rte` returns
`RTEPair(event, pair_id, tweet_a, tweet_b, premise, hypothesis, label, stories)` with `UNKNOWN`
mapped to `neutral`, the `URL` placeholder removed, and duplicate ids and identical-text pairs
dropped (5,413 pairs). Entailment pairs name one rumour story, contradiction and unknown pairs
name two, so labels are assigned through stories. Tweet ids match `Claim.claim_id` after the
`<event>:` prefix. These pairs are used for evaluation only.

## NLI examples

MultiNLI is read by `multinli.py` into `NLIExample(premise, hypothesis, label, genre, pair_id)`.
Rows whose `gold_label` is `-` (no annotator consensus) are dropped. Probability arrays
everywhere in the project use the column order `(entailment, neutral, contradiction)`.
