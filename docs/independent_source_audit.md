# Independent source audit

Read-only analysis of `data/raw/Cards-2.xml`, `data-9.csv`, and `data-10.csv`; initially checked 2026-09-26 and updated with user clarification and supplementary files on 2026-09-27. Counts and indices below refer to the uploaded sources, before cleanup. Card row numbers are 1-based unless explicitly marked otherwise.

**Current token resolution:** the user confirmed deletion of unused `rd` and authorized replacing card120/index119's `awaken on the 3rd turn` with `awaken after 2 turns`. The clean matrix now preserves the supplied tokenization and vocabulary, reindexing only IDs above the deleted slot. Earlier findings below describe the unchanged raw sources; the final section verifies the authorized resolution.

## Source identity and structure

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| Cards-2.xml | 3,190,795 | `0bfc42b8aeff53df9249076556bc84bc1abeeb076b2ddc0d7c68308957502505` |
| data-9.csv | 594 | `68bc5f566989aa896d9ed5830676d77d83a99e4e875dee20ae49f6b580ddcfbf` |
| data-10.csv | 608 | `a5ebeffa9cb5167b7c2aaf5ccf5cb528e2a7d73873b83b98b219eaf29119670b` |

Both external CSV files have a UTF-8 BOM, contain one headerless record, and have no final newline. Decode with `utf-8-sig` for analysis while keeping the raw bytes intact.

The project identifies itself as `Cards`, exported by `Snap! 12.2.0`, project format version `2`. It has one scene (`Cards`), one stage (`Cards`), and one sprite (`Evaluator`). There are 28 scene globals, no stage-local variables, and 13 sprite-local variables. Preserve variable scope: `V` and `v` are distinct sprite variables.

There are six scene custom block definitions, nine stage script stacks, and nine sprite script stacks. Nested control scripts bring the total XML `<script>` count to 36. The project also contains a thumbnail, stage pentrails, one costume, watcher state, three comments, and application metadata; a variables-only export is not a complete project archive. There are no duplicate XML `id` attribute values.

## Variable dimensions

The counts below parse `struct="atomic"` list contents as CSV. Plain empty atomic lists have length zero. A blank CSV cell remains an empty string.

| Scene global | Shape or saved scalar |
| --- | --- |
| Cards | 289 |
| Card Weather | 290 |
| Weathers | 13 |
| Card Descriptions | 290 |
| Weather Multiplers | 13 |
| Support Cards (Red) | 28 |
| Support Cards (Blue) | 15 |
| Setup ID | 2 × 4 |
| Setup | 2 × 4 |
| Data | scalar `0` |
| Your Cards | 4 empty strings |
| Custom | scalar `0` |
| Their Cards | 4 empty strings |
| Their Card's Borders | 4 empty strings |
| Your Card's Borders | 4 empty strings |
| Borders | 16; first entry is the empty string |
| Border ID | 2 × 4 |
| Border | 2 × 4 |
| Card Rarities | 289 |
| Card Damage Modifiers | scalar `0` |
| Exclusion characters | 6 |
| data-5_refined | 289 |
| data-5_refined_vocab_ordered | 440 |
| data-5_refined_tokenized_2d | 289 ragged rows; 3,941 values; row lengths 5–34 |
| card_simulator_phase_registry | 373 × 17, including header |
| Support Card Descriptions (Red) | 28 |
| Support Card Descriptions (Blue) | 16 |
| Border Rarities | 16 |

| Evaluator sprite variable | Shape or saved scalar |
| --- | --- |
| d | scalar `0` |
| Card Tokens | 289 × 2 |
| Support Tokens (Red) | 28 × 2 |
| Support Tokens (Blue) | 15 × 2 |
| Tokens | 2 × 2 |
| Q, K, V, MLPs, v | each an empty list |
| L | scalar `0` |
| Rarity Tokens | 32,768 × 2 |
| Classifier | 2 |

`Rarity Tokens` has 32,768 rows because the `Tokenize Cards` block creates `2^(length(Borders)-1)` tokens, not because the source contains that many border classes. Saved `d=0` disagrees with the existing 2-wide embedding vectors. No training should resume by assuming these model settings are consistent.

## Catalog alignment, duplicates, and weather provenance

- `Cards` has no duplicate names, no leading or trailing whitespace in names, and one name containing a comma: row 216, `Shay, Heart of the Cards`. Names must be CSV parsed, not comma-split.
- `Card Descriptions` rows 289 and 290 both read `Every turn, becomes a different card.` Row 289 has named card `Glamour`; row 290 has no corresponding card name. Preserve row 290 in the raw archive and record its exclusion from a 289-card view.
- Description rows 109 and 110 are equal (`Gain 40% stats for each unique 'Friendship' card.`), but their names are different (`A0-ON1`, `AK4-ON1`). They are valid separate rows, not duplicate cards to remove.
- Blue support descriptions 15 and 16 are identical Guardian Angel text; only row 15 has a support name. The extra description is an orphan tail, not a new named support.
- `Card Weather` and `data-10.csv` both have 290 entries. The final entry is `1`, with no card row 290. It cannot establish that the earlier 289 weather values are semantically correct.
- External `data-10.csv` differs from XML `Card Weather` at 22 positions. Treat a selected source as an explicit override with provenance, never a quiet synchronization.
- External `data-9.csv` contains 289 values: `1.333333` at rows 33 (`Hell's Army`) and 38 (`Heaven's Armor`), and `1` elsewhere. Its stored decimals are not exact `4/3`. The XML `Card Damage Modifiers` variable is scalar `0`; this external list is supplemental evidence, not an exact export of that saved XML variable. **User clarification, 2026-09-27:** the canonical name is **Card Modifier**, and the modifier applies to **both HP and damage**. Preserve the old XML name only as source provenance. The user confirms both cards have **2,560 HP and 1,280 ATK**.
- `data-5.csv` has 289 descriptions and matches exactly the first 289 XML `Card Descriptions` entries.
- Saved `Setup ID` resolves exactly to saved `Setup`; saved `Border ID` resolves exactly to saved `Border` using 1-based lookup.

| Card row | Card | XML weather ID | data-10 weather ID |
| ---: | --- | ---: | ---: |
| 19 | Wind Spirit | 1 | 5 |
| 40 | ToadBoiGaming | 5 | 1 |
| 41 | Boreas | 12 | 5 |
| 43 | Loki | 1 | 12 |
| 44 | Poseidon | 13 | 1 |
| 45 | Limitless Rivals | 10 | 13 |
| 46 | Vicious | 8 | 10 |
| 47 | Typhon | 13 | 8 |
| 48 | The Curse | 1 | 13 |
| 49 | Hades | 6 | 1 |
| 50 | Judgement Day | 12 | 6 |
| 51 | Odin | 10 | 12 |
| 52 | Black Plague | 9 | 10 |
| 53 | Chaos | 1 | 9 |
| 54 | Deus Ex | 2 | 1 |
| 55 | Siegfried | 7 | 2 |
| 56 | Astraeus | 12 | 7 |
| 57 | Surtr | 8 | 12 |
| 58 | Cronus | 1 | 8 |
| 61 | Ragon | 4 | 1 |
| 62 | Serpent Mist | 3 | 4 |
| 63 | Yeti | 1 | 3 |

The apparent shift in a weather segment is evidence of a possible earlier editing mistake; it is insufficient to reconstruct true assignments automatically.

## Vocabulary/token inconsistency

All 289 token rows start with ID `1` and end with ID `2`; the vocabulary's first two entries are the mathematical sans-serif Unicode strings `𝖡𝖮𝖲` and `𝖢𝖠𝖱𝖣`. These are not ASCII `BOS` or `CARD` in a lossless export.

The token sequences use every integer ID from 1 through 441 except 250 (440 distinct IDs), while the vocabulary has only 440 entries. Row 52 (`Black Plague`) contains ID 441, which is out of bounds for a direct 1-based lookup. Direct lookup also produces wrong words, e.g. row 2's ID 320 points to `reveal` instead of `deduct`.

At the initial audit, an independently checked *repair hypothesis* was that the vocabulary lost a slot at position 250: IDs 1–249 remain unchanged, and token IDs greater than 250 resolve through vocabulary index `ID-1`. With this remapping and the tokenizer regex `\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\w\s']`, 288 of 289 token rows reproduced their refined text's token sequence exactly. The user subsequently confirmed the deleted slot was unused `rd` and authorized the clean reindex; raw vocabulary and token rows remain archived unchanged.

The remaining mismatch is row 120 (`Demon Cultivator`; `Kuchisake-onna` is row 119):

- Refined description: `stats increase by 30 % for 2 turns; awaken on the 3rd turn`
- Decoded saved tokens after the hypothetical index repair: `stats increase by 30 % for 2 turns ; awaken after 2 turns`

The tokenizer splits punctuation and alphanumeric runs (`field:` into `field :`; `d6` into `d 6`) and drops standalone apostrophes. A space-joined textual comparison without the same tokenizer overcounts mismatches. The source also has semantically lossy refinement, e.g. row 68 raw `Cards of lower rarity deal half damage.` became `cards of reduce rarity deal 1 / 2 damage`. Retain raw and refined text independently; neither should silently overwrite the other.

## Card Stats and script behavior

The `Card Stats %'ID'` block accepts a numeric value or looks up an exact card-name match; it reports `Identification error` otherwise. The numeric branch contains no explicit range or integer check. The core expression is:

```text
dmg = 5 × Weather Multiplers[Card Weather[ID]] × Card Rarities[ID]^(log(2))
report ceiling([2 × dmg, dmg])
```

All list indexing in the block is Snap's 1-based indexing. The stage has `hyperops="true"`, which matters for the list passed to `ceiling`. The intended ordinary interpretation is separately rounded HP and damage, with `log` interpreted according to Snap's monadic logarithm semantics. Do not compute HP as `2 × ceil(dmg)`: the code applies the ceiling after forming both values. The reporter's metadata says `reports="number"`, although its expression builds a list.

The original block does not reference `Card Damage Modifiers`, `Borders`, `Border Rarities`, either support list, or descriptions. The user's 2026-09-27 clarification supplies a documented correction to the original formula: **Card Modifier affects both HP and ATK**. Applying it to both unrounded expressions before their existing ceilings is consistent with this correction: the supplied modifier `1.333333`, weather multiplier `3`, and rarity `1,000,000` then produce the confirmed 2,560 HP and 1,280 ATK. These two examples alone do not distinguish every possible intermediate-rounding convention. The untouched XML remains evidence of the earlier incomplete implementation.

The remaining custom definitions are scaffolding:

- `Tokenize Cards` initializes random embeddings after interactive confirmation. It does not tokenize ability prose.
- `Initialize` initializes Q/K/V/v/MLPs and the classifier using `d` and `L`.
- `Train %'Order'` and `ON_ENTRY` have no script body.
- `Simulate` consists only of reporting an empty literal.

The space-key stage script samples eight card IDs and eight border IDs or uses the four custom card and border inputs for each team. It reshapes them into 2 × 4, asks `Outcome: 0/1 (Top Won/Bottom Won)`, and appends `[Setup ID, Border ID, answer]` to `Data`. Thus `0` means top won and `1` means bottom won. The saved `Data` value is scalar `0`, not a list of historical battle records.

A detached sprite stack transforms token IDs by filtering values >0, adding `-1` to the start and `0` to the end, then adding 2 through hyperoperations. Re-running this on the already transformed saved rows is not idempotent. Other detached stage stacks insert, append, or delete support names/descriptions and should not execute as part of data extraction.

## Embedded phase registry

The registry has 372 handler records plus one header with 17 columns. Handler IDs are sequential 0–371. Every card index 0–288 is represented, and every `card_row_1` equals `card_index_0 + 1`. Every `full_description` exactly equals that card's `data-5_refined` text. All 372 `parameters_json` fields parse to JSON objects.

Confidence values are `HIGH` (217), `MEDIUM` (79), and `REVIEW` (76). Those are source labels, not independently verified simulator confidence. A populated `transition` in every row does not establish an implemented state machine: the actual `Simulate` block is empty.

Phase labels/order are CONTINUOUS/0, ENCOUNTER/5, ENTRY/10, TURN_START/20, PRE_ACTION/30, ATTACK_DECLARE/40, PRE_HIT/50, DAMAGE_CALC/60, POST_DAMAGE_DEALT/80, POST_DAMAGE_TAKEN/90, AFTER_ATTACK/100, LETHAL_REPLACEMENT/110, DEATH/120, ON_KILL/130, TURN_END/140, and DELAYED/150. Preserve clause and handler order; sorting records by phase alone would lose the author's ordering within a card.

## Lossless migration requirements

1. Keep original bytes with hashes. XML reserialization changes whitespace, self-closing tags, quote style, and entity spelling even when the parsed tree is equivalent.
2. Decode atomic lists with CSV quoting rules. Nested lists use `<item><list>…</list></item>`; registry fields contain quoted JSON. Preserve empty atomic lists, leading empty border labels, and empty entries in four-slot inputs.
3. Keep each value's original lexical text. Automatic numeric conversion loses decimal spellings, arbitrary-precision integers, and scalar/list distinctions. Preserve floats as strings in archival representations, with explicit typed interpretations in separate derived tables.
4. Keep source row order and 1-based IDs stable. Sorting names, borders, rarities, vocabulary, or phases changes dependent references.
5. Preserve the spelling `Weather Multiplers` as the original variable name; an ergonomic derived alias must record the original name. Preserve case, apostrophes, Unicode, and user-entered typos in the source layer.
6. Preserve attributes and scopes and archive all project XML, not only active card fields. The very large embedding arrays are saved data even though their utility is uncertain.
7. Record each repair, discarded duplicate tail in a cleaned view, weather-source override, and provisional tokenizer mapping. Raw values remain recoverable.
8. Handle integer magnitude deliberately. Border rarity `100000000000000000` exceeds JavaScript's safe-integer range, even though this particular decimal happens to be representable. JSON consumed as ordinary JavaScript Numbers offers no general arbitrary-integer guarantee. Card rarities range from 2 to 100,000,000,000.

No raw source was edited during this audit.

## Supplementary token files supplied 2026-09-27

These are distinct files under `data/raw/user_2026-09-27/`; the earlier top-level `data-6.csv` is unrelated and must not be overwritten or confused with this file. Manifest and archive filenames must include the relative directory, not only the basename.

| File within dated directory | Bytes | SHA-256 | Shape |
| --- | ---: | --- | --- |
| data-6.csv | 21,588 | `1df3064bbad8c8579171ba2877e7b38465ec4ffaff32e6532413eb33c11262b8` | 289 × 32 |
| data-7.csv | 2,793 | `7bb9074be42d2b26caa2058a54bd515b286a35b144befd6d9ad23fa3f8669a9d` | 1 × 438 |

Independent comparisons confirm:

- New vocabulary `data-7.csv` exactly equals the original XML vocabulary with its first two boundary symbols removed.
- New matrix `data-6.csv` uses zero padding only at row ends. Every nonzero row exactly equals the corresponding XML token row after removing its BOS/CARD boundaries and subtracting 2 from every remaining token ID.
- Nonzero token IDs range from 1 through 439, with unused ID 248. The vocabulary has only 438 entries. Thus the same original inconsistency persists, expressed in the earlier ID namespace; it is not fixed by these latest files.
- The previous missing-slot hypothesis at saved XML ID250 translates to missing slot248 in the new files. The remaining text mismatch belongs to **Demon Cultivator, card120**, after applying the same hypothetical remapping.
- Preserve the full padded matrix as supplied, including the 32-column width and trailing zeros. A stripped matrix is a derived comparison, not a replacement for the user's file.

The audit has not changed either raw source vocabulary or token matrix. The later user-authorized canonical reindex is a separate, explicitly documented derived dataset.

## Independent importer check, 2026-09-27

An import into a temporary output directory was compared with a separate direct XML/CSV decoder. All 41 persistent variable names and complete values matched exactly, including the 32,768-row learned table and empty fields. The recursively inventoried source-file hashes matched independently computed hashes for every file under `data/raw`, including both dated supplementary CSVs. The resulting catalog retained 289 cards, 43 supports, and 372 unverified registry records; only cards 33 and 38 had nonunit canonical `card_modifier` values. This check establishes archival value parity and source inventory coverage; it does not validate combat semantics.

## Verified authorized token resolution, 2026-09-27

The user's clarification identifies deleted unused lexical ID248 as `rd` in the supplementary CSV namespace (saved XML ID250 after adding the two original special tokens). The user also explicitly chose `awaken after 2 turns` for zero-based row119, card120, Demon Cultivator. This removes the only textual disagreement with the supplied token sequences after reindexing.

Independent checks of a fresh import established:

- Every cell of the clean 289 × 32 padded matrix equals the supplied matrix under precisely `ID > 248 ? ID - 1 : ID`. Exactly **191 token occurrences** change their numeric ID. No cell contains the deleted ID248 in the supplied matrix, and all zero-padding positions are unchanged.
- All **438 lexical vocabulary strings and their order remain unchanged** from supplementary `data-7.csv`. Canonical lexical IDs are 1–438. Special IDs are PAD=0, BOS=439, and CARD=440.
- Each runtime token row contains BOS, exactly the unpadded reindexed supplied sequence, then CARD. No additional lexical token is inserted or regenerated.
- Comparing all refined descriptions with the source XML shows exactly **one changed row**, card120/index119, with exactly the authorized phrase replacement.
- All **289 refined descriptions reconstruct character for character** from token strings plus stored separators. Standalone possessive apostrophes in cards8,26,51,128,168,236,273 remain in separator sidecars, preserving the source tokenizer's omission of these apostrophes from its neural-network tokens.
- Every registry record's canonical `full_description` matches the clean refined description; `source_full_description` matches the untouched original refined source.

The original XML token mapping remains invalid when read literally with its saved compacted vocabulary. That historical diagnosis must be distinguished from the now-validated canonical token matrix. The current process imports the supplied tokens with the authorized reindex; it does **not** regenerate tokenization from the text.
