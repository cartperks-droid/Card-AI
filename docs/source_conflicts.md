# Screenshot rarity discrepancies

The review export keeps the XML rarities and records the screenshot readings separately. Neither discrepancy has been resolved by the user. The generic stat columns therefore still use the XML values, with intrinsic weather and Card Modifier applied.

| Card ID | Card name in source | XML rarity | Screenshot rarity | Evidence |
| --- | --- | ---: | ---: | --- |
| 48 | The Curse | 6,660,000 | 6,660,666 | Screenshot 2, fourth row, sixth column |
| 132 | Xīfāng Báihü | 400,000,000 | 450,000,000 | Screenshot 7, third row, first column |

The Curse's screenshot label reads **6,660,666**, including a zero in the middle group. The prior working summary's reading of 6,666,666 was incorrect. Both original files remain unchanged.

The machine-readable [source_conflicts.json](../data/review/source_conflicts.json) includes exact source-variable locations and screenshot paths. This is a review of two known discrepancies, not an exhaustive comparison of every screenshot number.

Run `python3 -m card_engine.reports` from the project directory to regenerate:

- `data/review/catalog_overview.csv`: all 289 cards in original order, with pack, visual tags, candidate gameplay classes and their unverified status, availability evidence, stats from the generic formula, descriptions, and screenshots.
- `data/review/simulator_coverage.csv`: complete ability compilation status for each card; the current nine experimental cards and 280 unsupported cards all prohibit training labels.
- `data/review/source_conflicts.json`: the two readings above, the retained canonical values, and the dataset hash used to produce the report.

Availability inferred from brightness is recorded as an inference, not as a verified inventory. The user's confirmed Good Boy availability is distinguished from the other visual estimates. All cards remain usable as Training Dummy opponents according to the user's instruction. Class candidates also remain separate from verified gameplay membership.
