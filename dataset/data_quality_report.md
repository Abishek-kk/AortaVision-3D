# Data quality report

Rows: 378
Columns: 59
Exact duplicates: 0
Missing values: 0

## Cleaning actions
- Sex: Fmale -> Female (unambiguous category spelling correction)
- VHD: mild -> Mild (unambiguous category casing correction)

## Removed model-input features
- Exertional CP: constant feature in current training data

## Cath/vessel inconsistency
- Rows found: 1
- Decision: retained. CAD is derived from Cath by project definition; vessel labels are retained for vessel-specific targets. No correction was invented.

## TG > 1000
- Rows found: 1
- Decision: retained as physiologically possible but extreme; robust scaling is used for numeric preprocessing. No target information was used.
