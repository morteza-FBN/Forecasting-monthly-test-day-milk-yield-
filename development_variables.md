# Development-file variables

The restricted development file is a lactation-level file. Each row represents one retained or candidate lactation curve.

Core variables required by the public workflow:

- `cow_id`: deidentified cow identifier used for grouped validation.
- `parity`: lactation parity.
- `milk_1` to `milk_9`: monthly milk-yield records, kg/d.
- `dim_1` to `dim_9`: days in milk corresponding to `milk_1` to `milk_9`.
- optional prefix composition variables: fat percentage, protein percentage, and SCC within records 1 to k.
- optional cow-management and retrospective covariates as described in the manuscript.

Raw cow IDs and commercial-farm identifiers should not be committed to the public repository.
