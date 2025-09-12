# MetaOpt Documentation

- Overview: `docs/metaopt_overview.md`
- API Reference: `docs/metaopt_api.md`
- Examples: `docs/metaopt_examples.md`

This documentation covers the feature engineering pipeline (`src/parrhesia/metaopt`) used to rank and group flows for regulation planning around hotspots.

Prerequisites
- A `FlightList` with occupancy intervals and basic TVTW indexing.
- A `TVTWIndexer` defining time binning.
- Per‑TV hourly capacities parsed from GeoJSON (see `parrhesia/optim/capacity.py`).
- A mapping of nominal travel minutes between TVs.

Quick start
- Build capacities and caches.
- Build travel offsets and per‑flow activity series.
- Compute per‑flow prices and pairwise features.
- Cluster flows into groups and assemble regulation proposals.

See the linked documents for detailed guidance and code snippets.

