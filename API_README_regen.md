# Regen Regulation Proposals API

This document describes the `/propose_regulations` endpoint, which surfaces the
parrhesia regen engine through the Tailwind API server. The wrapper reuses the
shared `FlightList`, TVTW indexer, capacities, centroids, and travel-minute
lookup that are preloaded by the server, so repeated calls avoid re-reading large
artifacts.

---

## Endpoint Summary

- **Method**: `POST`
- **Path**: `/propose_regulations`
- **Auth**: Same bearer token as other endpoints.
- **Purpose**: Generate up to *K* regulation proposals for a specific hotspot
  (traffic volume + time window) using the regen engine. Each proposal bundles
  per-flow rate cuts, objective improvements, component deltas, and rich flow
  diagnostics (including the flights impacted).

---

## Request Body

```json
{
  "traffic_volume_id": "MASB5KL",
  "time_window": "09:00-10:15",
  "top_k_regulations": 5,
  "threshold": 0.15,
  "resolution": 1.2
}
```

| Field | Type | Required | Notes |
| ----- | ---- | -------- | ----- |
| `traffic_volume_id` | string | ✔ | Hotspot TV identifier; must exist in the shared TV list. |
| `time_window` | string | ✔ | Time interval for the hotspot in `HH:MM-HH:MM` (end-exclusive) or `HH:MM:SS-HH:MM:SS`. The server converts this to TVTW bins using the global indexer, so any values that would collapse to zero bins for the configured bin size are rejected. |
| `top_k_regulations` | integer | ✖ | Optional cap on the number of proposals returned. When omitted, regen’s configured `k_proposals` is used (defaults to 6). |
| `threshold` | number | ✖ | Optional clustering similarity cutoff passed to `compute_flows`. Must be in `[0,1]`. Defaults to `0.1` when omitted. |
| `resolution` | number | ✖ | Optional Leiden resolution parameter (>0) passed to `compute_flows`. Defaults to `1.0` when omitted. |

- Times must be same-day and `end > start`. For example, with 15-minute bins a
  75-minute window produces 5 bins.
- If `top_k_regulations` is supplied it must be positive; otherwise the server
  ignores it.
- `threshold` outside `[0,1]` or `resolution <= 0` triggers `400` validation errors.

### Validation failures

Invalid TV IDs, malformed time windows, or windows that collapse to zero bins
produce `400` responses with a descriptive message. Other errors bubble up as
`500`s.

---

## Response Payload

```json
{
  "traffic_volume_id": "MASB5KL",
  "time_window": "09:00-10:15",
  "time_bin_minutes": 15,
  "top_k": 5,
  "weights": {
    "w1": 10.0,
    "w2": 10.0,
    "w3": 0.0,
    "w4": 0.0,
    "w5": 0.25,
    "w6": 0.1
  },
  "num_proposals": 2,
  "proposals": [
    {
      "hotspot": {
        "traffic_volume_id": "MASB5KL",
        "input_time_window": "09:00-10:15",
        "timebins": [36, 41]
      },
      "control_window": {
        "bins": [34, 43],
        "label": "08:30-10:45"
      },
      "objective_improvement": {
        "delta_deficit_per_hour": 5.8,
        "delta_objective_score": 2.41
      },
      "objective_components": {
        "before": {
          "J_cap": 18.6,
          "J_delay": 7.9
        },
        "after": {
          "J_cap": 12.1,
          "J_delay": 6.1
        },
        "delta": {
          "J_cap": -6.5,
          "J_delay": -1.8
        }
      },
      "flows": [
        {
          "flow_id": 1024,
          "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF"],
          "control_volume_id": "MASB6KL",
          "baseline_rate_per_hour": 18.0,
          "allowed_rate_per_hour": 12.0,
          "assigned_cut_per_hour": 6.0,
          "time_window_label": "08:30-10:45",
          "time_window_bins": [34, 42],
          "features": {
            "gH": 0.67,
            "v_tilde": 13.2,
            "rho": 0.41,
            "slack15": 3.6,
            "slack30": 4.9,
            "slack45": 5.1,
            "coverage": 0.82,
            "r0_i": 18.0,
            "xGH": 62.4,
            "DH": 30.2,
            "tGl": 34,
            "tGu": 42,
            "bins_count": 9,
            "num_flights": 27
          },
          "final_score": 23.45
        }
      ],
      "diagnostics": {
        "ranking_score": 19.88,
        "E_target": 7.5,
        "E_target_occupancy": 7.5,
        "diversity_penalty": 0.4
      }
    }
  ]
}
```

Field notes:

- `time_bin_minutes` echoes the global bin size used for conversion.
- `weights` exposes the FlowScoreWeights used to reconstruct the per-flow
  `final_score` values (`w1..w6`).
- `proposals[].hotspot.timebins` is the bin sequence derived from the input
  window. The hotspot window is the evaluation span for deficit removal.
- `control_window` describes the bins the regen engine chose for control
  regulations; this may extend beyond the input window.
- `objective_improvement` and `objective_components` summarize the predicted
  effect of applying the flow bundle (values mirror regen diagnostics).
- Each flow entry includes:
  - `flight_ids`: impacted flights (from cached `compute_flows`).
  - `baseline_rate_per_hour`, `allowed_rate_per_hour`, `assigned_cut_per_hour`:
    rates produced by regen.
  - `features`: all flow-level metrics used for scoring (gH, ṽ, slack, coverage,
    etc.).
  - `final_score`: linear combination of the feature set using the reported
    weights.
  - `time_window_bins`: control window bounds for the flow derived from regen’s
    diagnostics (`tGl`, `tGu`).
- `diagnostics` collects regen-level metadata such as the selected proposal’s
  `ranking_score`, `E_target`, and the diversity penalty applied during selection.

---

## Behavioural Notes

1. **Shared Resources** – The wrapper relies on the process-wide `AppResources`
   singleton. No additional disk reads are performed once the server has
   preloaded.
2. **Flow Extraction** – `compute_flows` is run once per request for the hotspot
   window. Its output provides flight membership per flow and is passed directly
   into regen so the feature extractor avoids recomputing demand histograms.
3. **Scoring Weights** – At present the endpoint uses the default weights from
   `parrhesia.flow_agent35.regen.config.resolve_weights(None)`. If custom weights
   are introduced in the future, surface them via the request body and update the
   response metadata accordingly.
4. **Time Window Parsing** – The server refuses windows where the end does not
   exceed the start or where the parsed window yields zero bins. Seconds are
   optional; `"09:00-10:15"` and `"09:00:00-10:15:00"` behave identically with a
   15-minute bin size.
5. **Error Handling** – Domain errors (unknown TV, invalid time window) return
   HTTP 400. Unexpected errors during flow computation or regen raise HTTP 500
   with a generic message; inspect server logs for stack traces.
6. **Clustering Controls** – `threshold` and `resolution` are forwarded to
   `compute_flows` to tune the Leiden clustering behaviour. The server validates
   them (`threshold` in `[0,1]`, `resolution > 0`) before invoking the regen
   wrapper.

---

## Example `curl`

```bash
curl -X POST "http://localhost:8000/propose_regulations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "traffic_volume_id": "MASB5KL",
        "time_window": "09:00-10:15",
        "top_k_regulations": 3,
        "threshold": 0.1,
        "resolution": 1.0
      }'
```

---

## Change Log

- **2024-09-29** – Initial publication alongside the `/propose_regulations`
  endpoint.
