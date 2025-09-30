# Goals and Context

The goal is to build a regulation proposal engine (here by called *regen*) as part of an Air Traffic Flow Management (ATFM) optimization framework. Each regulation consists of a *control* traffic volume, a time window, one or several "flows" (sets of flights), and the allowed hourly rate imposed on each flow. The slot allocation algorithm will then find the concrete delay to be imposed on flights (but the slot allocation is not part of what we are doing).

The `regen` module will receive:

- The current `HotspotInventory`, which basically gives the congestion picture, as well as returning the "traffic flows" that we will use for the regulation proposal.

- A `k_proposals` number, which tells the module how many proposals the caller wants.

- A set of weights to weigh the features of each flow produced by `docs/flow_features_extractor.md` (code in `src/parrhesia/metaopt/per_flow_features.py`).

The `regen` module will propose `k_proposals` regulation options to the caller, along with the objective change (improvement).

# Methodology

> This is the part where I need you to reason and propose an approach that is:
> 1. Not too complicated, try to go for something simple.
> 2. Fast, since `regen` is part of another optimization framework.

## Approaches

1. Reasoned from the features computed in `per_flow_features.py` module to propose a heuristic way to pick flows . In that module, essentially we have the features for two objectives: (1) to reduce overload, and (2) to best ensure that by delaying that flow, problems do not crop up somewhere else. 
2. To pick the rates: the idea is first we see what are the maximum exceedances from all the hotspots that our flow is involved in. For example, if the exceedance is 14 flights, the heuristic is that we will need to cut 14 flights. If there are three flows, we could do something like 4 + 6 + 4 (perhaps depending on the features). We then perform a local search over the nearby rates within some limited budget allowance (I'm not sure about the heuristic rate to try yet :( )

Since we are dealing with **rolling-hour occupancy count**, but the "rate cut" is over number of flights (or entrances), I think we may need some sort of "quick and dirty conversion rules": What about computing a "dwelling ratio"?