The goal is to design a Monte Carlo Tree Search algorithm adapted to the context of Air Traffic Flow Management (ATFM) regulation design. 

# Definitions

A *regulation plan* consists of one or many regulations. Each *regulation* is a tuple of: control traffic volume, an active time window (e.g., 9:00-10:15), an associated flight list (a flow), and the (hourly) entry rate.

# Instructions

- Based on how the regulation plan (or an individual regulation) is input in `base_evaluation.py` and `automatic_rate_adjustment.py`, design the `DFRegulation` and `DFRegulationPlan` classes. The `DFRegulationPlan` is only a wrapper for many plans in the `DFRegulation`, along with some useful methods such as `number_of_regulations`, `number_of_flows`, `number_of_flights_affected` and a general `metadata` field which is left `None` for the moment.

- Implement these two classes in `src/parrhesia/actions`. 

