I would like to plan in detail, in the context of this project, to implement a Tabu search algorithm to find a network plan. 

# Preambles

- A **network plan** is a set that consists of multiple **regulations**. 

- A **hotspot** is a *traffic volume* (in the context of this project, it is synonymous to an *airspace sector*). We will use these terms interchangeably.

- A **regulation**, in essence, is a *delay instruction* consisting of three components: `<REFERENCE LOCATION> <TARGETED FLIGHTS LIST> <RATE>`.

    - `<REFERENCE LOCATION>` is often a traffic volume (it may or may not coincide with the hotspots, but usally coincide).

    - `<TARGETED FLIGHTS LIST>` is a list of flights that will be included in the queue. We call this list of flights **a flow** (note that it is a separate concept from a **cluster of trajectory**, unlike in common understandings. In other words, a **flow** is NOT similar to a cluster of trajectory, though they can be related in operations, but conceptually, they are two distinct concepts).

    - `<RATE>` is the number of flights per **hour** that the server (the air traffic controllers) will be able to serve for that particular list of flights.

- When a regulation is activated, it will setup a **First Come First Serve** (FCFS) queue at the `<REFERENCE LOCATION>`, with the rate of `<RATE>`. 

# Desired Heuristics

Preliminary theoretical work revealed that there are 4 hints to create a regulation that **minimizes the risk for inducing secondary hotspots** (in decreasing order of importance):

1. Flights going through many hotspots (overloaded traffic volumes) at once. These flights will have multiplicity effect in downregulating overloads in the network.

2. Flights sharing similar "traffic volume footprints" (i.e., passing through similar set of traffic volumes, not necessarily at the same time). This is to limit the "spatial spread" of the problem. This should be a rather soft constraint, e.g., with Jaccard distance. I am not sure how to achieve this effectively, perhaps with the *rich got richer process*?

3. Flights that, for each traffic volume it passes through, have a "valley" of slack (spare) capacity. It usually means that these flights take delay hits "easier" in terms of not causing additional hotspots.

# Goals

> We design a Tabu search process to compute the **optimal plan** that attempts to restore balances between capacity and demand, while keeping the delay cost minimal (a trade-off). 

My rough idea is as follows:


We start with the most severe hotspots (a cell - traffic volume and time bin - with huge imbalance between capacity and demand), an empty plan with no regulation. We build the first regulation by first picking that hotspot traffic volume as the reference location, with no flights, and no rate. Then we gradually add flights **from the list of flights going through the hotspot cell**:

1. With high probability, selecting flights that go through multiple hotspots.

2. Through an enrichment process (I'm not sure about this yet), pick the flights with small Jaccard distance to the existing flights in the regulation's flight list.

3. Pick the flights that have large slack valleys at same traffic volumes in the footprint (since we are delaying flights, the same traffic volumes will be traveled, just at a later time) by picking the 5 percentile (approximating minimum slack) and the mean.

4. Adding another regulation for another hotspot? Consolidating the regulations??? We would not want too many regulations in the plan (fragmented plan).

Afterwards, we run the FCFS queue to obtain the delay for each flight, we get the total overload and the total delay minutes.

# Inputs

- The occupancy information of all flights (each cell - `(TV, time_bin)`) will allow quick computation of entry counts. For context, the time_bin length is 15 minutes. In there the entry and exit times are also shown:

    ```json
    "263854795": {
            "occupancy_intervals": [
                {
                    "tvtw_index": 33504,
                    "entry_time_s": 0.0,
                    "exit_time_s": 38.0
                },
                {
                    "tvtw_index": 33408,
                    "entry_time_s": 0.0,
                    "exit_time_s": 38.0
                },
                ...
            ]
        }
    ```

- 

# Instructions

- Some of the helper functions may be available in `src/project_tailwind/optimize/eval/network_evaluator.py`. You may propose plans to amend or extend it if necessary. 

- Probably we may need to perform "incremental evaluation" since a full run of NetworkEvaluation from scratch could make the move extremely slow.

- An FCFS algorithm MUST be reimplemented. Do not rely on existing C-CASA algorithm (it is faulty!).

- Please plan for this task in details. Do not write any code yet, just detailed planning based on the current state of the project.

- If you are looking for a succinct code to see what kind of data and results are available, see `src/project_tailwind/optimize/eval/test_network_eval.py`. But beware that this project is currently a melting pot of many of my previous failed attempts, please pardon me for the messiness. 

- Don't care about ALNS (it is a failed previous attempt).