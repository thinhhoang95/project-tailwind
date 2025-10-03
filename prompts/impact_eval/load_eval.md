# Module Name
> **Nominal count and regulation cost** (short name: `nominal`), as part of the evaluation function computation.

# High level inputs - outputs
## Inputs
1. Original flight count (as a vector of `N` traffic-volume-time-windows (TVTWs)). This is a huge sparse matrix, called the *occupancy matrix*, one row for each flight, and one column for each TVTW. The indexing of TVTW follows the `tvtw_indexer.py` module.
2. The network plan, which is a set of *regulations* to be implemented (roughly speaking, a regulation is a list of flights to be delayed through setting the entry rate at some traffic volumes or location).
3. The horizon H (in time windows).

## Outputs
1. The nominally updated flight count as the regulations are implemented.
2. The total excess traffic (within the horizon).
3. The maximum excess traffic in any of the (within the horizon).

# Implementation Details
### How regulations work
Each *network plan* is a set of *regulations*. Each *regulation* is a triplet of `<REFERENCE LOCATION OR TRAFFIC VOLUME> <FILTERING CONDITION> <RATE> <TIME WINDOWS>`. 

### General Approach
1. From the original flight plans (for example: from `D:/project-cirrus/cases/flights_20230801.csv`), we compute the *TVTW occupancy vector*, which is one row in the *occupancy matrix*. This calculation needs to be done once only.

2. For each *regulation* in the *network plan*, we look for the set of flights that are eligible to be targetted with delay.

    Comprehensively, we use the `<REFERENCE LOCATION>` and `<FILTERING CONDITION>` to select the flights. This is a simple matching problem. 

    For example: `<LFPPLW1> <LFP* > LI*> <60> <36, 37>` means that we select the flights that cross through the traffic volume `LFPPLW1`, going from any Paris airports e.g., LFPG, LFPO; to any Italian airports e.g., LIMC; and the crossing time is in time window 36, 37 (9:00-9:30 AM) for a time window of 15 minutes. 

    In this evaluation function implementation, we will assume the regulation is given (how we arrive at it is not relevant at this point).

3. After selecting the flights, we run a variant of the *Computer Assisted Slot Allocation* (more precisely called C-CASA) on this set of flights, with the given rate, and the C-CASA machinery spits out the ground delay value for each flight (in minutes).

4. Now each flight assigned with the delays, they might either:

    a. Accept the delay, if the delay is less than 25 minutes (this is a heuristic at this moment and will be refined eventually with more complex logic, but at this moment, we are content with this heuristic).

    b. If the delay is more than 25 minutes, it will seek for a reroute to dodge the delay. This is in fact can be done through a placeholder function call for our preliminary implementation. There are two outcomes: if a rerouting solution is found, a **TVTW occupancy vector** corresponding to the rerouting solution is returned. If the rerouting solution is not found, it will accept the original delay.

5. If the flight was rerouted, the *TVTW occupancy vector* is taken directly. If the flight was delayed, we just "shift the time bin indices" for the *TVTW occupancy vector* according to the delay in minutes, then we replace the original *TVTW occupancy vector* of that flight with the "shifted" one.

**Remarks:** technically there is still a possibility that a rerouted flight might still need to be delayed. However, this will complicate our calculations because the C-CASA algorithm needs to be invoked again. As a result, we propose a first order assumption that the rerouted flight will not induce any secondary delay. This will be compensated (if necessary) in the next cycle of the control.

6. The rerouting function will also return the cost difference (additional fuel burn, crew time...). If the flight was rerouted, and there was no additional delay, 

6. We then find out the TVTW with most excess traffic (i.e., the count subtracted by capacity), during the horizon H (i.e., considering all time windows within H). This will be `z_max`. We also compute `z_sum` by summing all excess traffic across all TVTWs within the horizon H.

7. We return the new occupancy matrix, `z_max` and `z_sum`.

# Instructions
We will focus **exclusively** on the computation of TVTW excess traffic vector.

We load the traffic volumes, which is defined in `D:/project-cirrus/cases/traffic_volumes_simplified.geojson`. Here is an excerpt of the file:
```.geojson
...
"coordinates": [
            [
                4.4833333333,
                50.9
            ],
            ...
        ]
    ]
},
"properties": {
    "traffic_volume_id": "EBBUELS1",
    "name": "",
    "category": "MO",
    "airspace_id": "EBBUELS",
    "role": "G",
    "skip_in": "60",
    "skip_out": "900",
    "min_fl": 45,
    "max_fl": 185,
    "airblock_count": 23,
    "elementary_sectors": [
        "EBBUELS"
    ],
    "capacity": {
        "6:00-7:00": 23,
        "7:00-8:00": 10,
        "8:00-9:00": 17,
        "9:00-10:00": 23,
        "10:00-11:00": 15,
        "11:00-12:00": 24
    }
    },...
```

You may find in this, the capacity values set for each hour (note that this is different from the time window employed by the TVTW's time window length).

The `src/project_tailwind/impact_eval/tvtw_indexer.py` (the output json can be located in `output/tvtw_indexer.json`) module will give you the mapping between TVTW and the index of the excess traffic vector. 

The occupancy matrix is the sparse matrix where each row is a flight, and the count is 1 for each TVTW that should be counted towards, and 0 otherwise.

**Instruction:** We are going to implement the entire scaffold for overload detection:

1. Design and implement a class of `FlightList`, which loads the original (unregulated) flight list from `output/so6_occupancy_matrix_with_times.json` into a desirable format with sparse matrix indicating the occupancy vectors, and the relevant information such as entry and exit time can be rapidly retrieved.

2. Implement a function in a separate Python module called `NetworkEvalulator` class which takes the traffic volumes GDF object, the `FlightList` object, and outputs an excess flight vector, which shows 0 if the TVTW is not overloaded, and `count - capacity` for the overloaded TVTWs.

# Requirements
1. Make sure you gracefully handle the time window difference between the capacity in the traffic volumes GDF and the time windows length (it is part of the `tvtw_indexer.json`).

2. Efficiency and speed is of paramount importance. Think about the approach first before you implement.

3. Implement everything in `src/project_tailwind/optimize/eval` please.