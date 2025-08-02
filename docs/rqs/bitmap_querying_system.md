# Route Query System Documentation

This document provides a detailed explanation of the `RouteQuerySystem` class, its underlying algorithms, data structures, and usage. The system is designed for efficient querying of aviation route data, leveraging bitmaps for high-performance set operations.

## 1. Overview

The `RouteQuerySystem` is a Python class that builds an in-memory database of aviation routes from a JSON file. It creates several indexes to allow for fast retrieval of routes based on various criteria:

-   **Origin-Destination (OD) pairs:** Find all routes between a specific origin and destination.
-   **Traffic Volume Time Windows (TVTWs):** Find routes that either contain or avoid specific TVTWs, which represent occupancy in a discretized 4D space (3D space + time).
-   **Combined Queries:** Combine OD and TVTW criteria to perform complex filtering (e.g., find all routes from A to B that avoid a set of conflicted TVTWs).

The core of its efficiency lies in the use of `pyroaring`'s `BitMap` data structure. By representing sets of routes as bitmaps, complex logical operations like intersection, union, and difference can be performed extremely quickly.

## 2. Core Concepts & Algorithm

### 2.1. Data Ingestion and ID Assignment

The system starts by reading a JSON file containing the route data. The JSON file is expected to be a dictionary where keys are route strings (e.g., `"BKPR ALELU EDDM"`) and values are objects containing at least the route's `distance` and its `impact_vectors` (a list of TVTW IDs).

To enable efficient indexing, each unique route string is assigned a unique integer ID, from `0` to `n-1`, where `n` is the total number of routes. This allows us to reference routes in our data structures using simple integers instead of strings.

-   `routes`: A list of all route strings. The index of a route in this list is its ID.
-   `route_str_to_id`: A dictionary mapping the route string to its integer ID for quick lookups.
-   `route_id_to_str`: A list where the index is the route ID and the value is the route string.

### 2.2. Impact Vector Storage

A key piece of data for each route is its "impact vector," which is a list of integers representing the TVTWs it passes through. To store these efficiently, all vectors for all routes are concatenated into a single, contiguous NumPy array called `vec_data`.

To locate the specific vector for a given route ID, an offset array `vec_off` is used. `vec_off[i]` stores the starting index in `vec_data` for the vector of route `i`. The vector for route `i` is therefore located at `vec_data[vec_off[i]:vec_off[i+1]]`. This storage method is highly memory-efficient and fast.

### 2.3. Indexing with Roaring Bitmaps

The power of the system comes from its indexes, which use `BitMap`s to store posting lists. A posting list is a list of document IDs (in our case, route IDs) that contain a certain term.

#### a. Origin-Destination (OD) Index

-   **Structure:** `od_index: Dict[Tuple[str, str], BitMap]`
-   **Purpose:** To quickly find all routes for a given origin-destination pair.
-   **How it's built:** The system iterates through every route. For each route, it parses the origin and destination from the route string. It then adds the route's ID to the `BitMap` associated with the `(origin, destination)` tuple in the `od_index` dictionary.

#### b. TVTW Index (Inverted Index)

-   **Structure:** `tvtw_index: Dict[int, BitMap]`
-   **Purpose:** To quickly find all routes that pass through a specific TVTW. This is a classic inverted index.
-   **How it's built:** The system iterates through every route and its corresponding impact vector. For each TVTW ID in the vector, it adds the route's ID to the `BitMap` associated with that TVTW ID in the `tvtw_index` dictionary.

## 3. Implementation Details

The class is implemented with a focus on performance and memory efficiency.

-   **Initialization (`__init__` & `_build_indexes`):** The constructor takes the path to the route data file and immediately calls `_build_indexes`. This method performs the one-time setup of all data structures and indexes. It uses `orjson` for fast JSON parsing.
-   **Data Structures:** NumPy arrays are used for numerical data (`distances`, `vec_off`, `vec_data`) due to their memory efficiency and performance. Dictionaries and `defaultdict` are used for the primary index structures.
-   **Query Methods:** The query methods leverage the pre-built indexes to answer questions. The logic typically involves:
    1.  Retrieving one or more `BitMap`s from the indexes.
    2.  Performing bitmap operations (`&` for intersection, `|` for union, `-` for difference).
    3.  Formatting the final `BitMap` of route IDs into a user-friendly list of `(route_string, distance)` tuples.

## 4. Usage Examples

Here is how to instantiate and use the `RouteQuerySystem`.

### 4.1. Initialization

First, you need a JSON file with the route data. Assuming you have a file named `route_distances.json` in the `output` directory, you can initialize the system as follows:

```python
from project_tailwind.rqs import RouteQuerySystem

# Path to your route data file
file_path = "D:/project-tailwind/output/route_distances.json"

# Initialize the system
rqs = RouteQuerySystem(file_path)
print("System ready.")
```

### 4.2. Example Queries

Once initialized, you can perform various queries.

#### a. Get Basic Route Information

You can retrieve the impact vector and distance for a specific route.

```python
route_str = "BKPR ALELU EDDM"
vector = rqs.get_vector(route_str)
distance = rqs.get_distance(route_str)

print(f"Vector for '{route_str}': {vector}")
print(f"Distance for '{route_str}': {distance:.2f} NM")
```

#### b. Get Routes by Origin-Destination

Find all routes and their distances for a given OD pair. This uses the `od_index`.

```python
origin, dest = "BKPR", "EDJA"
routes_od = rqs.get_routes_by_OD(origin, dest)

print(f"Found {len(routes_od)} routes from {origin} to {dest}:")
for route, dist in routes_od:
    print(f"  - {route} (Distance: {dist:.2f})")
```

#### c. Get Routes Avoiding Specific TVTWs

Find all routes for an OD pair that **do not** pass through any of a given list of TVTWs. This is useful for de-confliction. The logic finds all candidate routes for the OD pair and subtracts the routes that contain any of the banned TVTWs.

```python
origin, dest = "BKPR", "EDJA"
banned_tvtws = [3461] # TVTW to avoid

routes_avoiding = rqs.get_routes_avoiding_OD(origin, dest, banned_tvtws)

print(f"Routes from {origin} to {dest} avoiding TVTWs {banned_tvtws}:")
for route, dist in routes_avoiding:
    print(f"  - {route} (Distance: {dist:.2f})")
```

#### d. Get Routes Matching TVTWs

You can also find routes that match one or more TVTWs.

**1. Matching ANY of the TVTWs (Union):**
Find OD routes that pass through at least one of the specified TVTWs.

```python
origin, dest = "BKPR", "EDDM"
tvtws_any = [1706, 3001]

routes_any = rqs.get_routes_matching_OD(origin, dest, tvtws_any, require_all=False)

print(f"Routes from {origin} to {dest} matching ANY of {tvtws_any}:")
for route, dist in routes_any:
    print(f"  - {route} (Distance: {dist:.2f})")
```

**2. Matching ALL of the TVTWs (Intersection):**
Find OD routes that pass through all of the specified TVTWs.

```python
origin, dest = "BKPR", "EDDM"
tvtws_all = [1706, 2522]

routes_all = rqs.get_routes_matching_OD(origin, dest, tvtws_all, require_all=True)

print(f"Routes from {origin} to {dest} matching ALL of {tvtws_all}:")
for route, dist in routes_all:
    print(f"  - {route} (Distance: {dist:.2f})")
```
