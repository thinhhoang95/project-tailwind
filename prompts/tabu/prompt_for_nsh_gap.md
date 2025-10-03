Can you help me extend the `flight_features.py` to include a more mathematically correct scoring component from Slack condition.

# Context

The starting point for the implementation is the computation of the slack valley metrics in `flight_features.py`, and we write some code to compute the `NSH_score` as follows:


Let the *(t, L) - NSH score component* defined at a reference time $t$ and window length $L$ as:

$$ J(t, L) = \frac{1}{L} \sum_{\tau = t}^{t + L - 1} \bigg( x_s(\tau) + \min_{s \in S_k} sl_s (\tau + \xi_{k, s}) \bigg) $$

Where $x_s (t) $ is the occupancy count at time t at the hotspot traffic volume, and $sl_s(\tau + \xi_{k,s})$ is the slack value (defined as `capacity - occupancy`) of traffic volume s at time $\tau + \xi_{k, s}$. Note that $\tau_{k,s}$ is the travel time from the hotspot (the regulation location) to the traffic volume $s$. 

In a nutshell, the second term of the sum is the minimum value of the slack among the "traffic volume footprint", i.e., a snapshot of the slack state of each traffic volume where the flight travel through.

The current code already showed how to retrieve these slack values for each flight.

Then the final NSH score is:
$$ J = \max_{t, L} J(t, L) $$

# Simplified Version

To support the optimization, we approximate $x_s(\tau) \approx 0$ and fix $L=4$ (but add L as an argument default to 4 instead), limit the range of $t$ from 1 to 8. Then our task simplifies to computing:

$$ J = \max_{t \in [1, 8]} \min_{s \in S_k} sl_s (\tau + \xi_{k, s}) $$

i.e., finding the best minimum slack among the flight's footprint. This indicates the flight with strong potential to take a "delay hit" without causing secondary hotspot. 

# Requirements
- First verify whether this request is sound and logic.
- Vectorization is important to maintain high performance.
- Leave the existing code for the slack valley computation intact.
