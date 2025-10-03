# The Ultimate goal

The ultimate goal is to extend the current code written in `regen_second_order.py` to be able to successfully derive the second order regulation proposals.

In the current code context, we were given a hotspot cell (traffic volume plus time window), and we managed to derive a set of regulation proposals thanks to the `regen` module.

In the extension, we would like to extend the current code. Particularly, after receiving the first set of regulation proposals, we will pick the best regulation plan, apply this regulation plan using `DFRegulationPlan` 

I would like to plan to extend the current code written in `regen_second_order.py` in the following ways:

