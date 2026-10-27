"""Chaos synchronization via reservoir computing — a reproduction of
Nazerian, Nathe, Hart & Sorrentino, "Synchronizing chaos using reservoir
computing", Chaos 33, 103121 (2023).

Package: src
  - systems.py   : drive/response vector fields (Chen, Roessler) + coupling terms
  - reservoir.py : discrete-time reservoir computer (train / estimate / errors)
  - integration.py: coupled modified 4th-order Runge-Kutta (Algorithm 1)
  - observer.py  : high-level train-and-control workflow + sync errors
"""