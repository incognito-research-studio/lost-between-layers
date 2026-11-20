"""Reference implementation of the DSOI six-layer reference architecture.

Module map (paper section -> module):
  Sec. III Defs 1-4, Lemma 1  -> types, store, canon
  Sec. III Defs 5-8           -> context, constraints, layers.l5_gate, layers.l6_audit
  Sec. IV properties          -> properties
  Sec. V-VI propositions      -> tests/propositions (executable checks)
  Sec. VII layers L1-L6       -> layers.*, pipeline, registry
  Sec. VIII scenarios S1-S7   -> scenarios/, tests/scenarios
  Simulation environment      -> sim
"""

__version__ = "0.1.0"
