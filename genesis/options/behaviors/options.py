from genesis.options.options import Options


class BehaviorOptions(Options):
    """Base class of the options declaring one behavior of an entity.

    Each behavior type has its own options class deriving from this one, given in the 'behaviors' of rigid or kinematic
    entity options under the name the behavior is looked up by ('entity.get_behavior'). The runtime class simulating it
    registers itself by deriving from 'genesis.engine.behaviors.Behavior' parameterized with that options class, e.g.
    'class LatchBehavior(Behavior[Latch])'. A saved scene names a behavior type by the name of its options class, so
    that name is unique among behavior types.
    """
