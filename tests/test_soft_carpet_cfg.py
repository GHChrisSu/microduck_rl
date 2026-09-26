"""Soft-carpet ground DR (branch soft_carpet, 2026-09). CPU only."""
import re

import mujoco
import pytest

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as vs



@pytest.mark.parametrize("robot_cfg", [MICRODUCK_ALLCOLLISIONS_ROBOT_CFG, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG])
def test_foot_pattern_hits_exactly_the_priority_feet(robot_cfg):
    """The carpet softness must reach the feet: they have contact priority, so
    the floor's solref is ignored for foot-floor contacts."""
    spec = robot_cfg.spec_fn()
    for c in robot_cfg.collisions:
        c.edit_spec(spec)
    m = spec.compile()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    hits = [g for g, n in enumerate(names) if re.search(r"(^|/)(left|right)_foot_collision$", n)]
    assert len(hits) == 2
    prio = {g for g in range(m.ngeom) if m.geom_priority[g] > 0 and m.geom_contype[g] | m.geom_conaffinity[g]}
    assert prio == set(hits), "every priority collision geom must be carpet-softened, and only the feet"


@pytest.mark.parametrize("rough", [False, True])
def test_carpet_event_and_curriculum_registered(rough):
    cfg = vs.make_microduck_velstand_env_cfg(play=False, rough=rough)
    assert cfg.events["ground_softness"].mode == "reset"
    assert cfg.events["ground_softness"].func is microduck_mdp.randomize_ground_softness
    assert cfg.curriculum["carpet_severity"].params["event_name"] == "ground_softness"


def test_carpet_stages_monotone_and_keep_hard_floor():
    st = vs.CARPET_RAMP_STAGES
    steps = [s["step"] for s in st]
    assert steps == sorted(steps) and steps[0] == 0
    for key in ("timeconst_range", "width_range", "margin_range"):
        his = [s["params"][key][1] for s in st]
        assert his == sorted(his), key
    for s in st:
        p = s["params"]
        assert 0.0 < p["carpet_prob"] < 1.0  # hard floor stays in-distribution
        assert p["timeconst_range"][0] >= 0.02 and p["margin_range"][0] == 0.0


def test_play_uses_hardest_carpet_stage():
    cfg = vs.make_microduck_velstand_env_cfg(play=True)
    assert cfg.events["ground_softness"].params["margin_range"] == vs.CARPET_RAMP_STAGES[-1]["params"]["margin_range"]
