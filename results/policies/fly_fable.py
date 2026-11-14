"""fly_fable: claude-fable-5-2 in the fly control mode through sim.agent_modes."""
from sim import agent_modes
from sim.write_policy import MODEL

EXPECTED = "claude-fable-5-2"
assert MODEL == EXPECTED, f"GAPHUNTER_MODEL is {MODEL!r}; fly_fable needs {EXPECTED!r}"
AGENT = agent_modes.new_log("fly")


def run(api):
    agent_modes.run_fly(api, agent_modes.task_of(api), AGENT)
