"""live_astra: gpt-6-astra in the live control mode through sim.agent_modes."""
from sim import agent_modes
from sim.write_policy import MODEL

EXPECTED = "gpt-6-astra"
assert MODEL == EXPECTED, f"GAPHUNTER_MODEL is {MODEL!r}; live_astra needs {EXPECTED!r}"
AGENT = agent_modes.new_log("live")


def run(api):
    agent_modes.run_live(api, agent_modes.task_of(api), AGENT)
