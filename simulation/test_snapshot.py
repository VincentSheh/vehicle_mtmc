
import numpy as np
import torch
import os
import yaml
from environment import build_env_base

def test_snapshot_restoration():
    cfg_path = "configs/simulation_0.yaml"
    if not os.path.exists(cfg_path):
        print(f"Config {cfg_path} not found, skipping test.")
        return

    env = build_env_base(cfg_path)
    env.reset(seed=42)

    # Take initial snapshot
    snapshot = env._snapshot_edges()
    
    # Run some steps
    ids_cpus = [1.0] * len(env.edge_areas)
    env.step(ids_cpus)
    
    state_after_step = [edge.get_state() for edge in env.edge_areas]
    rng_after_step = np.random.get_state()
    
    # Restore initial snapshot
    env._restore_edges(snapshot)
    
    state_after_restore = [edge.get_state() for edge in env.edge_areas]
    rng_after_restore = np.random.get_state()
    
    # Compare
    initial_edge_states, initial_rng_state = snapshot
    
    for i, (initial, restored) in enumerate(zip(initial_edge_states, state_after_restore)):
        for key in initial:
            if key == "attacker_states":
                for j, (atk_init, atk_rest) in enumerate(zip(initial[key], restored[key])):
                    assert atk_init == atk_rest, f"Edge {i} Attacker {j} state mismatch for {key}: {atk_init} != {atk_rest}"
            else:
                assert initial[key] == restored[key], f"Edge {i} state mismatch for {key}: {initial[key]} != {restored[key]}"
    
    # Compare RNG states
    # np.random.get_state() returns a tuple (str, ndarray, int, int, float)
    assert initial_rng_state[0] == rng_after_restore[0]
    np.testing.assert_array_equal(initial_rng_state[1], rng_after_restore[1])
    assert initial_rng_state[2:] == rng_after_restore[2:]

    print("Snapshot and restoration test passed!")

    # Now verify that running a step again after restoration produces the same state as before
    env._restore_edges(snapshot)
    env.step(ids_cpus)
    
    state_after_second_step = [edge.get_state() for edge in env.edge_areas]
    
    for i, (first, second) in enumerate(zip(state_after_step, state_after_second_step)):
        for key in first:
             if key == "attacker_states":
                for j, (atk_f, atk_s) in enumerate(zip(first[key], second[key])):
                    assert atk_f == atk_s, f"Edge {i} Attacker {j} state mismatch after second step for {key}"
             else:
                assert first[key] == second[key], f"Edge {i} state mismatch after second step for {key}"

    print("Consistency test passed!")

if __name__ == "__main__":
    test_snapshot_restoration()
