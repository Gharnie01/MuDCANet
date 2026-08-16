##############################################################################
#####     SCRIPT 4 — PSO-TUNED BASE CNN                                  #####
#####     Particle Swarm Optimization (PySwarms GlobalBestPSO)           #####
#####     Kennedy & Eberhart (1995); PySwarms v1.3.0.                    #####
#####                                                                    #####
#####     All shared logic (data pipeline, CV/final training, save       #####
#####     helpers, transfer flow) lives in ids_pipeline.py               #####
#####     tuner_base.py. This script                                     #####
#####     only defines the HHO-specific search function.                 #####
##############################################################################

##############################################################################
# SCRIPT 4 — PSO-TUNED BASE CNN
##############################################################################
import os
import numpy as np

from pyswarms.single import GlobalBestPSO

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import tuner_base as tb

MODEL_NAME    = "PSO"
OUTPUT_DIR    = os.path.join(ip.BASE, "experiment_results", "pso")
SCRIPT_NUMBER = 4

PSO_PARTICLES  = 8
PSO_ITERATIONS = 5
PSO_W          = 0.7
PSO_C1         = 1.5
PSO_C2         = 2.0

tb.setup_seeds()
tb.setup_output_dirs(OUTPUT_DIR)
print(f"[Script {SCRIPT_NUMBER} — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


def run_pso_search(fold_data):
    """PySwarms GlobalBestPSO over the 4-D HP_SPACE."""
    dim        = len(tb.HP_SPACE)
    all_scores = []

    def pso_objective(positions):
        costs = np.zeros(len(positions))
        for i, pos in enumerate(positions):
            score    = tb.hp_objective(tb.decode_position(pos), fold_data)
            all_scores.append(score)
            costs[i] = -score
        return costs

    options = {"c1": PSO_C1, "c2": PSO_C2, "w": PSO_W}
    bounds  = (np.zeros(dim), np.ones(dim))

    print(f"\n  PSO: {PSO_PARTICLES} particles x {PSO_ITERATIONS} iterations "
          f"= {PSO_PARTICLES*PSO_ITERATIONS} evaluations...")
    optimizer = GlobalBestPSO(n_particles=PSO_PARTICLES, dimensions=dim,
                               options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(
        pso_objective, iters=PSO_ITERATIONS, verbose=False)

    iter_scores = [-c for c in optimizer.cost_history]
    pbest_val   = -optimizer.swarm.pbest_cost
    best_cfg    = tb.decode_position(best_pos)
    best_f1     = -best_cost

    print(f"  PSO complete. Best F1: {best_f1:.4f}")
    print(f"  Best config: {best_cfg}")
    return best_cfg, iter_scores, all_scores, pbest_val


if __name__ == "__main__":
    tb.run_all_experiments(MODEL_NAME, OUTPUT_DIR, run_pso_search, SCRIPT_NUMBER)