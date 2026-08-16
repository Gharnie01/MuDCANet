##############################################################################
#####     SCRIPT 3 — HHO-TUNED BASE CNN                                  #####
#####     Harris Hawks Optimization (mealpy OriginalHHO)                 #####
#####     Heidari et al., Future Generation Computer Systems, 2019.      #####
#####                                                                    #####
#####     All shared logic (data pipeline, CV/final training, save       #####
#####     helpers, transfer flow) lives in ids_pipeline.py               #####
#####     tuner_base.py. This script                                     #####
#####     only defines the HHO-specific search function.                 #####
##############################################################################

##############################################################################
# SCRIPT 3 — HHO-TUNED BASE CNN
##############################################################################
import os
import numpy as np

from mealpy.swarm_based.HHO import OriginalHHO
from mealpy import FloatVar, Problem

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import tuner_base as tb

MODEL_NAME    = "HHO"
OUTPUT_DIR    = os.path.join(ip.BASE, "experiment_results", "hho")
SCRIPT_NUMBER = 3

HHO_POP_SIZE = 8
HHO_EPOCHS   = 5

tb.setup_seeds()
tb.setup_output_dirs(OUTPUT_DIR)
print(f"[Script {SCRIPT_NUMBER} — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


def run_hho_search(fold_data):
    """mealpy OriginalHHO over the 4-D HP_SPACE. Returns
    (best_cfg, iter_scores, all_scores, pbest_val).
    """
    all_scores = []

    class _HHOProblem(Problem):
        def __init__(self, bounds, minmax):
            super().__init__(bounds, minmax, log_to=None)
        def obj_func(self, position):
            cfg   = tb.decode_position(position)
            score = tb.hp_objective(cfg, fold_data)
            all_scores.append(score)
            return -score

    dim     = len(tb.HP_SPACE)
    bounds  = FloatVar(lb=[0.0]*dim, ub=[1.0]*dim)
    problem = _HHOProblem(bounds=bounds, minmax="min")

    print(f"\n  HHO: {HHO_POP_SIZE} hawks x {HHO_EPOCHS} iterations "
          f"= {HHO_POP_SIZE*HHO_EPOCHS} evaluations...")
    hho = OriginalHHO(epoch=HHO_EPOCHS, pop_size=HHO_POP_SIZE)
    hho.solve(problem, seed=tb.RANDOM_STATE)

    iter_scores = [-v for v in hho.history.list_global_best_fit]
    pbest_val   = np.array([-agent.target.fitness for agent in hho.pop])
    best_cfg    = tb.decode_position(hho.g_best.solution)
    best_f1     = -hho.g_best.target.fitness

    print(f"  HHO complete. Best F1: {best_f1:.4f}")
    print(f"  Best config: {best_cfg}")
    return best_cfg, iter_scores, all_scores, pbest_val


if __name__ == "__main__":
    tb.run_all_experiments(MODEL_NAME, OUTPUT_DIR, run_hho_search, SCRIPT_NUMBER)