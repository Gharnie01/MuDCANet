##############################################################################
#####     SCRIPT 5 — GWO-TUNED BASE CNN                                  #####
#####     Grey Wolf Optimizer (mealpy OriginalGWO)                       #####
#####     Mirjalili et al. (2014); mealpy v3.0.3.                        #####
#####                                                                    #####
#####     All shared logic (data pipeline, CV/final training, save       #####
#####     helpers, transfer flow) lives in ids_pipeline.py               #####
#####     tuner_base.py. This script                                     #####
#####     only defines the HHO-specific search function.                 #####
#####                                                                    #####
#####     Note on pbest_val:                                             #####
#####       mealpy GWO does not track per-wolf personal bests. Final     #####
#####       population fitness is used as a diversity proxy.             #####
##############################################################################

##############################################################################
# SCRIPT 5 — GWO-TUNED BASE CNN
##############################################################################
import os
import numpy as np

from mealpy.swarm_based.GWO import OriginalGWO
from mealpy import FloatVar, Problem

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import tuner_base as tb

MODEL_NAME    = "GWO"
OUTPUT_DIR    = os.path.join(ip.BASE, "experiment_results", "gwo")
SCRIPT_NUMBER = 5

GWO_POP_SIZE = 8
GWO_EPOCHS   = 5

tb.setup_seeds()
tb.setup_output_dirs(OUTPUT_DIR)
print(f"[Script {SCRIPT_NUMBER} — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


def run_gwo_search(fold_data):
    """mealpy OriginalGWO over the 4-D HP_SPACE. mealpy GWO does not
    track per-wolf personal bests; final population fitness used as
    diversity proxy.
    """
    all_scores = []

    class _GWOProblem(Problem):
        def __init__(self, bounds, minmax):
            super().__init__(bounds, minmax, log_to=None)
        def obj_func(self, position):
            cfg   = tb.decode_position(position)
            score = tb.hp_objective(cfg, fold_data)
            all_scores.append(score)
            return -score

    dim     = len(tb.HP_SPACE)
    bounds  = FloatVar(lb=[0.0]*dim, ub=[1.0]*dim)
    problem = _GWOProblem(bounds=bounds, minmax="min")

    print(f"\n  GWO: {GWO_POP_SIZE} wolves x {GWO_EPOCHS} epochs "
          f"= ~{GWO_POP_SIZE*GWO_EPOCHS} evaluations...")
    gwo = OriginalGWO(epoch=GWO_EPOCHS, pop_size=GWO_POP_SIZE)
    gwo.solve(problem, seed=tb.RANDOM_STATE)

    iter_scores = [-v for v in gwo.history.list_global_best_fit]
    pbest_val   = np.array([-agent.target.fitness for agent in gwo.pop])
    best_cfg    = tb.decode_position(gwo.g_best.solution)
    best_f1     = -gwo.g_best.target.fitness

    print(f"  GWO complete. Best F1: {best_f1:.4f}")
    print(f"  Best config: {best_cfg}")
    return best_cfg, iter_scores, all_scores, pbest_val


if __name__ == "__main__":
    tb.run_all_experiments(MODEL_NAME, OUTPUT_DIR, run_gwo_search, SCRIPT_NUMBER)