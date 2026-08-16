##############################################################################
#####                                                                    #####
#####                    BASE CNN — STANDALONE SCRIPT                    #####
#####             (Uses ids_pipeline & ablation_base.py)                 #####
#####                                                                    #####
#####   MODEL: Base CNN (CNN)                                            #####
#####          Plain Conv2D architecture, default hyperparameters,       #####
#####          no tuning of any kind.                                    #####
#####                                                                    #####
#####                                                                    #####
##############################################################################

##############################################################################
# SCRIPT 1 — BASE CNN (Ablation)
##############################################################################
import os

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab
from baselines.optimization_baselines.tuner_base import build_base_cnn


MODEL_NAME = "CNN"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "base_cnn")

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[Script 1 — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    ab.run_all_experiments(build_base_cnn, DEFAULT_HP,
                           MODEL_NAME, OUTPUT_DIR,
                           "base_cnn_master_summary.csv")