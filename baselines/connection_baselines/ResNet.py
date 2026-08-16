##############################################################################
#####                                                                    #####
#####     SCRIPT 6 OF 7 — RESIDUAL-ONLY CNN (ABLATION)                   #####
#####             (Uses ids_pipeline & ablation_base.py)                 #####
#####                                                                    #####
#####   Architecture : Base CNN with Residual Skip Connections ONLY      #####
#####                  Isolates the contribution of residual connections #####
#####                                                                    #####
#####   Ablation purpose:                                                #####
#####     Connection mechanism comparison:                               #####
#####                                                                    #####   
#####     Script 6  → ResCNN       residual skip connections  ← this     #####
#####     Script 8  → HWNet        highway connections                   #####
#####     Script 9  → DenseCNN     dense connections                     #####
#####                                                                    #####
#####                                                                    #####
##############################################################################

##############################################################################
# SCRIPT 6 — RESIDUAL-ONLY CNN (Ablation)
##############################################################################
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab

MODEL_NAME = "ResCNN"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "residual_only")

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    se_ratio=8,          # accepted, unused (no SE blocks here)
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[Script 6 — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


# ─── Model definition (preserved verbatim) ─────────────────────────────────

def _residual_block(x, filters, l2_reg, dropout, name):
    reg      = regularizers.l2(l2_reg)
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_b1")(x)
    x = layers.ReLU(name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_b2")(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same",
                                 kernel_regularizer=reg,
                                 name=f"{name}_pr")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_pb")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_r2")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sd")(x)
    return x


def build_rescnn(input_shape, n_classes, task,
                 filters_1=64, filters_2=128, filters_3=256,
                 dense_units_1=256, dense_units_2=128,
                 conv_dropout=0.25, dense_dropout=0.4,
                 learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")
    x = _residual_block(inputs, filters_1, l2_reg, conv_dropout, "b1")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2
    x = _residual_block(x, filters_2, l2_reg, conv_dropout, "b2")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2
    x = _residual_block(x, filters_3, l2_reg, 0.0, "b3")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units_1,
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense1")(x)
    x = layers.BatchNormalization(name="bn_d1")(x)
    x = layers.ReLU(name="relu_d1")(x)
    x = layers.Dropout(dense_dropout, name="drop_d1")(x)
    x = layers.Dense(dense_units_2,
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense2")(x)
    x = layers.BatchNormalization(name="bn_d2")(x)
    x = layers.ReLU(name="relu_d2")(x)
    x = layers.Dropout(dense_dropout, name="drop_d2")(x)
    outputs = (layers.Dense(1, activation="sigmoid", name="output")(x)
               if task == "binary" else
               layers.Dense(n_classes, activation="softmax", name="output")(x))
    model = keras.Model(inputs, outputs)
    opt   = keras.optimizers.Adam(learning_rate=learning_rate)
    loss  = "binary_crossentropy" if task == "binary" else "categorical_crossentropy"
    model.compile(optimizer=opt, loss=loss,
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


if __name__ == "__main__":
    ab.run_all_experiments(build_rescnn, DEFAULT_HP,
                           MODEL_NAME, OUTPUT_DIR,
                           "script6_master_summary.csv")