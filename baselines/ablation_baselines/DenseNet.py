##############################################################################
#####                                                                    #####
#####     SCRIPT 9 OF 9 — DENSE CONNECTION CNN (ABLATION)                #####
#####             (Uses ids_pipeline & ablation_base.py)                 #####
#####                                                                    #####
#####   Architecture : CNN with Dense connections (DenseNet-style)       #####
#####                  Huang et al., "Densely Connected Convolutional    #####
#####                  Networks", CVPR, 2017.                            #####
#####                                                                    #####
#####                    Connection mechanism comparison:                #####
#####     Script 6  → ResCNN       residual skip connections             #####
#####     Script 8  → HWNet        highway connections                   #####
#####     Script 9  → DenseCNN     dense connections          ← this     #####
#####                                                                    #####
#####   Ablation purpose:                                                #####
#####     Isolates the contribution of dense connections independently   #####
#####                                                                    #####
#####   Model name   : DenseCNN                                          #####
#####   Datasets    : CICIDS2017 | UNSW-NB15 (independent)               #####
#####   Tasks       : binary | multiclass                                #####
#####   Output dir  : experiment_results/dense/                          #####
#####                                                                    #####
##############################################################################

##############################################################################
# SCRIPT 9 — DENSE CONNECTION CNN (Ablation)
##############################################################################
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab

MODEL_NAME = "DenseCNN"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "dense")

GROWTH_RATE            = 16
DENSE_LAYERS_PER_BLOCK = 3

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[Script 9 — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


# ─── Model definition (preserved verbatim) ─────────────────────────────────

def _dense_block(x, n_layers, growth_rate, l2_reg, dropout, name):
    reg         = regularizers.l2(l2_reg)
    feature_map = x
    all_outputs = [x]
    for i in range(n_layers):
        lname = f"{name}_L{i+1}"
        feature_map = (layers.Concatenate(name=f"{lname}_cat")(all_outputs)
                       if len(all_outputs) > 1 else all_outputs[0])
        feature_map = layers.BatchNormalization(name=f"{lname}_bn1")(feature_map)
        feature_map = layers.ReLU(name=f"{lname}_r1")(feature_map)
        feature_map = layers.Conv2D(4 * growth_rate, (1, 1), padding="same",
                                    kernel_regularizer=reg,
                                    name=f"{lname}_bot")(feature_map)
        feature_map = layers.BatchNormalization(name=f"{lname}_bn2")(feature_map)
        feature_map = layers.ReLU(name=f"{lname}_r2")(feature_map)
        feature_map = layers.Conv2D(growth_rate, (3, 3), padding="same",
                                    kernel_regularizer=reg,
                                    name=f"{lname}_c")(feature_map)
        all_outputs.append(feature_map)
    return layers.Concatenate(name=f"{name}_out")(all_outputs)


def _transition_layer(x, out_filters, l2_reg, dropout, name):
    reg = regularizers.l2(l2_reg)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_r")(x)
    x = layers.Conv2D(out_filters, (1, 1), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sd")(x)
    return x


def build_densecnn(input_shape, n_classes, task,
                   filters_1=64, filters_2=128, filters_3=256,
                   dense_units_1=256, dense_units_2=128,
                   conv_dropout=0.25, dense_dropout=0.4,
                   learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(filters_1, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name="init_conv")(inputs)
    x = layers.BatchNormalization(name="init_bn")(x)
    x = layers.ReLU(name="init_r")(x)
    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, 0.0, "db1")
    x = _transition_layer(x, filters_1, l2_reg, conv_dropout, "tr1")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2
    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, 0.0, "db2")
    x = _transition_layer(x, filters_2, l2_reg, conv_dropout, "tr2")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2
    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, 0.0, "db3")
    x = _transition_layer(x, filters_3, l2_reg, 0.0, "tr3")
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
    ab.run_all_experiments(build_densecnn, DEFAULT_HP,
                           MODEL_NAME, OUTPUT_DIR,
                           "script9_master_summary.csv")