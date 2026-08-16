##############################################################################
#####                                                                    #####
#####     SCRIPT 11 — ECA ATTENTION CNN (ABLATION)                       #####
#####             (Uses ids_pipeline & ablation_base.py)                 #####
#####                                                                    #####
#####   Architecture : Base CNN with Efficient Channel Attention (ECA)   #####
#####                  Wang et al., "ECA-Net: Efficient Channel          #####
#####                  Attention for Deep Convolutional Neural           #####
#####                  Networks", CVPR, 2020.                            #####
#####                                                                    #####
#####   Ablation purpose — channel attention mechanism comparison:       #####
#####     Script 7  → SECNN    SE channel attention only                 #####
#####     Script 10 → CBAM     CBAM (channel + spatial attention)        #####
#####     Script 11 → ECA      ECA (efficient channel attention) ← this  #####
#####                                                                    #####
##############################################################################

##############################################################################
# SCRIPT 11 — ECA ATTENTION CNN (Ablation)
##############################################################################
import os
import math
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab

MODEL_NAME = "ECA"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "eca")

ECA_GAMMA = 2
ECA_B     = 1

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[Script 11 — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


# ─── Model definition (preserved verbatim) ─────────────────────────────────

def _eca_kernel_size(channels):
    t = int(abs(math.log2(channels) / ECA_GAMMA + ECA_B / ECA_GAMMA))
    k = t if t % 2 == 1 else t + 1
    return max(1, k)


def _eca_block(x, name):
    C = x.shape[-1]
    k = _eca_kernel_size(C)
    gap  = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gap  = layers.Reshape((C, 1), name=f"{name}_rs1")(gap)
    attn = layers.Conv1D(1, kernel_size=k, padding="same",
                          use_bias=False, name=f"{name}_conv1d")(gap)
    attn = layers.Activation("sigmoid", name=f"{name}_sig")(attn)
    attn = layers.Reshape((1, 1, C), name=f"{name}_rs2")(attn)
    return layers.Multiply(name=f"{name}_out")([x, attn])


def _eca_conv_block(x, filters, l2_reg, dropout, name):
    reg = regularizers.l2(l2_reg)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_b1")(x)
    x = layers.ReLU(name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_b2")(x)
    x = _eca_block(x, name)
    x = layers.ReLU(name=f"{name}_r2")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sd")(x)
    return x


def build_eca_cnn(input_shape, n_classes, task,
                  filters_1=64, filters_2=128, filters_3=256,
                  dense_units_1=256, dense_units_2=128,
                  conv_dropout=0.25, dense_dropout=0.4,
                  learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")
    x = _eca_conv_block(inputs, filters_1, l2_reg, conv_dropout, "b1")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2
    x = _eca_conv_block(x, filters_2, l2_reg, conv_dropout, "b2")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2
    x = _eca_conv_block(x, filters_3, l2_reg, 0.0, "b3")
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
    ab.run_all_experiments(build_eca_cnn, DEFAULT_HP,
                           MODEL_NAME, OUTPUT_DIR,
                           "script11_master_summary.csv")