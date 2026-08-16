##############################################################################
#####                                                                    #####
#####     SCRIPT 10 — CBAM ATTENTION CNN (ABLATION)                      #####
#####             (Uses ids_pipeline & ablation_base.py)                 #####
#####                                                                    #####
#####   Architecture : Base CNN with Convolutional Block Attention       #####
#####                  Module (CBAM).                                    #####
#####                  Woo et al., "CBAM: Convolutional Block Attention  #####
#####                  Module", ECCV, 2018.                              #####
#####                                                                    #####
#####   Ablation purpose — channel attention mechanism comparison:       #####
#####                                                                    #####
#####     Script 7  → SECNN    SE channel attention only                 #####
#####     Script 10 → CBAM     CBAM (channel + spatial attention) ← this #####
#####     Script 11 → ECA      ECA (efficient channel attention)         #####
#####                                                                    #####
#####   CBAM applies two sequential attention operations:                #####
#####     1. Channel attention: uses both AvgPool AND MaxPool,           #####
#####        passed through a shared MLP, combined via sigmoid gate.     #####
#####     2. Spatial attention: pools across channels (avg + max),       #####
#####        concatenates to produce a 2D attention map via 7x7 conv     #####
#####        + sigmoid.                                                  #####
#####                                                                    #####
##############################################################################

##############################################################################
# SCRIPT 10 — CBAM ATTENTION CNN (Ablation)
##############################################################################
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab

MODEL_NAME = "CBAM"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "cbam")

CBAM_RATIO     = 8
CBAM_SPATIAL_K = 7

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[Script 10 — {MODEL_NAME}] Output root: {OUTPUT_DIR}")


# ─── Model definition (preserved verbatim) ─────────────────────────────────

def _cbam_channel_attention(x, ratio, name):
    f = x.shape[-1]
    shared_fc1 = layers.Dense(max(1, f // ratio), activation="relu",
                               name=f"{name}_ca_fc1")
    shared_fc2 = layers.Dense(f, name=f"{name}_ca_fc2")
    avg = layers.GlobalAveragePooling2D(name=f"{name}_ca_avg")(x)
    avg = shared_fc1(avg); avg = shared_fc2(avg)
    mx  = layers.GlobalMaxPooling2D(name=f"{name}_ca_max")(x)
    mx  = shared_fc1(mx); mx = shared_fc2(mx)
    ca  = layers.Add(name=f"{name}_ca_add")([avg, mx])
    ca  = layers.Activation("sigmoid", name=f"{name}_ca_sig")(ca)
    ca  = layers.Reshape((1, 1, f), name=f"{name}_ca_rs")(ca)
    return layers.Multiply(name=f"{name}_ca_out")([x, ca])


def _cbam_spatial_attention(x, kernel_size, name):
    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f"{name}_sa_avg")(x)
    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f"{name}_sa_max")(x)
    concat = layers.Concatenate(name=f"{name}_sa_cat")([avg_pool, max_pool])
    sa     = layers.Conv2D(1, (kernel_size, kernel_size), padding="same",
                            activation="sigmoid",
                            name=f"{name}_sa_conv")(concat)
    return layers.Multiply(name=f"{name}_sa_out")([x, sa])


def _cbam_block(x, ratio, kernel_size, name):
    x = _cbam_channel_attention(x, ratio, name)
    x = _cbam_spatial_attention(x, kernel_size, name)
    return x


def _cbam_conv_block(x, filters, l2_reg, dropout, name):
    reg = regularizers.l2(l2_reg)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_b1")(x)
    x = layers.ReLU(name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_b2")(x)
    x = _cbam_block(x, CBAM_RATIO, CBAM_SPATIAL_K, name)
    x = layers.ReLU(name=f"{name}_r2")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sd")(x)
    return x


def build_cbam_cnn(input_shape, n_classes, task,
                   filters_1=64, filters_2=128, filters_3=256,
                   dense_units_1=256, dense_units_2=128,
                   conv_dropout=0.25, dense_dropout=0.4,
                   learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")
    x = _cbam_conv_block(inputs, filters_1, l2_reg, conv_dropout, "b1")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2
    x = _cbam_conv_block(x, filters_2, l2_reg, conv_dropout, "b2")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2
    x = _cbam_conv_block(x, filters_3, l2_reg, 0.0, "b3")
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
    ab.run_all_experiments(build_cbam_cnn, DEFAULT_HP,
                           MODEL_NAME, OUTPUT_DIR,
                           "script10_master_summary.csv")