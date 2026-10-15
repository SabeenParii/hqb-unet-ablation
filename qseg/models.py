"""Model definitions.

Architectures are kept as in the original notebooks (filter widths unchanged), so the
models are the same networks as described in the manuscript, with three corrections:

1. Hybrid Quantum U-Net: the trainable circuit weights are now applied with general
   rotations Rot(phi, theta, omega) instead of RZ only. With RZ-only weights the circuit
   output was independent of the weights (see qseg/quantum_reference.py).
2. Two ablation variants share every layer of the hybrid model except the 4-dimensional
   bottleneck block, so any effect of the quantum circuit can be isolated:
     - "classical": the circuit is replaced by Dense(4, tanh)  (20 parameters vs 24)
     - "none":      the circuit is removed (identity)
3. All models take 1-channel grayscale input and output float32 probabilities.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------------------------------------------------------- blocks
def conv_block(x, filters, dropout=0.0, name=None):
    for k in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal",
                          name=None if name is None else f"{name}_conv{k}")(x)
        x = layers.BatchNormalization(name=None if name is None else f"{name}_bn{k}")(x)
        x = layers.Activation("relu", name=None if name is None else f"{name}_relu{k}")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=None if name is None else f"{name}_drop")(x)
    return x


def output_layer(x):
    return layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="output_mask")(x)


# ----------------------------------------------------------------------------- Classical U-Net
def build_unet(input_shape=(256, 256, 1), filters=(64, 128, 256, 512, 1024)):
    inputs = layers.Input(input_shape, name="input_image")
    skips, x = [], inputs
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, name=f"enc{i + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = conv_block(x, filters[-1], name="bottleneck")
    for i, (f, s) in enumerate(zip(reversed(filters[:-1]), reversed(skips))):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", kernel_initializer="he_normal",
                                   name=f"up{i + 1}")(x)
        x = layers.Concatenate(name=f"concat{i + 1}")([x, s])
        x = conv_block(x, f, name=f"dec{i + 1}")
    return models.Model(inputs, output_layer(x), name="Classical_UNet")


# ----------------------------------------------------------------------------- Attention U-Net
def attention_gate(g, x, filters, name):
    g1 = layers.BatchNormalization(name=f"{name}_bn_g")(layers.Conv2D(filters, 1, padding="same", name=f"{name}_wg")(g))
    x1 = layers.BatchNormalization(name=f"{name}_bn_x")(layers.Conv2D(filters, 1, padding="same", name=f"{name}_wx")(x))
    psi = layers.Activation("relu", name=f"{name}_relu")(layers.Add(name=f"{name}_add")([g1, x1]))
    psi = layers.Conv2D(1, 1, padding="same", name=f"{name}_psi")(psi)
    psi = layers.Activation("sigmoid", name=f"{name}_sigmoid")(layers.BatchNormalization(name=f"{name}_bn_psi")(psi))
    return layers.Multiply(name=f"{name}_mul")([x, psi])


def build_attention_unet(input_shape=(256, 256, 1), filters=(16, 32, 64, 128, 256)):
    inputs = layers.Input(input_shape, name="input_image")
    skips, x = [], inputs
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, name=f"enc{i + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = conv_block(x, filters[-1], name="bottleneck")
    for i, (f, s) in enumerate(zip(reversed(filters[:-1]), reversed(skips))):
        up = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i + 1}")(x)
        att = attention_gate(up, s, f, name=f"att{i + 1}")
        x = conv_block(layers.Concatenate(name=f"concat{i + 1}")([up, att]), f, name=f"dec{i + 1}")
    return models.Model(inputs, output_layer(x), name="Attention_UNet")


# ----------------------------------------------------------------------------- quantum bottleneck
class QuantumCircuitLayer(layers.Layer):
    """4-qubit parameterised circuit simulated with PennyLane's default.qubit.

    input  : (batch, n_qubits) values in [-1, 1]  (tanh output)
    circuit: RY(pi/2 * x_i) encoding; n_layers x [Rot(phi, theta, omega) on every qubit + CNOT ring]
    output : (batch, n_qubits) expectation values <Z_i>
    Trainable parameters: n_layers * n_qubits * 3 (= 24 for 4 qubits, 2 layers).
    """

    def __init__(self, n_qubits=4, n_layers=2, **kwargs):
        super().__init__(**kwargs)
        import pennylane as qml  # imported lazily so the rest of the package works without it
        self.n_qubits, self.n_layers = n_qubits, n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(np.pi / 2 * inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def build(self, input_shape):
        self.qweights = self.add_weight(name="qweights", shape=(self.n_layers, self.n_qubits, 3),
                                        initializer=tf.keras.initializers.RandomUniform(0.0, 2 * np.pi),
                                        trainable=True)
        super().build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs):
        def one(sample):
            out = self.circuit(tf.cast(sample, tf.float64), tf.cast(self.qweights, tf.float64))
            return tf.cast(tf.stack(out), tf.float32)

        return tf.map_fn(one, tf.cast(inputs, tf.float32),
                         fn_output_signature=tf.TensorSpec((self.n_qubits,), tf.float32))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


class TFCircuitLayer(layers.Layer):
    """Same circuit as QuantumCircuitLayer, simulated exactly in TensorFlow (no PennyLane).
    Fallback when the PennyLane-TensorFlow interface is unavailable (config.quantum_backend = "tf")."""

    def __init__(self, n_qubits=4, n_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits, self.n_layers = n_qubits, n_layers

    def build(self, input_shape):
        self.qweights = self.add_weight(name="qweights", shape=(self.n_layers, self.n_qubits, 3),
                                        initializer=tf.keras.initializers.RandomUniform(0.0, 2 * np.pi),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        from .quantum_sim import circuit_expvals, tf_ops
        return circuit_expvals(tf.cast(inputs, tf.float32), tf.cast(self.qweights, tf.float32), tf_ops(),
                               self.n_qubits)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)


def bottleneck_block(x, kind, n_qubits=4, n_layers=2, backend="tf"):
    """GAP features (512) -> Dense(4, tanh) -> [circuit | Dense(4, tanh) | identity] -> Dense(512, relu)."""
    z = layers.Dense(n_qubits, activation="tanh", name="pre_dense")(x)
    if kind == "quantum":
        layer_cls = QuantumCircuitLayer if backend == "pennylane" else TFCircuitLayer
        z = layer_cls(n_qubits, n_layers, name="quantum_circuit")(z)
    elif kind == "classical":
        z = layers.Dense(n_qubits, activation="tanh", name="classical_replacement")(z)
    elif kind != "none":
        raise ValueError(kind)
    return layers.Dense(512, activation="relu", name="post_dense")(z)


def build_hybrid_unet(input_shape=(256, 256, 1), bottleneck="quantum", n_qubits=4, n_q_layers=2,
                      filters=(32, 64, 128, 256, 512), dropouts=(0.1, 0.1, 0.2, 0.2, 0.3), backend="tf"):
    """Same encoder/decoder as the original Hybrid Quantum U-Net."""
    inputs = layers.Input(input_shape, name="input_image")
    skips, x = [], inputs
    for i, (f, d) in enumerate(zip(filters[:-1], dropouts[:-1])):
        x = conv_block(x, f, d, name=f"enc{i + 1}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = conv_block(x, filters[-1], dropouts[-1], name="bottleneck")
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    q = bottleneck_block(gap, bottleneck, n_qubits, n_q_layers, backend)
    h, w = input_shape[0] // 16, input_shape[1] // 16
    x = layers.Reshape((h, w, filters[-1]), name="q_reshape")(
        layers.Dense(h * w * filters[-1], activation="relu", name="q_expand")(q))
    for i, (f, s, d) in enumerate(zip(reversed(filters[:-1]), reversed(skips), reversed(dropouts[:-1]))):
        x = layers.UpSampling2D(2, name=f"up{i + 1}")(x)
        x = conv_block(layers.Concatenate(name=f"concat{i + 1}")([x, s]), f, d, name=f"dec{i + 1}")
    return models.Model(inputs, output_layer(x), name=f"Hybrid_UNet_{bottleneck}")


def build_model(name, cfg):
    shape = (cfg.image_size, cfg.image_size, cfg.in_channels)
    if name == "unet":
        return build_unet(shape)
    if name == "attention_unet":
        return build_attention_unet(shape)
    if name.startswith("hybrid_"):
        kind = {"hybrid_quantum": "quantum", "hybrid_classical": "classical", "hybrid_none": "none"}[name]
        return build_hybrid_unet(shape, kind, cfg.n_qubits, cfg.n_q_layers,
                                 backend=getattr(cfg, "quantum_backend", "tf"))
    raise ValueError(f"Unknown model: {name}")


def parameter_breakdown(model):
    """Total / trainable parameters, share held by q_expand, and size in MB (float32)."""
    total = model.count_params()
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    out = dict(total_params=int(total), trainable_params=trainable, size_mb=round(total * 4 / 2 ** 20, 2))
    names = [l.name for l in model.layers]
    if "q_expand" in names:
        out["q_expand_params"] = int(model.get_layer("q_expand").count_params())
    if "quantum_circuit" in names:
        out["circuit_params"] = int(model.get_layer("quantum_circuit").count_params())
    return out
