from flask import Flask, render_template, request, jsonify, url_for
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------- Target functions --------
def f_sine(x):   return np.sin(2 * np.pi * x)
def f_abs(x):    return np.abs(x - 0.5)
def f_bump(x):   return np.exp(-60.0 * (x - 0.5) ** 2)
def f_cubic(x):  return (2 * x - 1) ** 3
def f_square(x): return 0.8 * np.sign(np.sin(2 * np.pi * x))  # discontinuous demo

TARGETS = {
    "sine":   ("sin(2πx)", f_sine),
    "abs":    ("|x - 0.5|", f_abs),
    "bump":   ("Gaussian bump", f_bump),
    "cubic":  ("(2x-1)^3", f_cubic),
    "square": ("square wave (discontinuous)", f_square),
}

# -------- Activations --------
def act_tanh(z):    return np.tanh(z)
def act_relu(z):    return np.maximum(0.0, z)
def act_sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

ACTIVATIONS = {
    "tanh": ("tanh", act_tanh),
    "relu": ("ReLU", act_relu),
    "sigmoid": ("sigmoid", act_sigmoid),
}

# ---- Utilities ----
def parse_layers(s: str):
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return [64]
        widths = []
        for p in parts[:10]:
            w = max(1, min(2048, int(p)))
            widths.append(w)
        return widths
    except Exception:
        return None

def he_scale(fan_in, act_key):
    if act_key == "relu":
        return np.sqrt(2.0 / max(1, fan_in))
    else:
        return 1.0 / np.sqrt(max(1, fan_in))

def init_random_params(layer_sizes, act_key, rng):
    params = []
    in_dim = 1
    for width in layer_sizes:
        scale = he_scale(in_dim, act_key)
        W = rng.normal(0.0, scale, size=(in_dim, width))
        b = rng.uniform(-1.0, 1.0, size=(width,))
        params.append((W, b))
        in_dim = width
    return params

def forward_features(x, params, act_fn):
    H = x.reshape(-1, 1)
    for (W, b) in params:
        H = act_fn(H @ W + b)
    return H

def fit_ridge_closed_form(Phi, y, lam):
    N, W = Phi.shape
    Phi_aug = np.concatenate([np.ones((N, 1)), Phi], axis=1)
    A = Phi_aug.T @ Phi_aug + lam * np.eye(W + 1)
    b = Phi_aug.T @ y
    w = np.linalg.solve(A, b)
    return w

def predict_with_weights(Phi, w):
    Phi_aug = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    return Phi_aug @ w

def make_grid(n=500):
    return np.linspace(0.0, 1.0, n)

@app.route("/")
def index():
    return render_template(
        "index.html",
        targets={k: v[0] for k, v in TARGETS.items()},
        activations={k: v[0] for k, v in ACTIVATIONS.items()},
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/approximate")
def approximate():
    target_key = request.args.get("target", "sine")
    act_key    = request.args.get("activation", "tanh")
    layers_str = request.args.get("layers", "64")
    lam        = float(request.args.get("lam", 1e-3))
    seed       = int(request.args.get("seed", 0))
    noise_std  = float(request.args.get("noise", 0.0))

    if target_key not in TARGETS or act_key not in ACTIVATIONS:
        return jsonify({"error": "Invalid target or activation"}), 400

    layer_sizes = parse_layers(layers_str)
    if layer_sizes is None:
        return jsonify({"error": "Invalid layers format. Use comma-separated integers like 128,64,32"}), 400
    if len(layer_sizes) > 10:
        return jsonify({"error": "Max 10 layers allowed"}), 400

    target_name, f = TARGETS[target_key]
    act_name, act_fn = ACTIVATIONS[act_key]
    rng = np.random.default_rng(seed)

    x_train = np.linspace(0.0, 1.0, 512)
    y_train = f(x_train)
    if noise_std > 0:
        y_train = y_train + rng.normal(0, noise_std, size=y_train.shape)

    params = init_random_params(layer_sizes, act_key, rng)
    Phi_train = forward_features(x_train, params, act_fn)
    w = fit_ridge_closed_form(Phi_train, y_train, lam)

    x_grid = make_grid(500)
    y_true = f(x_grid)
    Phi_grid = forward_features(x_grid, params, act_fn)
    y_pred = predict_with_weights(Phi_grid, w)

    mse = float(np.mean((y_true - y_pred) ** 2))

    return jsonify({
        "x": x_grid.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "mse": mse,
        "params": {
            "target": target_name,
            "activation": act_name,
            "layers": layer_sizes,
            "depth": len(layer_sizes),
            "lambda": lam,
            "seed": seed,
            "noise": noise_std
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
