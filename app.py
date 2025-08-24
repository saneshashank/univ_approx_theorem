from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# -------- Target functions (continuous ones show UAT best) --------
def f_sine(x):   return np.sin(2 * np.pi * x)
def f_abs(x):    return np.abs(x - 0.5)
def f_bump(x):   return np.exp(-60.0 * (x - 0.5) ** 2)
def f_cubic(x):  return (2 * x - 1) ** 3
def f_square(x): return 0.8 * np.sign(np.sin(2 * np.pi * x))  # discontinuous (for fun)

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

def build_random_features(x, width, activation_fn, rng):
    """
    Random kitchen-sink features: phi(x) = activation(x * W + b)
    x: shape (N,)
    returns Phi: shape (N, width)
    """
    x = x.reshape(-1, 1)  # (N, 1)
    # Heuristic scales so different activations behave nicely without tuning
    W = rng.normal(loc=0.0, scale=2.0, size=(1, width))        # (1, width)
    b = rng.uniform(low=-np.pi, high=np.pi, size=(width,))      # (width,)
    Z = x @ W + b  # (N, width)
    Phi = activation_fn(Z)
    return Phi

def fit_ridge_closed_form(Phi, y, lam):
    """
    Ridge regression: w = (Phi^T Phi + lam I)^-1 Phi^T y
    We also add a bias column.
    """
    N, W = Phi.shape
    Phi_aug = np.concatenate([np.ones((N, 1)), Phi], axis=1)  # bias term
    A = Phi_aug.T @ Phi_aug + lam * np.eye(W + 1)
    b = Phi_aug.T @ y
    w = np.linalg.solve(A, b)  # (W+1,)
    return w  # includes bias as w[0]

def predict(Phi, w):
    N, W = Phi.shape
    Phi_aug = np.concatenate([np.ones((N, 1)), Phi], axis=1)
    return Phi_aug @ w

def make_grid(n=400):
    return np.linspace(0.0, 1.0, n)

@app.route("/")
def index():
    return render_template(
        "index.html",
        targets={k: v[0] for k, v in TARGETS.items()},
        activations={k: v[0] for k, v in ACTIVATIONS.items()},
    )

@app.route("/approximate")
def approximate():
    # --- Parse UI params ---
    target_key = request.args.get("target", "sine")
    act_key    = request.args.get("activation", "tanh")
    width      = max(1, min(int(request.args.get("width", 50)), 2048))
    lam        = float(request.args.get("lam", 1e-3))
    seed       = int(request.args.get("seed", 0))
    noise_std  = float(request.args.get("noise", 0.0))

    if target_key not in TARGETS or act_key not in ACTIVATIONS:
        return jsonify({"error": "Invalid target or activation"}), 400

    target_name, f = TARGETS[target_key]
    act_name, act_fn = ACTIVATIONS[act_key]

    rng = np.random.default_rng(seed)

    # Training data (small but enough)
    x_train = np.linspace(0.0, 1.0, 512)
    y_train = f(x_train)
    if noise_std > 0:
        y_train = y_train + rng.normal(0, noise_std, size=y_train.shape)

    # Build random features and fit ridge
    Phi = build_random_features(x_train, width, act_fn, rng)
    w = fit_ridge_closed_form(Phi, y_train, lam)

    # Evaluate on a grid for plotting
    x_grid = make_grid(500)
    y_true = f(x_grid)
    Phi_grid = build_random_features(x_grid, width, act_fn, rng)
    y_pred = predict(Phi_grid, w)

    mse = float(np.mean((y_true - y_pred) ** 2))

    return jsonify({
        "x": x_grid.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "mse": mse,
        "params": {
            "target": target_name,
            "activation": act_name,
            "width": width,
            "lambda": lam,
            "seed": seed,
            "noise": noise_std
        }
    })

if __name__ == "__main__":
    # Run in dev mode; for production use a proper WSGI server
    app.run(host="127.0.0.1", port=5000, debug=True)
