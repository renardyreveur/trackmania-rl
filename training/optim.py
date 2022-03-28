import jax.numpy as jnp
from jax import grad, jit


# SGD
def sgd(w, g, lr=0.001, op_params=None, **kwargs):
    if op_params is None:
        return [{k: {"step": 0} for k, v in layer.items()} for layer in w]
    return w - lr * g, op_params


# Adam with weight decay
def adamw(w, g, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, op_params=None, **kwargs):
    if op_params is None:
        return [{k: {"m0": 0, "v0": 0, "step": 0} for k, v in layer.items()} for layer in w]

    # Momentum
    # Gradient direction is smoothed by exponentially weighing the moving averages
    op_params["m0"] = betas[0] * op_params["m0"] + (1 - betas[0]) * g
    # RMSProp
    # Gradient magnitude is smoothed such that it slows down near flats, and doesn't flick off at suboptimal gradients
    op_params["v0"] = betas[1] * op_params["v0"] + (1 - betas[1]) * (g ** 2)

    # Estimation bias correction
    mt_hat = op_params["m0"] / (1 - betas[0] ** op_params['step'])
    vt_hat = op_params["v0"] / (1 - betas[1] ** op_params['step'])

    # Weight decay is not the same as L2 regularization when not standard SGD ->
    # substituting g with (g + wd*w) which comes from adding sum of squared weights to loss
    # is not equal to weight decay when gradients are altered (such as momentum, etc.)
    new_w = w - (lr * mt_hat / (jnp.sqrt(vt_hat) + eps) + weight_decay * w)
    return new_w, op_params


@jit
def update(loss_fn, model, params, x, y, optimizer, optimizer_params):
    # Calculate gradients of loss w.r.t params
    gradient = grad(loss_fn, argnums=0)(params, model, x, y)

    # Step counter in optimizer_params update
    [[v.update({"step": v['step'] + 1}) for k, v in ly.items()] for ly in optimizer_params]

    # Get updated parameters
    new_params, new_oparams = [], []
    for p, g, o in zip(params, gradient, optimizer_params):  # For each layer
        upd_params = {}
        upd_oparams = {}
        for k, v in p.items():  # For each parameter associated with the layer
            new_w, new_o = optimizer(v, g[k], op_params=o[k])
            upd_params.update({k: new_w})
            upd_oparams.update({k: new_o})
        new_params.append(upd_params)
        new_oparams.append(upd_oparams)
    return new_params, new_oparams
