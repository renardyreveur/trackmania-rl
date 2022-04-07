from functools import partial

import jax
import jax.numpy as jnp
from jax import grad


# Polyak Averaging of weights
def polyak(param1, param2, smooth_factor):
    smoothed_param = []
    for p1, p2 in zip(param1, param2):
        if isinstance(p1, list):
            smoothed_param.append(polyak(p1, p2, smooth_factor))
        else:
            smoothed_param.append({k: smooth_factor * v + (1 - smooth_factor) * p2[k] for k, v in p1.items()})
    return smoothed_param


# SGD
def sgd(w, g, op_params=None, lr=0.001, **kwargs):
    def init_sgd_params(params):
        params = [init_sgd_params(x) if isinstance(x, list)
                  else {k: {"step": 0} for k, v in x.items()}
                  for x in params]
        return params

    if op_params is None:
        return init_sgd_params(w)
    return w - lr * g, op_params


# Adam with weight decay
def adamw(w, g, op_params=None, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001):
    def init_adamw_params(params):
        params = [init_adamw_params(x) if isinstance(x, list)
                  else {k: {"m0": jnp.zeros_like(v), "v0": jnp.zeros_like(v), "step": 0} for k, v in x.items()}
                  for x in params]
        return params

    if op_params is None:
        return init_adamw_params(w)
        # return [{k: {"m0": 0, "v0": 0, "step": 0} for k, v in layer.items()} for layer in w]

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


# ----- Optimizer -----
def update_optim_params(params):
    [update_optim_params(x) if isinstance(x, list) else [v.update({"step": v['step'] + 1}) for k, v in x.items()]
     for x in params]


# TODO: The for loop and nested function structure here is what probably causes the long JIT compilation time
def update_parameters(optimizer, params, gradient, op_params):
    new_params, new_oparams = [], []
    for p, g, o in zip(params, gradient, op_params):
        if isinstance(p, list):
            n_p, n_op = update_parameters(optimizer, p, g, o)
            new_params.append(n_p)
            new_oparams.append(n_op)
        else:
            upd_params = {}
            upd_oparams = {}
            for k, v in p.items():  # For each parameter associated with the layer
                new_w, new_o = optimizer(v, g[k], op_params=o[k])
                upd_params.update({k: new_w})
                upd_oparams.update({k: new_o})
            new_params.append(upd_params)
            new_oparams.append(upd_oparams)
    return new_params, new_oparams


@partial(jax.jit, static_argnums=(2,))
# What if loss_fn, optimizer partial functions break tracing? Separate functions for each?
def update(in_data, loss_fn, model, params: tuple, optimizer, optimizer_params: tuple):
    # Calculate loss
    loss = loss_fn(params, model, in_data)

    # Calculate gradients of loss w.r.t params
    gradient = grad(loss_fn, argnums=0)(params, model, in_data)

    # Step counter in optimizer_params update
    update_optim_params(optimizer_params)

    # Updatable parameters
    # The length of the optimizer_params will determine how many from the start of the parameter list to actually update
    num_up_params = len(optimizer_params)
    new_params, new_oparams = update_parameters(optimizer,
                                                params[:num_up_params],
                                                gradient[:num_up_params],
                                                optimizer_params)

    """
    # Update parameters using tree_map and tree_transpose from JAX
    # Thought this might speed up the JAX compilation, but it didn't...
    # Still leaving it in here just in case...
    updated_params = jax.tree_map(optimizer, params[:num_up_params], gradient[:num_up_params], optimizer_params)
    new_params, new_oparams = jax.tree_util.tree_transpose(outer_treedef=jax.tree_structure(params[:num_up_params]),
                                                           inner_treedef=jax.tree_structure(
                                                               (0, {'m0': 0, 'v0': 0, 'step': 0})),
                                                           pytree_to_transpose=updated_params)
    new_oparams = jax.tree_util.tree_transpose(outer_treedef=jax.tree_structure({'m0': 0, 'v0': 0, 'step': 0}),
                                               inner_treedef=jax.tree_structure(params[:num_up_params]),
                                               pytree_to_transpose=new_oparams)
    """

    # If input was single parameter wrapped in a tuple, reduce that tuple when returning
    if len(new_params) == 1 and isinstance(new_params[0], list):
        new_params, new_oparams = new_params[0], new_oparams[0]

    return loss, new_params, new_oparams
