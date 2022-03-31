import jax.numpy as jnp

from .policy_model import frame_feature_extractor, get_params, linear, gelu, attention, layer_norm


def q_func(in_state, in_action, feat_dims, params=None):
    feature_seq = []
    new_params = []
    for frame_num in range(in_state.shape[1]):
        x, p0 = frame_feature_extractor(jnp.expand_dims(in_state[:, frame_num, :, :], 1), dims=feat_dims,
                                        params=get_params(params, 0))
        feature_seq.append(x)
    new_params.append(p0)

    feature_seq = jnp.stack(feature_seq, axis=1)

    # Multi-head attention, concat results
    a_out, p1 = attention(feature_seq, dim=feat_dims[-1], params=get_params(params, 1))
    new_params.append(p1)

    # Residual skip-connection and layer norm
    reproj, p2 = linear(a_out, feat_dims[-1], params=get_params(params, 2))
    new_params.append(p2)
    reproj = gelu(reproj + feature_seq)
    nmres, p3 = layer_norm(reproj, dim=2, params=get_params(params, 3))
    new_params.append(p3)

    # Flatten
    nmres = jnp.reshape(nmres, (nmres.shape[0], -1))

    # Action head
    in_action, p4 = linear(in_action, feat_dims[-1], params=get_params(params, 4))
    new_params.append(p4)
    in_action = gelu(in_action)

    # Combine state and action features
    nmres = jnp.concatenate([nmres, in_action], axis=-1)

    # Return Q value
    q_val, p5 = linear(nmres, feat_dims[-1], params=get_params(params, 5))
    new_params.append(p5)
    q_val = gelu(q_val)

    q_val, p6 = linear(q_val, feat_dims[-1] // 2, params=get_params(params, 6))
    new_params.append(p6)
    q_val = gelu(q_val)

    q_val, p7 = linear(q_val, 1, params=get_params(params, 7))
    new_params.append(p7)

    return q_val, new_params
