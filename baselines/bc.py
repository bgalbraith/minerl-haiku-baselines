import dill
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp

from dataset import load_data


MINERL_ENV = 'MineRLTreechopVectorObf-v0'
PARAMS_FILENAME = 'bc_params_treechop.pkl'


class PovStack(hk.Module):
    """ PovStack is a module for processing the point-of-view image data that
    comes from the agent's viewport. This input is in NHWC format for a shape
    of (N, 64, 64, 3).

    This model is inspired from
    https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/behavioral_cloning.py
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        conv_0 = hk.Conv2D(output_channels=32,
                           kernel_shape=(8, 8),
                           stride=4,
                           padding='SAME',
                           name='conv_0')
        layer_0 = (conv_0, jax.nn.relu)

        conv_1 = hk.Conv2D(output_channels=64,
                           kernel_shape=(4, 4),
                           stride=2,
                           padding='SAME',
                           name='conv_1')
        layer_1 = (conv_1, jax.nn.relu)

        conv_2 = hk.Conv2D(output_channels=64,
                           kernel_shape=(3, 3),
                           stride=1,
                           padding='SAME',
                           name='conv_2')
        layer_2 = (conv_2, jax.nn.relu)

        layer_3 = (hk.Flatten(),
                   hk.Linear(512, name='fc_0'), jax.nn.relu,
                   hk.Linear(128, name='fc_1'), jax.nn.relu)

        self.layers = layer_0 + layer_1 + layer_2 + layer_3

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VectorStack(hk.Module):
    """ VectorStack is a module for processing the obfuscated "vector" data that
    is included in the agent's observation. This is a densely encoded form of
    the discrete information regarding the state of the agent other than the
    viewport, e.g. current inventory. The input is of shape (N, 64)
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        layer_0 = (hk.Linear(32, name='fc_0'), jax.nn.relu)

        self.layers = layer_0

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def behavioral_cloning(batch):
    """ The full forward model definition """
    x_0 = PovStack(name='pov_stack')(batch[0])
    x_1 = VectorStack(name='vector_stack')(batch[1])
    x = jnp.concatenate((x_0, x_1), axis=1)
    return jnp.tanh(hk.Linear(64)(x))


@jax.jit
def mse_loss(logits, labels):
    """ Mean Squared Error loss """
    return jnp.mean(jnp.power(logits - labels, 2))


def main():
    net = hk.transform(behavioral_cloning)
    opt = optix.adam(0.001)

    @jax.jit
    def loss(params, batch):
        """ The loss criterion for our model """
        logits = net.apply(params, None, batch)
        return mse_loss(logits, batch[2])

    @jax.jit
    def update(opt_state, params, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        params = optix.apply_updates(params, updates)
        return params, opt_state

    @jax.jit
    def accuracy(params, batch):
        """ Simply report the loss for the current batch """
        logits = net.apply(params, None, batch)
        return mse_loss(logits, batch[2])

    train_dataset, val_dataset = load_data(MINERL_ENV,
                                           batch_size=32, epochs=100)

    rng = jax.random.PRNGKey(2020)
    batch = next(train_dataset)
    params = net.init(rng, batch)
    opt_state = opt.init(params)

    for i, batch in enumerate(train_dataset):
        params, opt_state = update(opt_state, params, batch)
        if i % 1000 == 0:
            print(accuracy(params, val_dataset))
        
        if i % 10000 == 0:
            with open(PARAMS_FILENAME, 'wb') as fh:
                dill.dump(params, fh)

    with open(PARAMS_FILENAME, 'wb') as fh:
        dill.dump(params, fh)



if __name__ == '__main__':
    main()
