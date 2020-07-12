import dill
import gym
import haiku as hk
import jax.numpy as jnp
import minerl

from bc import behavioral_cloning, MINERL_ENV, PARAMS_FILENAME



def main():
    # prepare model and load params for inference
    net = hk.transform(behavioral_cloning)
    with open(PARAMS_FILENAME, 'rb') as fh:
        params = dill.load(fh)

    def preprocess(obs):
        return [jnp.expand_dims(obs['pov'].astype(jnp.float32) / 255, axis=0),
                jnp.expand_dims(obs['vector'], axis=0)]

    env = gym.make(MINERL_ENV)

    n_episodes = 10
    timeout = 1000
    episode_rewards = []
    for ep in range(n_episodes):
        done = False
        net_reward = 0

        obs = env.reset()
        t = 0
        while not done:
            action = {
                'vector': net.apply(params, None, preprocess(obs)).squeeze()
            }
            obs, reward, done, info = env.step(action)
            if reward <= 0:
                t += 1
            else:
                t = 0
            net_reward += reward
            if t >= timeout:
                break
        print(f"Total reward: {net_reward:.2f}")
        episode_rewards.append(net_reward)

    episode_rewards = jnp.array(episode_rewards)
    mean_r = jnp.mean(episode_rewards)
    std_r = jnp.std(episode_rewards)
    print(f"average reward (n={n_episodes}): {mean_r:.2f} +/- {std_r:.2f}")

    env.close()


if __name__ == '__main__':
    main()
