**[< Torna all'indice dei conenuti](../index.md)**

# Intro to Policy Optimization

Funzione obiettivo: massimizzare il **finite-horizon undiscounted expected return**

$$
\arg \max_\theta J(\pi_\theta) = \arg \max_\theta E_{\tau \sim \pi_\theta}[R(\tau)]
$$

Gradient ascent:

$$
\theta_{k+1} = \theta_{k} + \alpha \underbrace{\nabla_\theta J(\pi_\theta)\vert_{\theta=\theta_k}}_{\text{policy gradient}}
$$

Come calcolare?

1) portare il gradiente ad una forma chiusa con valore attesto
2) raccogliere traiettorie in modo da poter stimare il valore atteso (con una media)

Derivazione del policy gradient:

$$
\begin{aligned}
\nabla_\theta J(\pi_\theta) &= \nabla_\theta E_{\tau \sim \pi_\theta}[R(\tau)]\\
&= \nabla_\theta \int_\tau P(\tau \vert \theta)R(\tau) & \text{(def. } E[ \cdot ])\\
&= \int_\tau \nabla_\theta  P(\tau \vert \theta)R(\tau)\\
&= \int_\tau  P(\tau \vert \theta) \nabla_\theta \log P(\tau \vert \theta) R(\tau) & \text{(log-derivative trick)}\\
&= E_{\tau \sim \pi_\theta}[  \nabla_\theta \log P(\tau \vert \theta) R(\tau)] & \text{(def. } E[ \cdot ])\\
&= E_{\tau \sim \pi_\theta} \left[  \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \vert s_t) R(\tau) \right] & \text{(grad-log-prob)}\\
\end{aligned}
$$

Derivazione **grad-log-prob**:

$$
\begin{aligned}
\nabla_\theta \log P(\tau \vert \theta) &= \nabla_\theta \log \left( \underbrace
{\rho_0(s_0)}_{\text{prob }s_0} \prod_{t=0}^T P(s_{t+1} \vert s_t, a_t) \pi_\theta(a_t \vert s_t) \right)\\
&=\nabla_\theta \left( \log \rho_0(s_0) + \sum_{t=0}^T \left( \vphantom{\sum}\log P(s_{t+1} \vert s_t, a_t) + \log \pi_\theta(a_t \vert s_t) \right) \right)\\
&= \underbrace{\nabla_\theta \log \rho_0(s_0)}_{(*)} + \sum_{t=0}^T \left( \vphantom{\sum} \underbrace{\nabla_\theta \log P(s_{t+1} \vert s_t, a_t)}_{(*)} + \nabla_\theta\log \pi_\theta(a_t \vert s_t) \right) \\
&=  \sum_{t=0}^T \nabla_\theta\log \pi_\theta(a_t \vert s_t) & ((*)\text{ ind. da } \theta) \\
\end{aligned}
$$

Nella pratica, si colleziona un insieme $\mathcal{D}$ di traiettorie e si stima il valore del gradiente con

$$
\hat{g} = \frac{1}{ \vert \mathcal{D} \vert} \sum_{\tau \in D}\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \vert s_t) R(\tau)
$$

**Nota** $\pi_\theta( \cdot \vert s_t)$ è implementata con una rete neurale che impara ad associare un'osservazione ad una distribuzione di probabilità sulle azioni

Implementazione PyTorch di un semplice Vanilla Policy Gradient:

```python
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
```