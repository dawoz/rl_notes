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

## Implementazione PyTorch di un semplice Vanilla Policy Gradient

Definizione della policy network:

```python
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
```

La distribuzione di probabilità sulle azioni è calcolata utilizzando una distribuzione categorica dagli output (logits) della policy network:

```python
# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)
```

Sample azione dalla distribuzione ottenuta:

```python
# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()
```

Calcolo della loss $\nabla_\theta\frac{1}{ \vert \mathcal{D} \vert} \sum_{\tau \in D}\sum_{t=0}^T \log \pi_\theta(a_t \vert s_t) R(\tau)$:

- l'espressione è negata dal momento che PyTorch implementa il gradient descent e vogliamo calcolare il gradient ascent
- nota: il gradiente non è calcolato perché sarà calcolato successivamente con uno step dell'ottimizzatore

```python
# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

I pesi corrispondono a $R(\tau)$

```python
# the weight for each logprob(a|s) is R(tau)
batch_weights += [ep_ret] * ep_len
```

####  Codice completo: [repository Spinning Up 🔗](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py)

**Note**:

- le logits $x_j$ sono i valori in $(-\infty,+\infty)$ tali che possono essere convertiti in probabilità tramite la softmax $p_j = \frac{\exp x_j}{\sum_i \exp x_i}$
- il gradiente della loss è il policy gradient
- la loss calcolata non è la loss intesa nel senso dell'apprendimento supervisionato:
  - i target dipendono dalla distribuzione
  - non misura la performance: dopo il primo step non ha la garanzia di migliorare l'expected return
  - in pratica sta facendo overfitting: questo però non prende in considerazione l'errore di generalizzazione
  - la performance ora è l'average return
