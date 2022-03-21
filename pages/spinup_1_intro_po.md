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

## Osservazioni per migliorare VPG

### Expected grad-log-prob lemmma

- serve per poter inserire le **baseline**, che migliorano la stabilità dell'apprendimento
- dimostra che l'introduzione di una baseline non compromette l'average return

$$
\begin{aligned}
\int_x P_\theta(x) &= 1\\
\nabla_\theta \int_x P_\theta(x) &= 0\\
0&=\nabla_\theta \int_x P_\theta(x)\\
&=\int_x \nabla_\theta P_\theta(x)\\
&=\int_x  P_\theta(x) \nabla_\theta \log P_\theta(x)\\
&= E_{x \sim P_\theta}[ \nabla_\theta \log P_\theta(x) ] \\
\end{aligned}
$$

la proprietà vale per qualsiasi distribuzione di probabilità $P_\theta(x)$.

### Reward-to-go policy gradient

- il policy gradient prende in considerazione tutta la traiettoria per migliorare la policy
- intuitivamente, soltanto i reward futuri sono influenzati dalla scelta della policy allo stato corrente
- si può derivare questa intuizione matematicamente, ottenendo il **reward-to-go policy gradient**

$$
\nabla_\theta J(\pi_\theta) = E_{\pi_\theta}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t \vert s_t) \underbrace{\sum_{t'=t}^T R (s_{t'},a_{t'}, s_{t'+1})}_{\stackrel{def}{=}\hat{R}_t}\right]
$$

Vantaggi del reward-to-go policy gradient:

- senza il reward-to-go, quindi includendo azioni passate, si includono reward con media zero ma con varianza maggiore di zero: si aggiunge rumore al gradiente

#### Implementazione in VPG

Modifica rispetto all'implementazione di VPG. Calcolo del reward-to-go:

- prima i reward per ogni time-step erano uguali al reward totale della traiettoria
- ora al crescere di $t$ i reward diminuiscono perché si considera il reward residuo della traiettoria

```python
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs
```

Si modifica dunque la determinazione dei pesi per la loss:

```python
# the weight for each logprob(a_t|s_t) is reward-to-go from t
batch_weights += list(reward_to_go(ep_rews))
```

### Baselines

Per migliorare l'apprendimento, si vuole focalizzare il miglioramento della policy verso le traiettorie che hanno reward maggiore rispetto ad una **baseline**. L'introduzione di una baseline è giustificato dall'**expected grad-log-prob lemma** dimostrato in precedenza:

$$
E_{a_t \sim \pi_\theta}[ \nabla_\theta \log \pi_\theta(a_t \vert s_t) b(s_t) ]=0
$$

Possiamo quindi modificare il policy gradient senza cambiare l'expectation, fintanto che la baseline $b(s_t)$ non dipende da $\theta$:

$$
\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T\nabla_\theta \log \pi_\theta(a_t \vert s_t) \left( \sum_{t'=t}^T R(s_{t'},a_{t'},s_{t'+1}) - b(s_t) \right) \right]
$$

#### Baseline on-policy value function $V^\pi(s_t)$

Spesso si sceglie $b(s_t) = V^\pi(s_t)$:

- l'agente migliora la policy solo se il reward è maggiore di quello che si aspetta
- l'apprendimento è più stabile e più rapido
- la baseline riduce la varianza della stima del gradiente
- in pratica $V^\pi(s_t)$ non può essere calcolata esattamente ma deve essere approssimata da una rete neurale $V_\phi(s_t)$ ottimizzata parallelamente alla policy
  - il metodo più semplice è approssimare $V_\phi(s_t)$ con una loss MSE rispetto al reward-to-go
  - si fa uno o più step di SGD con la policy all'iterazione $k$

$$
\phi_k = \arg \min_\phi E_{s_t,\hat{R}_t \sim \pi_k} \left[ \left( V_\phi(s_t) - \hat{R}_t \right)^2  \right]
$$

### Riassunto delle forme del policy gradient

$$
\nabla_\theta J(\pi_\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T\nabla_\theta \log \pi_\theta(a_t \vert s_t) \Phi_t \right]
$$

$\Phi_t$ può assumere le sequenti forme:

|Nome|$\Phi_t$|
|:--|:--|
|reward semplice| $R(\tau)$|
|reward-to-go|$\sum_{t'=t}^t R(s_{t'},a_{t'},s_{t'+1})$|
|reward-to-go con baseline|$\sum_{t'=t}^t R(s_{t'},a_{t'},s_{t'+1}) - b(s_t)$|
|on-policy action-value function|$Q^{\pi_\theta}(s_t,a_t)$
|advantage function|$A^{\pi_\theta}(s_t,a_t)=Q^{\pi_\theta}(s_t,a_t)$-$V^{\pi_\theta}(s_t)$|

Osservazioni:

- la formulazione con $A^{\pi_\theta}(s_t,a_t)$ (approssimata) è molto comune
- varie scelte di $\Phi_t$ sono investigate in **General Advantage Estimation**

### Vanilla Policy Gradient (VPG)

Forma di VGP con advantage function

![VPG Spinning Up](img/vpg_spinup.svg "VPG Spinning Up")