**[< Torna all'indice dei conenuti](../index.md)**

# Proximal Policy Optimization

Motivazioni PPO:

- step di ottimizzazione grande ma che non si allontana troppo dalla policy corrente, in modo da non far collassare le prestazioni

Miglioramenti rispetto a TRPO:

- TRPO è un problema di secondo ordine complesso
- PPO è un problema di prim'ordine che usa altri metodi per rendere la nuova policy vicino alla prima

Varianti di PPO:

- **PPO-Penalty**:  inserimento della KL-divergence nella funzione obiettivo con una penalità
  - il vincolo della KL-divergence è approssimativamente rispettato
  - la penalità è rimodellata durante l'apprendimento
- **PPO-Clip**: non ha il termine con la KL-divergence ed è senza vincoli
  - inserisce un clipping per rimuovere incentivi ad allontanarsi molto dalla vecchia policy

Caratteristiche PPO:

- on-policy
- ambienti con azioni discrete o continue

## Problema di ottimizzazione PPO-Clip

$$
\theta_{k+1} = \arg \max_\theta E_{s,a \sim \pi_{\theta_k}}[L(s,a,\theta_k,\theta)]
$$

Normalmente l'ottimizzazione avviene in minibatch con SGD per massimizzare $L$.

$L$ è definita come 

$$
L(s,a,\theta_k,\theta) = \min \left( \frac{\pi_\theta (a,s)}{\pi_{\theta_k}(a,s)} A^{\pi_{\theta_k}}(s,a), clip\left(\frac{\pi_\theta (a,s)}{\pi_{\theta_k}(a,s)},1-\epsilon,1+\epsilon  \right) A^{\pi_{\theta_k}}(s,a)\right)
$$

dove $\epsilon$ è un iperparametro piccolo che indica quanto può essere diversa la nuova policy.

La loss può essere semplificata, nella formula usata nell'implementazione, come:

$$
L(s,a,\theta_k,\theta) = \min \left( \frac{\pi_\theta (a,s)}{\pi_{\theta_k}(a,s)} A^{\pi_{\theta_k}}(s,a), \ g(\epsilon, A^{\pi_{\theta_k}}(s,a))\right)
$$

con

$$
g(\epsilon,A) = \left \lbrace
\begin{aligned}
& (1+\epsilon)A, & A \ge 0\\
& (1-\epsilon)A, & A < 0
\end{aligned}
\right.
$$

**Intuizione**: si suppone un advantage positivo

$$
L(s,a,\theta_k,\theta) = \min \left( \frac{\pi_\theta (a,s)}{\pi_{\theta_k}(a,s)}, (1+\epsilon)\right)  A^{\pi_{\theta_k}}(s,a)
$$

- se l'advantage è positivo allora $\pi_\theta(a\vert s)$ tende a salire
- quando si ha $\pi_\theta(a \vert s) > (1+\epsilon)\pi_{\theta_k}(a \vert s)$
- si ottiene che il massimo miglioramente dell'obiettivo è $(1+\epsilon)A^{\pi_{\theta_k}}(s,a)$

- l'inverso vale quando si ha un advantage negativo: $L(s,a,\theta_k,\theta) = \max \left( \frac{\pi_\theta (a,s)}{\pi_{\theta_k}(a,s)}, (1-\epsilon)\right)  A^{\pi_{\theta_k}}(s,a)$

Osservazioni:

- si ha quindi che la nuova policy non beneficia ad andare troppo in là rispeto alla vecchia policy
- $\epsilon$ funziona come termine di regolarizzazione
- nonostante PPO-Clip eviti moltissime volte di raggiungere una policy troppo lontana, può capitare che la nuova policy sia troppo lontana
  - si può ad esempio usare **early stopping**: se la sample-average KL-divergence calcolata supera un certo threshold si smette di fare step di gradiente.

## Pseudocodice

![PPO Spinning Up](img/ppo_spinup.svg "PPO Spinning Up")
