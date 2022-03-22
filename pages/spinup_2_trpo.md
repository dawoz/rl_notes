**[< Torna all'indice dei conenuti](../index.md)**

# Trust Region Policy Optimization

Idea:

- step più grande possibile tale che soddisfi un vincolo aggiuntivo sulla **distanza** tra la veccha e la nuova policy
- distanza tra le policy: **Kullback-Leibler divergence**, circa la distanza tra due distribuzioni di probabilità
- in questo modo si evita di finire a costruire una cattiva policy, come talvolta capita con VPG
- TRPO incrementa monotonicamente la performance

Caratteristiche TPRO:

- on-policy
- ambienti con spazi di azioni discreti o continui

## Problema di ottimizzazione TPRO

$$
\begin{aligned}
\max_\theta \quad & \mathcal{L}(\theta_k,\theta)\\
s.t \quad & \overline{D}_{KL}(\theta_k \Vert \theta) \le \delta
\end{aligned}
$$

- $\mathcal{L}(\theta_k,\theta)$ è definito **surrogate advantage**, cioè quanto la nuova policy $\pi_\theta$ performa rispetto alla vecchia policy $\pi_{\theta_k}$ usando i dati raccolti dalla vecchia policy $\pi_{\theta_k}$:

$$
\mathcal{L}(\theta_k,\theta) = E_{a,s \sim \pi_{\theta_k}} \left[ \frac{\pi_\theta(a \vert s)}{\pi_{\theta_k}(a \vert s)} A^{\pi_{\theta_k}} (s,a)  \right]
$$

- $\overline{D}_{KL}(\theta_k \Vert \theta) \le \delta$ è la **sample average KL-divergence**, cioè la media delle KL-divergence delle policy rispetto agli stati raccolti dalla vecchia policy:

$$
\overline{D}_{KL}(\theta_k \Vert \theta) \stackrel{def}{=} E_{s \sim \pi_{\theta_k}}\left[ D_{KL}(\pi_\theta(\cdot \vert s) \ \vphantom{\sum}\Vert \ \pi_{\theta_k}(\cdot \vert s)) \right]
$$

**Nota**: sia la funzione obiettivo che il vincolo valgono $0$ per $\theta=\theta_k$

## Approssimazione del problema teorico

Osservazione: la formulazione di TRPO teorica è complessa dal punto di vista matematico e computazionale, perciò nella pratica si utilizza il seguente problema approssimato

$$
\begin{aligned}
\max_\theta \quad & g^T(\theta-\theta_k) & (\approx \mathcal{L}(\theta_k,\theta)) \\
s.t.\quad &\frac{1}{2}(\theta-\theta_k)^TH(\theta-\theta_k) \le \delta & (\approx \overline{D}_{KL}(\theta_k \Vert \theta))
\end{aligned}
$$

con $g=\nabla_\theta\mathcal{L}(\theta,\theta_k)$.

**Nota**: $g$ è uguale al policy gradient $\nabla_\theta J(\pi_\theta)$ se si valuta il gradiente con $\theta=\theta_k$.

## Soluzione analitica

Tramite la dualità lagrangiana si può trovare la soluzione 

$$
\theta_{k+1} = \theta_{k+1} + \sqrt{\frac{2 \delta}{g^T H^{-1}g}}H^{-1}g
$$

Osservazioni:

- $H$ è la matrice **hessiana** della sample average KL-divergence
- così com'è, abbiamo ottenuto il é **Natural Policy Gradient**
- siccome è una soluzione prodotta dall'approssimazione di Taylor, potrebbe non soddisfare il vincolo o non migliorare la funzione obiettivo

Soluzione: usare una **backtracking line search**

$$
\theta_{k+1} = \theta_{k+1} + \alpha^j\sqrt{\frac{2 \delta}{g^T H^{-1}g}}H^{-1}g
$$

con $\alpha \in (0,1)$ e $j=0,...$ più piccolo che soddisfa il vincolo e che migliora la surrogate loss.

## Evitare il calcolo di $H^{-1}$ e la memorizzazione di $H$

Vale la seguente equazione:

$$
Hx=g
$$

Problemi:

- potenzialmente $\theta$ contiene milioni di parametri, quindi calcolare la matrice $H-1$ è molto costoso
- ugualmente problematico è memorizzare l'intera matrice hessiana $H$

Soluzioni:

- esprimere l'**hessian-vector product** $Hx$ permette di evitare di calcolare interamente $H$
- utilizzare il **metodo del gradiente coniugato** permette di calcolare $x \approx H^{-1}g$ dal sistema $Hx=g$, in cui dunque si calcolano on-demand i prodotti vettoriali $Hx$ (da punto precedente)

Si può dunque derivare l'**hessian-vector product**

$$
Hx = \nabla_\theta \left( \left( \nabla_\theta \overline{D}_{KL}(\theta \Vert \theta_k) \right)^T x \right)
$$

## Pseudocodice

![TRPO Spinning UP](img/trpo_spinup.svg "TRPO Spinning UP")

Osservazioni:

- TRPO rende la polici sempre meno random
- la policy può convergere ad un minimo locale
