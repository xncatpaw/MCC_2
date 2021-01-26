# MCC_2
La Monte-Carlo challenge 2 de cours MAP556.

Dans ce projet, on emploie `pytorch` à réaliser le modèle de contrôle. Ce modèle se compose par des couchés linéaires avec les fonction d'activé `ReLu`. 

C'est un problème d'apprentissage par renforcement.
Le modèle exporte le contrôle $U_t$ en recevant $t$ et $X_t$. 
On définit alors une classe `Env` qui simule l'environnement. Donné le stat $X_{t_i}$, le contrôle $U_{t_i}$ et le vent $V_{t_i}$, `env` calcule le stat $X_{t_{i+1}}$. 

Pour un modèle de contrôle $\phi^\theta$ fixé, on simule un processus $(X_t^m)_{0\le t\le T}$ pour chaque processus de vent $(V_t^m)_{0\le t\le T}$, et en calcule la perte 
$$j^m(\phi^\theta) = \sum_{i=0}^{N-1}\|\phi^\theta(t_i, X^m_{t_i})\|^2 + L(X^m_T),$$
et la perte totale est 
$$J^M(\phi^\theta) = \frac{1}{M}\sum_{m=1}^{M}j^m(\phi^\theta).$$
En employant la décente de gradient, on met à jour les paramètres $\theta$:
$$\theta \leftarrow \theta - \lambda*\frac{\partial J^M}{\partial\theta}.$$

Le point clé est de déterminer les hyper-paramètres, tels le nombre de couches, les tailles de couches, le *learning-rate* etc. 
En testant avec certains paramètres, mon modèle emploie 3 couches cachées, dont les tailles sont 20, 10 et 10 respectivement, et que la fonction d'activation est `leaky-relu`.
