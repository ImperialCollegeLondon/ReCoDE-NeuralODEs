# The Adjoint Method
The adjoint method is a way of computing the derivatives associated with a numerical integration procedure. Take the Simple Harmonic Oscillator (SHA) which has two parameters: frequency and damping. 

Let's say we integrated the SHA system from $t_0=0$ to $t_1=T$ with initial state $\vec{x}(t_0)=[1, 0]$ with frequency $\omega=1.0$ and damping $\zeta=0.25$. The parameters of this integration are $t_0$, $t_1$, $\vec{x}(t_0)$, $\omega$ and $\zeta$, and suppose we wanted to know how $\vec{x}(t_1)$ changes wrt. (with respect to) each of these parameters. Formally, we would like to know the following gradients,
$$
\frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_0}, \frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_1}, \frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}\vec{x}(t_0)}, \frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}\omega}, \frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}\zeta}, 
$$
which are the sensitivities of the system.

While we can compute these using various approximations, the adjoint method provides an exact way to compute these derivatives with minimal memory cost and sometimes computational cost.

As we are interested in Neural ODEs, in many cases, these will not be the derivatives of $\vec{x}(t_1)$ directly, but the derivatives of some loss function $L(\vec{x}(t_1))$.

If the task were to reconstruct the frequency and damping of an SHA system, starting from $\left\{\vec{y}(t_i)\right\}_i$ which are samples of the system at times $\left\{t\right\}_i$, then the loss function may simply be $L=\sum_i\left|\vec{x}(t_i)-\vec{y}(t_i)\right|^2$. This is simply the sum of the errors at each discrete point in time specified by our set of samples.

Then we'd be looking at the derivatives $\frac{\mathrm{d}L}{\mathrm{d}\omega}$ and $\frac{\mathrm{d}L}{\mathrm{d}\zeta}$.

These derivatives are important as we can use them to update our parameters using a gradient descent scheme like:

$$
\omega_{n+1}=\omega_n - \alpha\frac{\mathrm{d}L}{\mathrm{d}\omega}
$$
with step-size $\alpha$ and says to update our value of $\omega$ such that it decreases $L$.

These relate to our integrator derivatives because we can rewrite them as
$$
\frac{\mathrm{d}L}{\mathrm{d}\omega}=\frac{\partial L}{\partial \vec{x}(t_i)}\frac{\mathrm{d}\vec{x}(t_i)}{\mathrm{d}\omega}
$$
allowing us to use the derivatives computed from the integrator to infer the derivatives of our error function.

The simplest derivation are the derivatives $\frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_0}$ and $\frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_1}$. If we explicitly write out $\vec{x}(t)$ as
$$
\vec{x}(t_1) = \vec{x}(t_0) + \int_{t_0}^{t_1}x^{(1)}(x(t^\prime), t^\prime, \Theta)\mathrm{dt^\prime}
$$
then the solution of this is equivalent to 
$$
\vec{x}(t_1) = \vec{x}(t_0) + \left[\vec{F}(t_1) - \vec{F}(t_0)\right]
$$
(omitting some of the dependencies of $\vec{F}$).

By the Fundamental Theorem of Calculus, the derivative 
$$
\frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_0} = \frac{\mathrm{d}\vec{F}(t_0)}{\mathrm{d}t_0} = -x^{(1)}(x(t_0), t_0, \Theta)
$$
and
$$
\frac{\mathrm{d}\vec{x}(t_1)}{\mathrm{d}t_1} = \frac{\mathrm{d}\vec{F}(t_1)}{\mathrm{d}t_1} = x^{(1)}(x(t_1), t_1, \Theta)
$$.

For the other derivatives, the adjoint method is necessary as it gives a general procedure for calculating how the variation evolves over the integration. As the derivation of the adjoint method is quite involved, we will not be covering it here. It requires an understanding of not only advanced calculus such as the Leibniz Rule, but also the Calculus of Variations/Lagrange variables. For further reading, the following references should provide a starting point for understanding the adjoint method.

[1] https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf
[2] https://indico.flatironinstitute.org/event/101/contributions/680/attachments/126/230/adjoint_methods.pdf
[3] https://www.sciencedirect.com/science/article/pii/S0377042702005289
[4] https://math.mit.edu/~stevenj/18.336/adjoint.pdf
