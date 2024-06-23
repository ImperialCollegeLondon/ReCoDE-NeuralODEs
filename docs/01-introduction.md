
# Introduction

## Overview of Project
This ReCoDE Project on Neural Differential Equations will walk you through the theoretical basics of Ordinary Differential Equations (ODEs), specifically in the context of numerical solvers, all the way to Neural Ordinary Differential Equations. The topics we'll cover will include answering what ODEs are, how to implement their numerical solution in research code, how to utilise an autograd equipped library to propagate gradients through a integration routine, and, most importantly, how one can then subsequently use these developed tools to tackle problems with Neural Networks.
While it looks like a lot of ground to cover, a lot of the mathematical machinery should be familiar to you, and hopefully through this project you will learn how to structure a distributable python research module, how to take theoretical ideas like backpropagation through an integrator and apply it to Neural Networks and most importantly, you will have fun.

## Neural Ordinary Differential Equations
Neural Ordinary Differential Equations combines Neural Networks with Ordinary Differential Equations (ODE). Broadly, NNs are a class of models where a (usually statistical) mapping between an input space and an output space can be learned. Differential Equations relate a function and its derivative(s), and ODEs are a subset of Differential Equations where there is only one independent variable.

In less vague terms, Neural Networks (NNs) model functions using a variety of nonlinear transformations, a simple feed-forward model uses affine projections combined with nonlinear transformations whereas a network that processes images may use convolutions. An example of this is a "Digit Classifier" that learns the mapping from images of digits (the input space) to the numeric value of the digit (the output space).

The input and output space could also be in relation to a sequence of some kind. For example, the input space could be a text fragement and the output space could be the distribution of probable letters that can succeed the input. Given an incomplete sentence "The dog jumps ove", a neural network could learn the mapping and predict the letter "r" as the output giving "The dog jumps over".

A more dynamical example is a falling ball. A ball dropped from some height will accelerate towards the ground until impact. Perhaps what we can measure is just the height of the ball at each point in time and we drop many balls from different heights with different masses, maybe even different sizes to create a "Ball-Drop" dataset. This dataset would have some input labels (the height from which the balls is dropped, the mass of the ball, the size of the ball) and output labels (the time of impact). We could train a neural network to predict the time of impact directly, but we could also ask the exact state of the ball at any given time between the drop and the impact? We'd need a huge dataset to get a dense sampling, and we would have to hope that the neural network gives reasonable predictions for times that we don't have the data for.

Instead, what if we learned the dynamics itself, the mapping between the current state and some future state. We could iteratively estimate the state of the ball by first putting in the initial state, getting the next state (prescribed by the data sampling) and then re-inputting this next state back into the network until the network says the ball hit the ground. But we're still restricting to our data sampling, if we have sampled at 0.1s increments, then our network won't know how to predict at 0.05s increments because there is no data. Ideally we'd be able to sample continuously, but in this scheme each finer increment requires increasingly more data to the point where simply storing this data becomes impractical.

This is where Differential Equations come in. Differential Equations (DEs) can be used to model the dynamics of some kind of system, this could be chemical, physical, etc., but ultimately the DEs we're interested in describe the time-evolution of a time-dependent dynamical system in a continuous fashion. Hence, the goal with Neural Ordinary Differential Equations is to learn this dynamical mapping from data and perhaps even control it. We will focus on Ordinary Differential Equations since they are the simplest DEs that can be dealt in a time-continuous fashion and do not have the complexity associated with other DEs such as Partial DEs, Stochastic DEs and Differential-Algebraic Equations (DEs but with algebraic constraints).


### What are Ordinary Differential Equations?
Ordinary Differential Equations are a class of differential equations where the flow is dependent on only a single independent variable[^1]. Formally, an ODE is a differential equation of the following form[^2]:
$$\vec{F}\left(t, \vec{y},\frac{\mathrm{d}\vec{y}}{\mathrm{d}t}, \frac{\mathrm{d}^2\vec{y}}{{\mathrm{d}t}^2},\ldots,\frac{\mathrm{d}^{n-1}\vec{y}}{{\mathrm{d}t}^{n-1}},\frac{\mathrm{d}^n\vec{y}}{{\mathrm{d}t}^n}\right)=0$$ [^3]
This expression looks somewhat daunting given the number of terms so let's simplify the notation slightly. First, let $\vec{y}^{(n)}=\frac{\mathrm{d}^n\vec{y}}{{\mathrm{d}t}^n}$ where we implicitly assume that $t$ is the independent variable. Let's also restrict our attention to explicit ODEs given that many systems commonly take this form and we'll also restrict our attention to first-order systems as higher-order explicit ODEs can be rewritten as first-order systems:
$$\vec{y}^{(1)}=\vec{F}\left(t,\vec{y}\right)$$
In this form, we can more clearly see what an ODE tells us, if we consider $(\vec{y}, t)$ as a point in $\mathbb{R}^k\times\mathbb{R}$, then the ODE gives the flow map $(\vec{y}, t)\mapsto(\vec{y}+\mathrm{d}\vec{y},t+\mathrm{d}t)$. Were we to follow this map we would be given a trajectory in the $\mathbb{R}^k\times\mathbb{R}$ space and it is this trajectory that we aim to compute using the methods described in this section.
### Example System of a Falling Ball
For example, the equation for a falling ball takes the form $h^{(2)}=-g$ where $h(t)$ is the height of the ball and $g$ is standard gravitational acceleration. We can rewrite this as a first-order equation by introducing velocity $v=h^{(1)}$:
$$
h^{(1)} = v \\
v^{(1)} = -g
$$

We can write this more compactly as:
$$
\begin{bmatrix}
h^{(1)} \\
v^{(1)}
\end{bmatrix} = \begin{bmatrix}
v \\
-g
\end{bmatrix}
$$

And if we write $\vec{y} = \begin{bmatrix}
h \\
v
\end{bmatrix}$ and $\vec{y}^{(1)} = \begin{bmatrix}
h^{(1)} \\
v^{(1)}
\end{bmatrix}$ we can see how this notation is equivalent.

You'll note that this information alone is not enough to deduce the values of $h(t)$ and $v(t)$, because we lack boundary conditions, and, more specifically in this case, initial values. The height of the ball will differ depending on if the ball was thrown at the ground and what height it was let go at, namely the values $h(t=0)$ and $v(t=0)$.

### Initial Value Problems (IVPs)
Initial Value Problems are a class of problems where the values of the differential equation are defined at $t=0$, at the "initiation" of the system. There are a separate class of problems called Boundary Value Problems (BVPs) where the values of the system are defined at both $t=0$ and $t=T$ for some $T$ (if $t$ is the independent variable), but these are not the systems of interest to us as they are solved using methods different to the ones found here[^5].
In this work, we are seeking to solve IVPs of the form
$$\vec{y}^{(1)}=\vec{f}\left(t, \vec{y}\right),\quad\vec{y}\left(t=0\right)=\vec{h}\in\mathbb{R}^n$$

## Numerical Integration Techniques
Numerical integration can refer to many different techniques ranging from the Trapezoid Rule (a Quadrature method) to Monte-Carlo methods to Runge-Kutta methods, but broadly, these are all methods to solve integrals of different kinds. Quadrature techniques compute definite integrals where the Boundary Conditions are defined exactly (ie. the values of the function and/or derivative on the integration boundaries are known) by subdividing the domain of integration; unlike ODEs, these integrals are not restricted to one continous dimension. Monte-Carlo methods compute multi-dimensional integrals using a probabilistic approach and solve the issues of poor scaling experienced in Quadrature methods which subdivide $d$-dimensions into $N$ parts leading to $N^d$ points of evaluation. Here we focus solely on ODEs and IVPs.

Many methods exist for numerically integrating differential equations, depending on the type some tools are more suitable than others, but we will entirely restrict ourselves to IVPs. First, let's define some more notation. Let $t\in\left[t_i,t_f\right]$ be the interval of integration such that we restrict our interest to only a subset of the values of $t$[^4] where our initial values are defined as $\vec{y}\left(t=t_i\right)$.
With this machinery setup, we can look at the most simple integration method: The (Explicit/Forward) Euler Method.
### The (Explicit/Forward) Euler Method
If we recall that $\vec{y}^{(1)} = \frac{\mathrm{d}\vec{y}}{\mathrm{d}t}$, then we could consider rewriting the differentials as:
$$\mathrm{d}\vec{y} = \vec{f}\left(t,\vec{y}\right)\mathrm{d}t$$
Now $\mathrm{d}t$ represents an infinitesimal increment of time and given calculus we know that $\mathrm{d}\vec{y}$ represents the infinitesimal change in $\vec{y}$, so if we could somehow start with $\vec{f}\left(t_i,\vec{y}(t=t_i)\right)\mathrm{d}t$ and sum each infinitesimal change, we would be able to estimate the trajectory $\vec{y}$. But this unfortunately doesn't work because computers do not handle infinities and infinitesimals in a way that would allow us to compute the above sum.
So maybe if we relax the equation a little bit[^6], and suppose that actually the infinitesimal change in time is no longer infinitesimal but some measurable delta and make the transformation $\mathrm{d}t\rightarrow\Delta{t}$, then perhaps we could sum the measurable changes $\Delta{y}$ to obtain the trajectory $\vec{y}$.
To simplify the notation, we'll assume that $\Delta{t}=\frac{t_f-t_i}{N},\ N\in\mathbb{N}$ and this will split our interval into evenly sized chunks in which we can obtain the discrete mapping of $\tilde{\vec{y}}\left(t\right)\rightarrow\tilde{\vec{y}}\left(t+\Delta{t}\right)$ where $\vec{y}\left(t\right)\approx\tilde{\vec{y}}\left(t\right)$ and $\tilde{\vec{y}}$ is our discrete approximation to $\vec{y}$ [^7][^15]
So now we have defined a very simple integrator where given some initial values and an interval of integration, we can, in principle compute the trajectory of a system. Writing this method out formally, we have that
$$
\tilde{\vec{y}}_0\left(t_0=t_i\right)=\vec{y}\left(t=t_i\right) \\
\tilde{\vec{y}}_k\left(t_k=t_i+k\Delta{t}\right)=\tilde{\vec{y}}_{k-1}+\vec{f}\left(t_{k-1},\tilde{\vec{y}}_{k-1}\right)\Delta{t},\ k=1,\ldots,N
$$
You will note that the next timestep depends only on the previous timestep in this method and that is why this is called the Explicit or Forward Euler Method.
While this method works, an error analysis using Taylor series shows that the error in this approximation is a linear function of $\Delta{t}$ (ie. an $O\left(h\right)$ algorithm where $h=\Delta{t}$). A computer has finite precision, and so if a step size of $h$ gives the solution to a precision of ${10}^{-2}$ then obtaining a solution with a precision that is machine precision ($\epsilon\sim{10}^{-8}$ for 32-bit floats and $\epsilon\sim{10}^{-16}$ for 64-bit floats), then you would require a step size that is 6 orders of magnitude smaller (or 14 orders for 64-bit doubles) to get to such precision. This translates to ${10}^{6}$ more steps and correspondingly, ${10}^{6}$ times more computation. So while Euler's Method is easy to understand from a variety of perspectives, it has terrible scaling with the step size and while it may handle some systems well, it is not a good general purpose integrator for almost any system.
Aside from errors, these methods also have stability considerations and many systems can show instability when the approximation is too inaccurate. More formally, instability here refers to a tendency for the approximation and the ideal trajectory diverging[^8] Euler's Method is not the most stable method especially for systems which can be described as "stiff". While there is no formal mathematical definition that accounts for all stiff systems, the general behaviour of a stiff system is that a method with a finite region of stability takes excessively small steps relative to the smoothness of the underlying solution[^lambert1991].

### Explicit vs Implicit Methods
You'll recall that the Euler Method above was labelled as being Explicit/Forward, and this was to highlight that there is a variation of Euler's Method where
$$
\tilde{\vec{y}}_0\left(t_0=t_i\right)=\vec{y}\left(t=t_i\right) \\
\tilde{\vec{y}}_k\left(t_k=t_i+k\Delta{t}\right)=\tilde{\vec{y}}_{k}+\vec{f}\left(t_{k},\tilde{\vec{y}}_{k}\right)\Delta{t},\ k=1,\ldots,N
$$
And this defines an algebraic equation given a general $\vec{f}$ where the solution $\tilde{\vec{y}}$ must be found iteratively. This is commonly referred to as the Implicit/Backward Euler Method as the solution $\tilde{\vec{y}}$ is implicitly defined. This method, while having the same error properties, generally exhibits better stability[^butcher2016] especially for stiff systems, but requires significantly more computation to solve.

### Runge-Kutta Methods and Butcher Tableaus
While there exist different methods and approaches than the Forward and Backward Euler Methods, such as linear multi-step methods, we will restrict our attention to Runge-Kutta Methods due to their ease of implementation and conceptually straightforward description.
Let $n$ be the order of a given Runge-Method and $k$ be the number of steps, the aforementioned Forward Euler method would be described as an order $n=1$, $k=1$ (as there is only one step in the approximation) Explicit Runge-Kutta method.
Runge-Kutta methods assume a form where the approximation at each timestep[^9], is given by
$$
\vec{y}\left(t+\Delta{t}\right)=\vec{y}(t)+\Delta{t}\sum_{i=1}^{k}b_i\vec{k}_i
$$
where $\vec{y}$ is henceforth assumed to be in reference to the approximated trajectory (unless stated explicitly), and 
$$
\vec{k}_i=\vec{f}\left(t+c_1\Delta{t},\ \vec{y}(t)+\sum_{j=1}^{k}a_{ij}\vec{k}_j\right)
$$
For explicit methods $a_{ij}=0,\ \forall j\ge i$, while for implicit methods there is no restriction.
Our Explicit Euler Method, in this notation, has only one set of coefficients, namely $a_{11}=0$, $b_1=1$ and $c_1=0$. For the Implicit Euler Method, the coefficients are almost the same except now $a_{11}=1$. 
As you can see, this is a powerful notation for writing out different integration schemes as it will help appropriately track coefficients when we're writing our own integrators.
If we denote the matrix of coefficients $A=\left[a_{ij}\right]$, and the vectors of coefficients $\vec{b}=\left[b_i\right]$ and $\vec{c}=\left[c_i\right]$, then we can write a general Runge-Kutta method as follows
$$
\begin{array}{c|c}
c_1 & a_{11} & a_{12} & \ldots & a_{1k} \\
c_2 & a_{21} & a_{22} & \ldots & a_{2k} \\
\vdots & \vdots & & & \vdots \\
c_k & a_{k1} & a_{k2} & \ldots & a_{kk} \\
\hline
& b_1 & b_2 & \ldots & b_k
\end{array}
$$
referred to as a Butcher Tableau.

## Summary

In summary, we have seen that there are methods of integrating ODEs called Explicit Runge-Kutta methods that approximate the trajectory using local estimates of the state, and we'll be implementing them using their Butcher Tableaus. The goal is to code a pipeline where, starting from data relating to a dynamical system, we can train a neural network to learn the dynamics of a system.

## Appendix 1 - Adaptive vs Fixed-Step

Up to this point we've discussed numerical integrators that use a fixed timestep in order to estimate a given trajectory, but let's consider a 1d function where the solution has two timescales, one long and one short such as $y(t) = y_0 e^{-\alpha t} + \frac{\alpha}{\omega}\cos{\omega t}$ where we'll assume $\alpha=1000$ and $\omega=\frac{\pi}{100}$. The ODE for this system would be $\mathrm{d}_{t} y=-\alpha y_0e^{-\alpha t} - \alpha\sin{\omega t}=\alpha\left[\alpha\cos{\omega t} + \sin{\omega t} - y(t)\right]$ We can see that the solution timescale for the exponential is $\alpha^{-1}\sim {10}^{-3}$ whereas the timescale for the oscillation is $\omega^{-1}\sim 50$. If we were to numerically integrate this using a timestep of $\Delta t>\alpha^{-1}$; when $y_0 >> 0$, we would introduce spurious oscillations in the solution as we'd overestimate the decay in value of $y_0$, if $y_0\sim 0$ then these oscillations would be too small in magnitude and the cosine/sine terms would dominate the solution, and so we would need to choose a timestep in accordance with their oscillation.  
  
Multiple timescales exist in many systems and require careful consideration depending on the system being observed. Ideally, we would resolve all timescales by choosing a "small enough" timestep when $\Delta\vec{y}$ is large and "large enough" when $\Delta\vec{y}$ is small. The idea of "small enough" and "large enough" depend strongly on the system in question, and so we must carefully consider how to measure this while integrating a given system.  
  
One idea, commonly used in Runge-Kutta methods, is the introduction of an auxiliary estimate of the trajectory, ideally one with a higher order error than the baseline method used for estimation. Then, assuming that the system is well-resolved (i.e. our timestep hits the "small/large enough" criterion), the difference between the two estimates of the trajectory should be similarly "small". Thus, one can consider a timestep adaptation strategy that "controls" the discrepancy between the two estimates.

Consider the following Runge-Kutta method that has both a 4th order and a 5th order estimate of the trajectory:  
  
$$  
\begin{array}{c|c}
0 \\
0.2 & 0.2 \\
0.3 & 0.075 & 0.225\\
0.8 & 0.9\bar{7} & -3.7\bar{3} & 3.\bar{5} \\
0.\bar{8} & 2.9526 & -11.5958 & 9.8229 & -0.2908 \\
1 & 2.8463 & -10.\bar{75} & 8.9064 & 0.2784\bar{09} & -0.2735 \\
1 & 0.0911 & 0 & 0.4492 & 0.6510 & -0.3224 & 0.1310 & 0 \\
\hline
& 0.0911 & 0 & 0.4492 & 0.6510 & -0.3224 & 0.1310 & 0 \\
& 0.0899 & 0 & 0.4535 & 0.6141 & -0.2715 & 0.0890 & 0.025
\end{array}
$$  

We see that there are now two rows where before we had one row of $\left[b_i\right]$ coefficients, the first row is the $4^{th}$ ($p^{th}$) order estimate and the second row is the $5^{th}$  ($q^{th}$) order estimate (where $q>p$).[^10] [^bogacki1996].
  
Since the estimate is given by $\vec{y}(t+\Delta t)=\vec{y}(t) + \sum_i\left[b_i\vec{k}_i\right]$, we can see that the estimated error is given by $\vec{\epsilon}(t)=\sum_i\left(\left[(\hat{b}_i-b_i\right)\vec{k}_i\right]$ where $\left[\hat{b}_i\right]$ are the coefficients for the higher order estimate.  

A detailed derivation is in the Appendix in [Derivation of Error Adaptation](#appendix-2---deriving-the-error-adaptation-equations), but suffice to say, we can change the timestep based on the error using the equation:

$$
\Delta t_{new} = \alpha\Delta t_{old} = 0.8\left[\frac{\sigma_a + \sigma_r\Vert\vec{y}(t)\Vert_p}{\Vert\vec{\epsilon}(t)\Vert_p}\right]^{1/(p+1)}\Delta t_{old}
$$

where $\sigma_a$ is the absolute error tolerance and $\sigma_r$ is the relative error tolerance.

## Appendix 2 - Deriving the Error Adaptation Equations
  
The error is for each component of $\vec{y}$ with a sign that indicates over-/under-estimation relative to the higher-order estimate. Generally, we are concerned with keeping the _magnitude_ of the error small and are not particularly concerned about over-/under-estimation since, if the magnitude is sufficiently small, the sign becomes irrelevant. Noting that the timestep is a scalar quantity, we must derive a scalar measure of the error from this vector estimate. There are several ways of calculating the magnitude of a vector such as the $L^1$ norm, the $L^2$ norm and the $L^\infty$ norm. By writing these as $L^p$-norms, we can leave the choice of $p$ until later and first write down how we can adjust the timestep.  
  
Consider a trajectory where a scalar $y(t)\sim 1$, an error of ${10}^{-2}$ is a 1% deviation from the ground truth. If this was an estimate of distance in meters, the magnitude of the error would be on the order of centimeters, a measurable and noteworthy quantity. Now let's consider a change of units from meters to nanometers, our trajectory now becomes $y(t)\sim {10}^{9}$. An error of ${10}^{-2}$ would now be on the order of one hundredth of a nanometer, for a system where the changes occur on the order of meters, this would be an insignificant change. If our units were kilometers, then $y(t)\sim {10}^{-3}$ and our error is larger than the quantities we're estimating. If we consider the problem in reverse, targeting an error of ${10}^{-2}$ would be unnecessarily precise (except in perhaps rare circumstances) or critical depending on the scale of the problem.  
  
This motivates the introduction of two concepts: absolute tolerance (\sigma_a) and relative tolerance (\sigma_r). The absolute tolerance is the maximum deviation we would allow irrespective of the scale of the problem, whereas the relative tolerance is the maximum deviation we would allow normalised to the scale of the problem. [^11]  
  
Going back to our error estimate, the ideal condition we would like to fulfil is the following (Loosely following the derivation in [^shampine2005]):
  
$$  
\epsilon_i(t) \le \sigma_a + \sigma_ry_i(t),\ \forall i  
$$  
  
where the subscript $i$ denotes the vector components and where we've introduced the scale of the problem through multiplying $\sigma_r$ by $y_i(t)$. We can rearrange this equation to obtain the following:  
  
$$  
\frac{\epsilon_i(t)}{\sigma_a + \sigma_ry_i(t)} \le 1  
$$  
  
And if we consider that the error $\epsilon_i$ is actually an estimate of the higher-order error term (i.e. the term $O({\Delta t}^p)$ in the Taylor expansion of the solution), then we can write the ratio as follows  
  
$$  
\frac{{\Delta t}^{p+1}\phi_i(t, \vec{y}(t))+O({\Delta t}^{p+1})}{\sigma_a + \sigma_ry_i(t)} \le 1  
$$  
  
Where we know that the leading error term is proportional to ${\Delta t}^{p+1}$ and $\phi_i(t,\vec{y}(t))$ (which in turn is dependent on the p+1 derivative of the solution). From this we see the relationship between the tolerances we'd like to achieve and the timestep, but there are still unknown terms and an inequality is not easy to solve. 

Consider now that we change the step by a factor $\alpha$ to $\alpha\Delta t$, then the error estimate of the step will be:

$$  
\frac{{(\alpha\Delta t)}^{p+1}\phi_i(t, \vec{y}(t))+O({(\alpha\Delta t)}^{p+1})}{\sigma_a + \sigma_ry_i(t)} \le 1  
$$ 

Solving for $\alpha$ (to the leading order term) we get:

$$  
\alpha^{p+1}  \le \frac{\sigma_a + \sigma_ry_i(t)}{{\Delta t}^{p+1}\phi_i(t, \vec{y}(t))+O({(\alpha\Delta t)}^{p+1})}
$$
$$
\alpha  \le \left[\frac{\sigma_a + \sigma_ry_i(t)}{{\Delta t}^{p+1}\phi_i(t, \vec{y}(t))+O({(\alpha\Delta t)}^{p+1})}\right]^{1/(p+1)}
$$[^12]

If we assume that the higher order terms, $O({(\alpha\Delta t)}^{p+1})$, are similar to the higher order terms of the unaltered timestep $O({\Delta t}^{p+1})$, then we can write substitute back our estimate of the error to obtain:

$$
\alpha  \le \left[\frac{\sigma_a + \sigma_ry_i(t)}{\epsilon_i(t)}\right]^{1/(p+1)}
$$

Which gives us a way to compute an adjustment to the timestep that obeys our tolerance criteria for each component of our trajectory. In common usage, a safety factor of approximately $0.8$ is incorporated to get the the stricter inequality:

$$
\alpha  \le 0.8\left[\frac{\sigma_a + \sigma_ry_i(t)}{\epsilon_i(t)}\right]^{1/(p+1)}
$$

That accounts for any spurious changes in the solution (i.e. mild violations of the equality of the higher order terms between the current step and the new step).

Whenever our original error does not satisfy our inequality, we can reject the step and recalculate it more accurately using a timestep $\alpha\Delta t$ from the following formula. Conversely, whenever the step is accepted, we can use $\alpha\Delta t$ to increase the step and avoid extra computation in solving the system.

$$
\Delta t_{new} = \alpha\Delta t_{old} = 0.8\left[\frac{\sigma_a + \sigma_ry_i(t)}{\epsilon_i(t)}\right]^{1/(p+1)}\Delta t_{old}
$$

[^13]

Up to here we have ignored the fact that $\vec{y}$ is a vector quantity and our inequality is on a per-component basis. Since we cannot take separate steps for each component[^14], we need to make an adjustment that accounts for all our components simultaneously. One way would be to take some kind of average over the different $\alpha$'s for each component such the arithmetic or geometric mean. Another might be to consider the maximum or minimum, but a simple solution is to convert the error into a scalar over all the components:

$$
\Delta t_{new} = \alpha\Delta t_{old} = 0.8\left[\frac{\sigma_a + \sigma_r\Vert\vec{y}(t)\Vert_p}{\Vert\vec{\epsilon}(t)\Vert_p}\right]^{1/(p+1)}\Delta t_{old}
$$

Where we now consider the magnitude of our errors in a vector sense. The p-norm, discussed before, is left up to the developer's choice and can even be exposed as part of the integration interface. A suitable choice would be the $\infty$-norm for the denominator and the $2$-norm for the numerator as this adjusts the error proportional to the maximum error incurred while considering the overall scale of the problem.

[^1]: In many systems in physics and chemistry, this is time.
[^2]: I've used $t$ here since many systems we will look at use time as the independent variable, but this could also be $x$ which usually denotes a spatial dimension meaning that $y(x)$ is some spatially varying function. Mathematically they are equivalent, but I will follow the physics conventions in this project.
[^3]: This expression encompasses both explicit and implicit ODEs, where it may be the case that $\frac{\mathrm{d}^n\vec{y}}{{\mathrm{d}t}^n}$ cannot be isolated and thus one cannot find an explicit closed-form expression for it.
[^4]: Suppose we want to know what a system does in the limit $t\rightarrow\infty$, then a computer cannot numerically represent this interval in a useful fashion. In that case, we can use a suitable change of variables such that $t(u)=g(u)$ where $u=g^{-1}(t),\ u\in\left[g^{-1}(t_i), g^{-1}(t_f)\right]$ and integrate as a function of $u$.
[^5]: In fact, I've omitted an entire class of problems with mixed boundary conditions where both initial values and boundary values are specified depending on the components of $\vec{y}$. These are more common in ODE systems derived from partial differential equations where, for example, temperature diffusion through a metal plate may occur where one end of the plate is always at a fixed temperature.
[^6]: And here you should be getting echoes of Riemannian Sums.
[^7]: If we could somehow take the limit $\Delta{t}\rightarrow0$ then we would have an exact solution to our system.
[^8]: You may ask, how would you measure that if these systems cannot be solved analytically. One method involves linearising some ideal system of equations and looking at how that linear mapping behaves when composed many times while accounting for the error terms introduced by the approximation. In the ideal case, these error terms increase as a linear or sub-linear function of the timestep, but in the unstable case, these may increase in a super-linear or even exponential fashion leading to divergence from the true trajectory.
[^9]: Timestep will denote steps in $\Delta{t}$ whereas step will denote steps taken to approximate the timestep
[^10]: The coefficients are normally given as fractions, but for brevity they have been rounded to 4 decimal places or the repeating portion of the decimal has been indicated with a bar.
[^11]: While we've discussed these tolerances as scalar quantities, one could consider specifying them on a per component basis and we will come back to this at a future point.
[^12]: This is valid as all quantities are strictly positive and exponentiation is a monotonic function for positive quantities.
[^13]: It should be noted that there are other methods of choosing the step size adjustment during a numerical integration procedure including estimation of the global error (here we have only considered the local error incurred in taking one step, but not the cumulative error) and the error per unit step.
[^14]: there are methods that take fractional steps for some quantities, but these are beyond the scope of this work and require significant work/bookkeeping to correctly implement
[^15]: There are alternative ways of deriving the same result, this is a loose derivation based on intuition, but a more formal approach would be to first take the integral of both sides wrt. $\mathrm{d}t$ from $t_i$ to $t_f$ to obtain $\vec{y}(t_f)-\vec{y}(t_i) = \int_{t_i}^{t_f}\vec{f}\left(t, \vec{y}(t)\right)\mathrm{d}t$ and then approximate the integral on the RHS using known quadrature rules such as the Rectangle Rule.
[^butcher2016]: Butcher, J. C. (John C. (2016). _Numerical methods for ordinary differential equations_ (Third edition.). John Wiley & Sons, Ltd.
[^lambert1991]: Lambert, J. D. (1991). _Numerical methods for ordinary differential systems : the initial value problem_. Wiley.
[^bogacki1996]: Bogacki, P., & Shampine, L. F. (1996). An efficient Runge-Kutta (4,5) pair. In Computers &amp; Mathematics with Applications (Vol. 32, Issue 6, pp. 15–28). Elsevier BV. <https://doi.org/10.1016/0898-1221(96)00141-1>
[^shampine2005]: Shampine, L. F. (2005). Error estimation and control for ODEs. In Journal of Scientific Computing (Vol. 25, Issues 1–2, pp. 3–16). Springer Science and Business Media LLC. <https://doi.org/10.1007/bf02728979>
