
# The Simple Harmonic Oscillator  
  
The Simple Harmonic Oscillator (SHA) is a common system that arises in the study of many physical systems, a spring-mass system being the archetypal example.  
  
In this page, we'll go through the features and details of the SHA system and describe some of the variations of it that are relevant to the notebooks shown in this repository.  
  
While the SHA is written as a second-order dynamical system (conventionally, the dynamical equations of the forces acting on a mass due to a spring), we can rewrite in first order form as seen in the introduction. Using the same state notation where $x$ is the displacement and $v$ is the velocity we can write the equations as
$$  
\begin{bmatrix}  
    x^{(1)} \\  
    v^{(1)}  
\end{bmatrix} =   
\mathbf{A}  
\begin{bmatrix}  
    x \\  
    v  
\end{bmatrix}  
$$
where  
$$  
\mathbf{A} =  
\begin{bmatrix}  
    0 & 1 \\  
    -\omega^2 & -2\zeta\omega  
\end{bmatrix}  
$$, $\omega$ is the natural frequency of oscillation of the system and $\zeta$ is the damping of the system. The above shows that the system is linear and homogeneous (independent of time).

Furthermore, we can add a driving force to this system to obtain the damped-driven oscillator where the system can be actuated with some driver. This simply changes the equations to
$$  
\begin{bmatrix}  
    x^{(1)} \\  
    v^{(1)}  
\end{bmatrix} =   
\mathbf{A}  
\begin{bmatrix}  
    x \\  
    v  
\end{bmatrix}  + \begin{bmatrix}
0 \\
F(\vec{x}, t)
\end{bmatrix}
$$
where we've explicitly restricted the actuation to only affect the acceleration (derivative of velocity) and not the velocity itself. This is done with the intention of physicality where, a mass-spring system, can only be interacted with through forces applied to it and not through directly setting the velocity or position once it's going.

References:
[1] https://en.wikipedia.org/wiki/Harmonic_oscillator
