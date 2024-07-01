# Cart-Pole System
The cart-pole system (or inverted pendulum) is a well-known control problem involving balancing a pendulum attached to a cart. The usual dynamical model involves a rigid pendulum attached to the cart on a rotating pin such that the pendulum can swing $360\degree$ and the cart can move linearly along one dimension.

The image below illustrates the exact setup more clearly.
[ INSERT IMAGE HERE ]

This system is difficult to control due to the chaotic dynamics of the system arising from the coupling between the cart and the pole, and the many local minima. A very common controller is a Proportional-Integral-Derivative (PID) Controller which uses the error along with its time-variation as well as the cumulative error to control the cart and align the pendulum vertically.

While a PID controller works quite well, its parameters are fixed, may not adapt well to different environments (e.g. change in mass of the cart/pole), and requires tuning the parameters to get good results. Additionally, it may not be the most optimal in terms of time/energy. For these reasons, other controllers have been developed ranging from optimal control theory based, mathematically exact controllers to Reinforcement Learning controllers where the network has no idea what the dynamics are and has to infer a control policy from sparse rewards.

Our network will be somewhere in the middle where we assume access to the dynamics and can propagate gradients through them. Our network, at each timestep will specify the control force (essentially whether to push the cart left or right, and by how much), and this will be numerically integrated with the dynamics.

The end state is ideally the vertically balanced state and so our loss function will be the angle deviation from zero, the angular velocity deviation from zero, and a small portion will be the distance travelled by the cart (to penalise the network transporting the cart several kilometers in either direction) and the deviation from the target state in the intermediate steps (to encourage achieving the balanced state earlier).
