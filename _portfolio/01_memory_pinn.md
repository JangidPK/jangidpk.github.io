---
title: "Memory-Aware Physics-Informed Neural Networks for Non-Markovian Diffusion"
excerpt: "A deep learning project implementing Physics-Informed Neural Networks (PINNs) to solve generalized diffusion equations with temporal memory,
 where the governing dynamics depend on the entire history of the system rather than its instantaneous state.
 <br/><img src='/images/memory_pinn.png' style='max-width:600px; width:100%;'>"
collection: portfolio
---

## 1. Overview

This project implements a **Physics-Informed Neural Network (PINN)** that solves a **non-Markovian (memory-dependent) diffusion equation**:

$$ \frac{∂c(x,t)}{∂t} =  \int_0^t K(t - s) · D · ∇^2 c(x,s) ds$$

Unlike ordinary diffusion, where the rate of change at time $t$ depends only on the instantaneous state, this equation says that, the rate of change depends on the **entire history** of the concentration field, weighted by a **memory kernel** $$K(t-s)$$. This is a discrete/numerical analogue of the kind of memory effects seen in non-Markovian stochastic processes and generalized diffusion models.

The network jointly (or optionally separately) learns:

- $$c(x,t)$$ - the concentration field, represented by a feed-forward "solution network."
- $$K(Δt)$$ - the memory kernel, represented by a second, independent feed-forward "kernel network," learned directly from data rather than assumed to have a fixed functional form.

Both networks are trained using a composite loss that combines the PDE residual (evaluated via automatic differentiation and numerical quadrature), initial/boundary condition constraints, and a data-fitting term against sparse, optionally noisy observations.

The project deliberately favors **clarity over performance**. Every numerical method used (finite differences, trapezoidal quadrature) is implemented by hand: no external PINN libraries and no adaptive or higher-order numerical schemes.

---
## 2. Introduction
Physics-informed neural networks have been successfully extended to solve a wide class of nonlocal and memory-dependent partial differential equations [1-4]. In particular, several studies have investigated diffusion equations involving fractional derivatives or temporal memory operators, where the governing equations depend on the past evolution of the system rather than only the current state. These works demonstrate that PINNs can naturally incorporate nonlocal operators into the physics loss, making them well suited for anomalous diffusion and other non-Markovian transport problems. The present tutorial follows the same general ideas but focuses on a simple implementation for generalized diffusion equations with convolution-based memory kernels.




## 3. Mathematical Background

### 3.1 Ordinary diffusion (the warm-up problem)
Diffusion describes the spreading of particles due to random thermal motion.
 At sufficiently long time scales, when particle displacements are uncorrelated,
  the evolution of the probability density follows the classical diffusion equation. 
  This equation provides the reference model against which more complex transport
   processes with memory effects can be compared. The probability density $$c(x,t)$$ 
   evolves according to

   
$$ \frac{\partial c(x,t)}{\partial t} = D  \frac{\partial^2c(x,t)}{ \partial x^2} $$.

where $$D$$ is the diffusion coefficient. This equation has initial condition $$c(x,0) = f(x)$$. Further, based on the domain, $$x\in[a,b]$$, of the problem there will be boundary conditions, such as 

*  Dirichlet: $$c(a,t)=g_1(t), c(b,t)=g_2(t)$$,
*  Neumann: $$c_x(a,t)=h_1(t), c_x​(b,t)=h_2(t),$$

With a Gaussian pulse initial condition and zero-concentration (Dirichlet) boundary conditions, the general solution for an unbounded domain $$(−\infty<x<\infty)$$ can be written as


$$c(x,t) = \frac{1}{\sqrt{4 \pi D t}} 
\int_{-\infty}^{\infty} f(\xi) 
\exp \left( - \frac{(x-\xi)^2}{4Dt} \right)d\xi .
$$

For a bounded domain, the solution is usually written as an eigenfunction expansion

$$c(x,t)=  \sum_{n=1}^{\infty} A_n \phi_n(x)
e^{−Dλ_nt}$$ 

where $$\phi_n(x)$$ are eigenfunctions satisfying
$$\phi_n''+\lambda_n \phi_n=0$$
together with the specified boundary conditions, $$A_n$$ are determined by the initial condition. 

In the limit of infinite domain, $$-\infty < x < \infty$$, and with initial condition as a Dirac delta function.

$$c(x,0) = \delta(x - x_0)$$, then the solution is the fundamental solution (or Green's function) of the diffusion equation, written as 

$$c(x,t) = \frac{1}{\sqrt{4 \pi D t}}  
\exp \left( - \frac{(x-x_0)^2}{4Dt} \right).
$$

 Physically, this represents an instantaneous point release of one unit of mass at position $$x=0$$. This is the classical PINN benchmark: a PINN is trained to satisfy the PDE residual $$R = c_t - D·c_{xx} = 0$$ at a set of collocation points, plus penalty terms for the initial and boundary conditions. Such problems have been studies in the ref. [5-9]

### 3.2 Memory (non-Markovian) diffusion
The classical diffusion equation assumes that the diffusion process is Markovian, meaning that the evolution of the concentration at time $$t$$ depends only on its current state and not on its past history. In many physical, biological, and engineering systems, this assumption is not valid. Examples are diffusion in porous media, transport in viscoelastic materials, intracellular transport, and heterogeneous geological formations, where particles may become temporarily trapped or experience long waiting times before moving again. Such systems show memory effects, and the diffusion process is referred to as non-Markovian diffusion. In such cases the target equation replaces the instantaneous Laplacian term with a **convolution over time**:


$$
\frac{\partial c (x,t)}{\partial t} = \int_0^t K(t-s) · D · \nabla^2 c(x,s) ds, 
$$

where the diffusive flux at time $$t$$ isn't just a function of the current curvature
 of the concentration field, it's a weighted sum (memory) of the curvature
  at *every past time*, with weights given by the kernel
   $$K(t)$$. If $$K(t) = δ(t)$$ (a Dirac delta), the equation reduces to ordinary 
   diffusion. One commonly used kernel is the exponential kernel,
    $$K(t) = \frac{1}{\tau}e^{-t/τ}$$, here more recent history matters more than
     distant history, with a decay timescale $$τ$$.

This project uses $$K(t) = e^{-t}$$ (note here $$\tau=1$$) as the **known ground-truth kernel** for generating synthetic data, while the kernel network is capable of learning arbitrary shapes.

### 3.3 Memory integral is treated as a discrete convolution


Unlike the classical diffusion Markovian equation, the non-Markovian equation accumulates the contribution of all previous states. For the one-dimensional equation, the memory term is 

$$
I(x,t) = \int_0^{t} K(t-s) L(x,s)ds
$$

where $$L(x,s) = D \nabla^2c(x,s)$$, represents the spatial diffusion operator.
 The integral cannot, in general, be evaluated analytically because the concentration
  field $$c(x,t)$$ is unknown during training. Consequently, the memory term must be
   approximated numerically. 


Assume that the time interval is discretized into $$N+1$$ equally spaced points,


$$
t_0,t_1,\ldots,t_N,  
$$

with

$$\Delta t=t_{k+1}-t_k.  
$$

Since every spatial point $$x$$ shares the same temporal discretization, the memory 
integral at the current time $$t_j$$ is approximated using a quadrature rule, 

  
$$
I(x,t_j) =  
\int_0^{t_j}   K(t_j-s)   L(x,s),ds  
\approx   \sum_{k=0}^{j}   w_k   K(t_j-t_k)   L(x,t_k),  
$$

where $$w_k$$ are the quadrature weights. Then, using the trapezodial rule,

$$
w_k = 
\begin{cases}
    \Delta t /2 & \text{if } k = 0, j \\
    \Delta t& \text{if }  1\le k\le j-1, \\
\end{cases}
$$

The approximation replaces the continuous time integral with a finite weighted sum over
 all previous time levels. At every current time $$t_j$$, the diffusion operator is 
 evaluated not only at the present instant but also at every previous time level.
  The contribution from each previous state is multiplied by the kernel value 
  $$K(t_j-t_k)$$, which measures the influence of the state at time $$t_k$$ on the
   current solution. 

An important observation is that the kernel depends only on the **time difference**, $$t_j-t_k$$, rather than on the individual values of $$t_j$$ and $$t_k$$. Consequently, the coefficients of interest form the sequence

$$
K(0),  
K(\Delta t),  
K(2\Delta t),  
\ldots,  
K(j\Delta t),  
$$

which is multiplied by the corresponding history

$$ 
L(x,t_j),  
L(x,t_{j-1}),  
\ldots,  
L(x,t_0).  
$$

This operation is precisely the definition of a **discrete convolution** between two sequences. Then we can write

$$
I_j(x)
= \sum_{k=0}^{j}  
w_k  
K_{j-k}  
L_k(x), 
$$


where

$$
K_{j-k} = 
K(t_j-t_k),  
\qquad  
L_k(x)  = D\frac{\partial^2c(x,t_k)}{\partial x^2}.  
$$

For every spatial location $$x$$, the computation proceeds independently along the temporal axis. Suppose the neural network predicts the concentration

$$
c(x,t_0),  
c(x,t_1),  
\ldots,  
c(x,t_N).  
$$

Automatic differentiation is first used to compute

$$
L(x,t_0),    
L(x,t_1),  
\ldots,  
L(x,t_N), 
$$

after which the discrete convolution with the kernel is performed to obtain

$$ 
I(x,t_0),   
I(x,t_1),  
\ldots,  
I(x,t_N).
$$

Since the same kernel is applied to every spatial point, the operation is highly structured and can be implemented efficiently using vectorized matrix operations. Rather than evaluating a separate numerical integral for every collocation point, the entire history term is obtained through a single convolution operation over the time dimension.

From a computational viewpoint, this interpretation greatly simplifies the implementation of non-Markovian PINNs. The memory integral becomes a standard numerical convolution, allowing efficient implementation on modern GPU architectures while preserving the physical interpretation of temporal memory.


### 3.4 The PDE residual

As in conventional PINNs, our objective is to train a neural network that satisfies the governing partial differential equation together with the prescribed initial and boundary conditions. Instead of fully relying on labeled simulation data, the network learns by minimizing the residual of the governing equation at a set of collocation points distributed throughout the computational domain.

$$
R(x,t) = \frac{\partial c(x,t)}{\partial t} - 
\int_0^t   K(t-s), D \nabla^2c(x,s)ds.  
$$

The exact solution satisfies, $$R(x,t)=0$$, for every point in the space-time domain. Since exact $$c(x,t)$$ is unknown, the neural network provides an approximation

$$
c_\theta(x,t),  
$$

where $$\theta$$ denotes the trainable network parameters. During training, the network is evaluated at a collection of collocation points,

$$ 
(x_i,t_i),  
\qquad  
i=1,\ldots,N_c, 
$$

and the residual is computed at each point. The temporal derivative is obtained directly through automatic differentiation,

$$
\frac{\partial c_{\theta}}{\partial t}
=\frac{\partial}{\partial t}  
c_\theta(x,t),  
$$

while the spatial Laplacian is computed as

$$
\nabla^2c_\theta
=
\frac{\partial^2c_\theta}{\partial x^2}  
$$

These derivatives are evaluated exactly through PyTorch automatic differentiation, eliminating the need for finite-difference or finite-element approximations of the differential operators. Since the neural network is differentiable with respect to its inputs, higher-order derivatives can be computed directly using repeated applications of the automatic differentiation engine.


The only term that cannot be obtained directly through differentiation is the history integral. Once the Laplacian has been evaluated at every discretized time level,

$$  
L(x,t_k)
=
D\nabla^2c_\theta(x,t_k),  
$$

the memory contribution is computed using the trapezoidal approximation (as discussed in previous section),

$$
\int_0^{t_j}  
K(t_j-s)  
D\nabla^2c(x,s),ds  
\approx  
\sum_{k=0}^{j}  
w_k  
K(t_j-t_k)  
D\nabla^2c(x,t_k).  
$$

As discussed in the previous section, this summation is implemented as a discrete convolution along the temporal dimension. The residual evaluated at time $$t_j$$ therefore becomes

$$
R(x,t_j)
=
\frac{\partial c_\theta(x,t_j)}{\partial t}
- \sum_{k=0}^{j}  
w_k  
K(t_j-t_k)  
D\nabla^2c_\theta(x,t_k).  
$$

The network parameters are optimized by minimizing the mean-squared residual over all collocation points,

$$  
\mathcal{L}_{\mathrm{PDE}}
=
\frac{1}{N_c}  
\sum_{i=1}^{N_c}  
R(x_i,t_i)^2.  
$$

This term represents the physics loss arising from the governing differential equation.


## 4.  Learning objective

The neural network is trained by minimizing a weighted combination of the physics and data constraints,

$$
\mathcal{L}
=
\lambda_{\mathrm{PDE}}\mathcal{L}_{\mathrm{PDE}}
+
\lambda_{\mathrm{IC}}\mathcal{L}_{\mathrm{IC}}
+
\lambda_{\mathrm{BC}}\mathcal{L}_{\mathrm{BC}}
+
\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}},
$$

where each loss term enforces a different physical or data-driven constraint during training.

The physics loss, $$\mathcal{L}_{\mathrm{PDE}}$$, is computed from the residual of the governing diffusion equation, as described in the previous section.

The initial condition loss ensures that the network satisfies the prescribed initial distribution,

$$
\mathcal{L}_{\mathrm{IC}}
=
\frac{1}{N_{\mathrm{IC}}}
\sum_{i=1}^{N_{\mathrm{IC}}}
\left|
c_{\theta}(x_i,0)-c(x_i, 0)
\right|^2,
$$

where $$f(x)$$ denotes the prescribed initial concentration profile.

Similarly, the boundary condition loss penalizes deviations from the prescribed boundary conditions,

$$
\mathcal{L}_{\mathrm{BC}} 
= 
\frac{1}{N_{\mathrm{BC}}}
\sum_{i=1}^{N_{\mathrm{BC}}} 
\left | c_{\theta}(a,t_i)-g_1(t_i) \right|^2
+
\frac{1}{N_{\mathrm{BC}}}
\sum_{i=1}^{N_{\mathrm{BC}}} 
\left | c_{\theta}(b,t_i)-g_2(t_i) \right|^2,
$$


where $$g_1(t)$$ and $$g_2(t)$$ denote the prescribed Dirichlet boundary values at $$x=a$$ and $$x=b$$, respectively. And, $$N_{\mathrm{IC}}$$, $$N_{\mathrm{BC}}$$ are selected grid sizes along space and time dimensions, respectively.

Finally, when reference solution data are available, a supervised data loss is included using the mean squared error,

$$
\mathcal{L}_{\mathrm{data}}
=
\frac{1}{N_{\mathrm{data}}}
\sum_{i=1}^{N_{\mathrm{data}}}
\left|
c_{\theta}(x_i,t_i)-c_{\mathrm{data}}(x_i,t_i)
\right|^2.
$$

The weighting coefficients $$\lambda_{\mathrm{PDE}}$$, $$\lambda_{\mathrm{IC}}$$, $$\lambda_{\mathrm{BC}}$$, and $$\lambda_{\mathrm{data}}$$ control the relative contribution of each loss term during training.

In short:

- $$\mathcal{L}_{\mathrm{data}}$$ penalizes the mismatch between the predicted and observed concentration data.

- $$\mathcal{L}_{\mathrm{PDE}}$$ penalizes violations of the governing non-Markovian diffusion equation.
    
- $$\mathcal{L}_{\mathrm{IC}}$$ enforces the prescribed initial condition, and
    
- $$\mathcal{L}_{\mathrm{BC}}$$ enforces the boundary conditions.
    

Unlike the classical diffusion equation, where the residual depends only on the local values of the solution and its derivatives at $$(x,t)$$, the residual for non-Markovian diffusion depends on the **entire temporal history** of the Laplacian through the convolution term. Consequently, the prediction at one time level contributes to the residual at all subsequent time levels. This coupling across the temporal domain is the defining feature of non-Markovian PINNs and is responsible for both their increased computational complexity and their ability to model diffusion processes with long-term memory.


## 5. Numerical Solution Strategy

The following table summarizes the numerical methods used in this project.

| Quantity | Method |
|----------|--------|
| Ordinary diffusion  | Finite-difference discretization |
| Non-Markovian diffusion | Finite-difference discretization with temporal convolution |
| Memory integral | Trapezoidal quadrature implemented as a discrete convolution |
| Solution approximation | Physics-Informed Neural Network |
| Memory kernel | Feed-forward neural network |

This tutorial presents a minimal implementation of a memory-aware physics-informed neural network for solving generalized diffusion equations with temporal memory. The emphasis is on clarity rather than computational efficiency, making the code suitable as a starting point for researchers interested in extending PINNs to more general non-Markovian transport models, inverse problems, or learned memory kernels.


## Repository

GitHub Code:
[Link](https://github.com/JangidPK/memory-aware-pinn/)


# References

1. Hu, Z., Kawaguchi, K., Zhang, Z., & Karniadakis, G. E. (2024). Tackling the curse of dimensionality in fractional and tempered fractional PDEs with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 432, 117448. https://doi.org/10.1016/j.cma.2024.117448

1. Dabiri, D., DaRosa, J., Saadat, M., Mangal, D., & Jamali, S. (2024). A Detailed and Comprehensive Account of Fractional Physics-Informed Neural Networks: From Implementation to Efficiency. https://doi.org/10.2139/ssrn.5004046

1. Guo, L., Wu, H., Yu, X., & Zhou, T. (2022). Monte Carlo PINNs: deep learning approach for forward and inverse problems involving high dimensional fractional partial differential equations. Arxiv (Cornell University). https://doi.org/10.1016/j.cma.2022.115523

1. Wu, Q., Ma, F., & Wo, W. (2025). Fs‐Pinns: Fractional Spectrally Adapted Physics‐Informed Neural Networks for Fractional Partial Differential Equations. Mathematical Methods in the Applied Sciences, 48(14), 13882–13889. https://doi.org/10.1002/mma.11149

1. Ozeir, H., Bazzi, A. M., Owayjan, M., & Nuwayhid, R. (2025). Physics Informed Neural Networks to Solve the 2-D Heat Diffusion Equation. 1–5. https://doi.org/10.1109/actea66485.2025.11189901

1. Hasan, F., Ali, H., & Arief, H. A. (2025). From Mesh to Neural Nets: A Multi-method Evaluation of Physics Informed Neural Network and Galerkin Finite Element Method for Solving Nonlinear Convection–Reaction–Diffusion Equations. International Journal of Applied and Computational Mathematics, 11(3). https://doi.org/10.1007/s40819-025-01904-y

1. Garcia, S., & Miller, S. (2019). arXiv. American Mathematical Society Ebooks, 433–437. https://doi.org/10.1090/mbk/121/79

1. Guan, J., & Elman, H. C. (2024). Transformed Physics-Informed Neural Networks for The Convection-Diffusion Equation. Arxiv (Cornell University). https://doi.org/10.48550/arxiv.2409.07671

1. Bragone, F., Morozovska, K., Hilber, P., Laneryd, T., & Luvisotto, M. (2022). Physics-informed neural networks for modelling power transformer’s dynamic thermal behaviour. Electric Power Systems Research, 211, 108447. https://doi.org/10.1016/j.epsr.2022.108447

