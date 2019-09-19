$$x_k = f(x_{k-1}) + v_k$$
$$y_k = h(x_k) + w_k$$
$$v_k ~ \mathcal{N}(0, Q)$$
$$w_k ~ \mathcal{N}(0, R)$$

$$\mathcal{N}(m, P) = \frac{1}{\sqrt{|2 \pi P|}} e^{-0.5 m^T P^{-1} m}$$

## Bootstrap filter (BF)

**for** i = 1:n **do**  
&emsp;&emsp;Draw $v_k^i ~ \mathcal{N}(0, Q)$  
&emsp;&emsp;Compute the sample target state $\chi_k^i = f(\chi_{k-1}^{i*}) + v_k^i$  
&emsp;&emsp;Compute unnormaled weights (likelihoods) $\bar w_k^i = \mathcal{N}(y_k - h(\chi_k^i), R)$  
**end for**  
Normalize weights $w_k^i = \dfrac{\bar w_k^i}{\sum_{j=1}^n \bar w_k^j}$  
Compute updated state estimate 
$$\hat x_{k|k} = \sum_{i=1}^n w_k^i \chi_k^i$$
$$P_{k|k} = \sum_{i=1}^n w_k^i (\chi_k^i - \hat x_{k|k})(\chi_k^i - \hat x_{k|k})^T$$
**for** i = 1:n **do**  
&emsp;&emsp;Resampling: Draw samples $\chi_{k}^{i*} ~ Pr(\chi_{k}^{i*} = \chi_{k}^l) = w_{k}^l$  
**end for**  

## BF for PDA

**for** i = 1:n **do**  
&emsp;&emsp;Draw samples 
$x_{k-1}^{i*} ~ Pr(x_{k-1}^{i*} = x_{k-1}^l) = w_{k-1}^l$  
&emsp;&emsp;Draw $v_k^i ~ \mathcal{N}(0, Q)$  
&emsp;&emsp;Compute the sample target state $x_k^i = f(x_{k-1}^{i*}) + v_k^i$  
&emsp;&emsp;Compute unnormaled weights (likelihoods)  
$$\bar w_k^i = 1 - p_D + \frac{p_D}{\lambda} \sum_{j=1}^n \mathcal{N}(y_k - h(x_k^i), R)$$
**end for**  
Normalize weights $w_k^i = \dfrac{\bar w_k^i}{\sum_{j=1}^n \bar w_k^j}$  
Compute updated state estimate $\hat x_{k|k} = \sum_{i=1}^n w_k^i x_k^i$

## Gaussian mixture PHD (GM-PHD) filter

// Input: Density 
$$D_{k-1|k-1}(x_{k-1}|y^{k-1}) = \sum_{i=1}^{J_{k-1}} w_{k-1}^i \mathcal{N}(x_{k-1}, m_{k-1}^i, P_{k-1}^i)$$
Prediction
$$D_{k|k-1}(x_k|y^{k-1}) = D_{S,k|k-1}(x_k|y^{k-1}) + D_{b,k|k-1}(x_k|y^{k-1}) + \gamma_k(x_k)$$
where
$$D_{S,k|k-1}(x_k|y^{k-1}) = \sum_{i=1}^{J_{k-1}} w_{S,k}^i \mathcal{N}(x_k; m_{S,k}^i, P_{S,k|k-1}^i)$$
$$[m_{S,k}^i, P_{S,k|k-1}^i] = KF_P[m_{k-1}^i, P_{k-1}^i, F_k, Q_k]$$
$$w_{S,k}^i = p_{S,k} w_{k-1}^i$$
$$D_{b,k|k-1}(x_k|y^{k-1}) = \sum_{i=1}^{J_{k-1}}w_{b,k} \sum_{j=1}^{J_{b,k}} \mathcal{N}(x_k; m_{b,k}^{i,j}, P_{b,k|k-1}^{i,j})$$
$$m_{b,k}^{i,j} = F_{b,k-1}^i + u_{b,k}^j$$
$$P_{b,k|k-1}^{i,j} = F_{b,k-1}^j P_{b,k-1}^i (F_{b,k-1}^j)^T + Q_{b,k-1}^j$$
$$\gamma_k(x_k) = \sum_{i=1}^{J_{\gamma,k}} w_{\gamma,k}^i \mathcal{N}(x_k; m_{\gamma,k}^i, P_{\gamma,k}^i)$$

Update
$$D_{k|k}(x_k|y^k) = (1-p_{D,k}) D_{k|k-1}(x_k|y^{k-1}) + \sum_{z_k \in Z_k} z_k D_{z,k}(x_k; z_k)$$
where
$$D_{z,k}(x_k;z_k) = \sum_{i=1}^{J_{k|k-1}} w_k^i(z_k) \mathcal{N}(x_k; m_{k|k}^i(z_k), P_{k|k}^i)$$
$$w_k^i(z_k) = \frac{p_{D,k} w_{k|k-1}^i q_k^i(z_k)}{\lambda_k c_k(z_k) + p_{D,k}\sum_{j=1}^{J_{k|k-1}} w_{k|k-1}^j q_k^j(z_k)}$$
$$q_k^i(z_k) = \mathcal{N}(z_k; H_k m_{k|k-1}^i, H_k P_{k|k-1}^i H_k^T + R_k)$$
$$[m_{k|k}^i(z_k), P_{k|k}^i] = KF_E[z_k, m_{k|k-1}^i, P_{k|k-1}^i, H_k, R_k]$$

## JPDA

**for** &tau; = 1:T **do**  
&emsp;&emsp;Predict each track
$$[x_{k|k-1}^\tau, P_{k|k-1}^\tau] = KF_P[x_{k-1|k-1}^\tau, P_{k-1|k-1}^\tau, F, Q]$$
**for** s = 1:S **do**  
&emsp;&emsp;Compute cost matrix
$$C_{td} = KF_D[x_{k|k-1}, P_{k|k-1}, y^{s,d}, H, R], t=1,...,T,d=1,...,D$$
&emsp;&emsp;Form validation matrix
$$V_{dt'} = \begin{cases}
1, \, t' = 1 \, \text{or} \,  C_{t'-1,d} < \varepsilon_g\\
0, \, \text{otherwise}\\
\end{cases},d=1,...,D,t'=1,...,T+1$$
&emsp;&emsp;Use DFS from validation matrix to get clusters and unassigned   
**end for**  
**for** c = 1:n_Cl **do**  
Compute likelihood
$$p_k^t(i) = 
\begin{cases}
\mathcal{N}(y_k; H_k x_{k|k-1},H_k P_{k|k-1} H_k^T + R_k)\\
0
\end{cases}$$
Form all FJEs  
**for** each FJE e **do**  
posterior probability of FJE
$$p(e|Y^k) = c_k^{-1}\prod_{t\in T_0(e)} (1-p_D^t p_G^t \sigma_{k|k-1}^t) 
\prod_{t \in T_1(e)} (p_D^t p_G^t \sigma_{k|k-1}^t \frac{p_k^t(i(t,e))}{\rho_k(i(t,e))}$$

$$p(\chi_k^t, \theta_k^t(i)|Y^k) = 
\begin{cases}
\sum\limits_{e\in \varTheta(t,i)} p(e|Y^k) \, i>0\\
\dfrac{(1-p_d^t p_G^t) \sigma_{k|k-1}^t}{1 - p_D^t p_G^t \sigma_{k|k-1}^t} \sum\limits_{e\in \varTheta(t,0)} p(e|Y^k)
\end{cases}$$
Data association probabilities
$$\beta_k^t(i) = \frac{p(\chi_k^t, \theta_k^t(i)|Y^k)}{\sum_{i=0} p(\chi_k^t, \theta_k^t(i)|Y^k)}$$
Update
un
initialize
delete