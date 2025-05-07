import numpy as np
from scipy import stats

def solve_reservation_wage(beta, c, dist, 
                           N=100_000, tol=1e-8, max_iter=10_000, verbose=False):
    """
    Solve the McCall reservation‐wage model by fixed‐point iteration on VU:
        VU_{n+1} = -c + beta * E[max(w/(1-beta), VU_n)]
    then R = (1-beta) * VU.

    Args:
      beta      : discount factor in (0,1)
      c         : search cost (>0)
      dist      : a scipy.stats continuous distribution (must have .rvs())
      N         : Monte‐Carlo sample size for integration
      tol       : convergence tolerance for VU
      max_iter  : maximum number of iterations
      verbose   : print progress every 100 iters

    Returns:
      R  : reservation wage
      VU : value of remaining unemployed
    """

    # 1) predraw wage 
    w_samp = dist.rvs(size=N)
    VU = 0.0

    # 2) fixed‐point iteration
    for it in range(1, max_iter+1):
        # value if you accept each draw
        accept_vals = w_samp / (1 - beta)
        EV = np.mean(np.maximum(accept_vals, VU))

        VU_new = -c + beta * EV
        if verbose and it % 100 == 0:
            print(f"[iter {it:5d}] VU = {VU_new:.6f}")

        if abs(VU_new - VU) < tol:
            VU = VU_new
            if verbose:
                print(f"Converged after {it} iterations.")
            break
        VU = VU_new
    else:
        print("didn't converge")

    R = (1 - beta) * VU
    return R, VU


if __name__ == "__main__":
    beta = 0.9
    c    = 0.1

    # Example 1: uniform distributino U~[0,1]
    dist1 = stats.uniform(loc=0, scale=1)
    R1, VU1 = solve_reservation_wage(beta, c, dist1, verbose=True)
    print(f"\nUniform[0,1]:  R = {R1:.4f} (reservation wage),  VU = {VU1:.4f} (value of unemployment)\n")

    # Example 2: truncated normal on [0,∞) using scipy's truncnorm
    from scipy.stats import truncnorm
    mu, sigma = 1.0, 0.5
    a, b = (0 - mu) / sigma, np.inf  
    dist2 = truncnorm(a, b, loc=mu, scale=sigma)
    
    R2, VU2 = solve_reservation_wage(beta, c, dist2, verbose=False)
    print(f"Truncated-Normal: R = {R2:.4f} (reservation wage), VU = {VU2:.4f} (value of unemployment)") 