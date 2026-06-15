import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def fit_shash(x):
    """
    Fits the Sinh-Arcsinh (SHASH) distribution to a 1D numpy array.
    
    Parameters:
    -----------
    x : array_like
        The input data.
        
    Returns:
    --------
    dict
        A dictionary containing the fitted parameters of the SHASH distribution.
    
    dict
        A dictionary containing flags about the optimization process (success, nll).
    """
    # Fit directly using scipy's minimize function on the negative log-likelihood

    from scipy.optimize import minimize
    
    def negative_log_posterior(params):
        # Unpack parameters
        mu, log_sigma, log_delta, epsilon = params
        sigma = np.exp(log_sigma)
        delta = np.exp(log_delta)

        neg_log_likelihood = -np.sum(shash_logpdf(x, mu, sigma, delta, epsilon))
        neg_log_prior = -(log_delta + log_sigma) # Flat prior on delta and sigma (if positive)

        return neg_log_likelihood #+ neg_log_prior
    
    initial_guess = [np.mean(x), np.log(np.std(x)), -5, 0.0]
    result = minimize(negative_log_posterior, initial_guess)#method='L-BFGS-B')
    mu_opt, log_sigma_opt, log_delta_opt, epsilon_opt = result.x
    sigma_opt = np.exp(log_sigma_opt)
    delta_opt = np.exp(log_delta_opt)
    
    params ={
        'mu': mu_opt,
        'sigma': sigma_opt,
        'delta': delta_opt,
        'epsilon': epsilon_opt
    }
    flags = {
        'success': result.success,
        'ln_posterior': -result.fun
    }

    return params, flags


def shash_pdf(x, mu, sigma, delta, epsilon):
    """
    Computes the Probability Density Function (PDF) of the SHASH distribution.
    
    Parameters:
    -----------
    x : array_like
        The input quantiles.
    mu : float
        Location parameter (mean of the base distribution).
    sigma : float
        Scale parameter (standard deviation of the base distribution).
    delta : float
        Kurtosis/tail-weight parameter (delta > 0).
    epsilon : float
        Skewness parameter.
        
    Returns:
    --------
    array_like
        The evaluated PDF.
    """
    x = np.asarray(x)
    
    # Standardize the input
    w = (x - mu) / sigma
    
    # Calculate the transformation argument
    C = delta * np.arcsinh(w) - epsilon
    
    # Compute the components of the PDF
    normalization = delta / (sigma * np.sqrt(2 * np.pi * (1 + w**2)))
    hyperbolic_term = np.cosh(C)
    exponential_term = np.exp(-0.5 * np.sinh(C)**2)
    
    return normalization * hyperbolic_term * exponential_term


def shash_cdf(x, mu, sigma, delta, epsilon):
    """
    Computes the Cumulative Distribution Function (CDF) of the SHASH distribution.
    
    Parameters:
    -----------
    x : array_like
        The input quantiles.
    mu : float
        Location parameter (mean of the base distribution).
    sigma : float
        Scale parameter (standard deviation of the base distribution).
    delta : float
        Kurtosis/tail-weight parameter (delta > 0).
    epsilon : float
        Skewness parameter.
        
    Returns:
    --------
    array_like
        The evaluated CDF.
    """
    x = np.asarray(x)
    
    # Standardize the input
    w = (x - mu) / sigma
    
    # Calculate the transformation argument
    C = delta * np.arcsinh(w) - epsilon
    
    # The CDF is the standard normal CDF evaluated at the transformed variable
    transformed_variable = np.sinh(C)
    
    return norm.cdf(transformed_variable)


def shash_logpdf(x, mu, sigma, delta, epsilon):
    """Computes the log-PDF for numerical stability in the tails."""
    x = np.asarray(x)
    w = (x - mu) / sigma
    C = delta * np.arcsinh(w) - epsilon
    
    # Using np.logaddexp(C, -C) - np.log(2) is a stable way to compute log(cosh(C))
    log_normalization = np.log(delta) - np.log(sigma) - 0.5 * np.log(2 * np.pi * (1 + w**2))
    log_hyperbolic = np.logaddexp(C, -C) - np.log(2)
    # Clip |C| to prevent sinh overflow during optimisation.
    C = np.clip(C, -250.0, 250.0)
    log_exponential = -0.5 * np.sinh(C)**2
    
    return log_normalization + log_hyperbolic + log_exponential


def main():
    # Example usage

    # Generate synthetic data from a SHASH distribution
    rng = np.random.default_rng(seed=42)
    true_params = {
        'mu': 0.0,
        'sigma': 1.0,
        'delta': 2.0,
        'epsilon': 0.5
    }
    x = np.linspace(-5, 5, 1000)
    pdf_values = shash_pdf(x, **true_params)
    cdf_values = shash_cdf(x, **true_params)
    # Sample from the distribution using inverse transform sampling
    uniform_samples = rng.uniform(size=1024*4)
    shash_samples = np.interp(uniform_samples, cdf_values, x)

    # Fit the SHASH distribution to the samples
    print('Fitting...')
    fitted_params, flags = fit_shash(shash_samples)
    print("True SHASH parameters:", true_params)
    print("Fitted SHASH parameters:", fitted_params)
    print("Optimization flags:", flags)


if __name__ == "__main__":
    main()