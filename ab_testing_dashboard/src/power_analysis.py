import math
from scipy.stats import norm

def calculate_sample_size(baseline_rate, min_detectable_effect, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test
    
    Parameters:
    baseline_rate: Current conversion rate (e.g., 0.02 for 2%)
    min_detectable_effect: Minimum effect size to detect (e.g., 0.01 for 1% absolute improvement)
    alpha: Significance level (default 0.05 for 95% confidence)
    power: Statistical power (default 0.8 for 80% power)
    
    Returns:
    int: Required sample size per group
    """
    # Calculate effect size
    effect_size = abs(min_detectable_effect)
    
    # Z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed test
    z_beta = norm.ppf(power)
    
    # Sample size calculation for proportions
    p1 = baseline_rate
    p2 = baseline_rate + effect_size
    p_pooled = (p1 + p2) / 2
    
    numerator = (z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = effect_size ** 2
    
    sample_size = math.ceil(numerator / denominator)
    return sample_size

def calculate_statistical_power(conversion_summary, alpha=0.05):
    """
    Calculate achieved statistical power for completed test
    
    Parameters:
    conversion_summary: DataFrame with conversion metrics
    alpha: Significance level
    
    Returns:
    float: Statistical power achieved
    """
    try:
        groups = conversion_summary.index.tolist()
        if len(groups) < 2:
            return 0.0
            
        group_a, group_b = groups[0], groups[1]
        
        n1 = conversion_summary.loc[group_a, 'user_count']
        n2 = conversion_summary.loc[group_b, 'user_count']
        p1 = conversion_summary.loc[group_a, 'conversion_rate']
        p2 = conversion_summary.loc[group_b, 'conversion_rate']
        
        # Effect size
        effect_size = abs(p2 - p1)
        
        # Standard error
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        
        # Z-score for alpha
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size / se
        
        # Calculate power
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        
        return max(0.0, min(1.0, power))
        
    except Exception:
        return 0.0