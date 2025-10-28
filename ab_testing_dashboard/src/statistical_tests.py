import math
from scipy.stats import norm

def perform_ab_test(conversion_summary):
    """
    Perform A/B test statistical analysis
    
    Parameters:
    conversion_summary: DataFrame from get_conversion_summary
    
    Returns:
    dict: Statistical test results
    """
    try:
        groups = conversion_summary.index.tolist()
        if len(groups) < 2:
            return None
            
        group_a, group_b = groups[0], groups[1]
        
        # Extract conversion data
        conv_a = conversion_summary.loc[group_a, 'conversions']
        total_a = conversion_summary.loc[group_a, 'user_count']
        conv_b = conversion_summary.loc[group_b, 'conversions'] 
        total_b = conversion_summary.loc[group_b, 'user_count']
        
        # Conversion rates
        p1 = conv_a / total_a
        p2 = conv_b / total_b
        
        # Pooled probability
        p_pooled = (conv_a + conv_b) / (total_a + total_b)
        
        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))
        
        # Z-score
        z_score = (p2 - p1) / se
        
        # P-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Confidence interval
        diff = p2 - p1
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        
        # Relative improvement
        relative_improvement = (diff / p1) * 100 if p1 > 0 else 0
        
        return {
            'group_a': group_a,
            'group_b': group_b,
            'conversion_rate_a': p1,
            'conversion_rate_b': p2,
            'absolute_difference': diff,
            'relative_improvement': relative_improvement,
            'z_score': z_score,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': p_value < 0.05
        }
        
    except Exception as e:
        return None