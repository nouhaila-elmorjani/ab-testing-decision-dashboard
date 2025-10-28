import matplotlib.pyplot as plt
import seaborn as sns

def plot_conversion_rates(conversion_summary):
    """
    Create conversion rate comparison plot
    
    Parameters:
    conversion_summary: DataFrame from get_conversion_summary
    
    Returns:
    matplotlib.figure: Conversion rate plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    conversion_rates = conversion_summary['conversion_rate']
    groups = conversion_rates.index.tolist()
    
    bars = ax.bar(groups, conversion_rates.values, 
                 color=['lightblue', 'lightcoral'], alpha=0.7)
    
    ax.set_title('Conversion Rates by Test Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Conversion Rate', fontsize=12)
    ax.set_xlabel('Test Group', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, conversion_rates.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    return fig

def plot_user_distribution(conversion_summary):
    """
    Create user distribution plot
    
    Parameters:
    conversion_summary: DataFrame from get_conversion_summary
    
    Returns:
    matplotlib.figure: User distribution plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    user_counts = conversion_summary['user_count']
    groups = user_counts.index.tolist()
    
    bars = ax.bar(groups, user_counts.values,
                 color=['lightgreen', 'orange'], alpha=0.7)
    
    ax.set_title('User Distribution by Test Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_xlabel('Test Group', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, user_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(user_counts.values) * 0.01,
               f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    return fig