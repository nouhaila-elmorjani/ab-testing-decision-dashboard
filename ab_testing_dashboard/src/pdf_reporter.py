import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import tempfile

class ABTestReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'A/B Test Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def add_content(self, content):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 8, content)
        self.ln(3)

def generate_ab_test_report(conversion_summary, test_results, output_path):
    """
    Generate a professional PDF report for A/B test results
    
    Parameters:
    conversion_summary: Conversion metrics DataFrame
    test_results: Statistical test results dictionary
    output_path: Path to save the PDF report
    """
    pdf = ABTestReportPDF()
    pdf.add_page()
    
    # Report title and introduction
    pdf.add_section_title('Executive Summary')
    pdf.add_content('This report presents the results of the A/B test analysis, including statistical significance and business recommendations.')
    
    # Conversion Metrics Section
    pdf.add_section_title('Conversion Metrics')
    if conversion_summary is not None:
        metrics_text = f"""
        Group A ({test_results['group_a']}): {conversion_summary.loc[test_results['group_a'], 'conversion_rate']:.3%} conversion rate
        Group B ({test_results['group_b']}): {conversion_summary.loc[test_results['group_b'], 'conversion_rate']:.3%} conversion rate
        Absolute Difference: {test_results['absolute_difference']:.4f}
        Relative Improvement: {test_results['relative_improvement']:.1f}%
        """
        pdf.add_content(metrics_text)
    
    # Statistical Results Section
    pdf.add_section_title('Statistical Analysis')
    stats_text = f"""
    Z-Score: {test_results['z_score']:.4f}
    P-Value: {test_results['p_value']:.6f}
    95% Confidence Interval: [{test_results['confidence_interval'][0]:.4f}, {test_results['confidence_interval'][1]:.4f}]
    Statistical Significance: {'Yes' if test_results['is_significant'] else 'No'}
    """
    pdf.add_content(stats_text)
    
    # Business Recommendation Section
    pdf.add_section_title('Business Recommendation')
    if test_results['is_significant'] and test_results['absolute_difference'] > 0:
        recommendation = f"IMPLEMENT GROUP {test_results['group_b']}: Statistically significant improvement of {test_results['relative_improvement']:.1f}% detected."
    elif test_results['is_significant'] and test_results['absolute_difference'] < 0:
        recommendation = f"IMPLEMENT GROUP {test_results['group_a']}: Statistically significant improvement detected."
    else:
        recommendation = "NO CLEAR WINNER: Test results are not statistically significant. Consider extending test duration or increasing sample size."
    
    pdf.add_content(recommendation)
    
    # Sample Size Information
    pdf.add_section_title('Test Details')
    if conversion_summary is not None:
        details_text = f"""
        Total Users in Test: {conversion_summary['user_count'].sum():,}
        Group A Sample Size: {conversion_summary.loc[test_results['group_a'], 'user_count']:,}
        Group B Sample Size: {conversion_summary.loc[test_results['group_b'], 'user_count']:,}
        Total Conversions: {conversion_summary['conversions'].sum():,}
        """
        pdf.add_content(details_text)
    
    # Save the PDF
    pdf.output(output_path)
    return output_path