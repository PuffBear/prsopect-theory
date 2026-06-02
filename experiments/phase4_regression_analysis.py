import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_regression_analysis():
    df = pd.read_csv('docs/phase4_figures/phase4_raw_data.csv')
    df['Log_Lambda'] = np.log(df['Lambda'])
    
    print("=======================================================")
    print("PHASE 4: EXTENDED STATISTICAL REGRESSION & ANOVA")
    print("=======================================================\n")
    
    # --- TEST A: Base-Stock (S) Significance ---
    print("--- TEST A: Base-Stock (S) ~ log(Lambda) ---")
    
    X_log = sm.add_constant(df['Log_Lambda'])
    model_log_s = sm.OLS(df['Mean S'], X_log).fit()
    
    print(f"  Coefficient (beta_1) : {model_log_s.params['Log_Lambda']:.4f}")
    print(f"  95% CI               : [{model_log_s.conf_int().loc['Log_Lambda', 0]:.4f}, {model_log_s.conf_int().loc['Log_Lambda', 1]:.4f}]")
    print(f"  P-value              : {model_log_s.pvalues['Log_Lambda']:.4e}")
    print(f"  R-squared            : {model_log_s.rsquared:.4f}")
    print(f"  Adjusted R-squared   : {model_log_s.rsquared_adj:.4f}\n")
    
    # --- ANOVA (Eta-Squared): Signal vs Noise ---
    print("--- ANOVA: Variance Explained (eta-squared) ---")
    # We want to compare the variance explained by Lambda (the signal) vs the Seed (the noise)
    # Since Lambda is a categorical-like variable in the design, we can treat it as categorical for ANOVA.
    df['Lambda_Cat'] = df['Lambda'].astype(str)
    model_anova = ols('Q("Mean S") ~ C(Lambda_Cat)', data=df).fit()
    anova_table = sm.stats.anova_lm(model_anova, typ=2)
    
    # Eta-squared = SS(Lambda) / (SS(Lambda) + SS(Residual))
    ss_lambda = anova_table.loc['C(Lambda_Cat)', 'sum_sq']
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    eta_squared = ss_lambda / (ss_lambda + ss_residual)
    
    print(f"  SS(Lambda)           : {ss_lambda:.2f}")
    print(f"  SS(Residual)         : {ss_residual:.2f}")
    print(f"  Eta-squared (eta^2)  : {eta_squared:.4f}")
    print(f"  (This means {eta_squared*100:.1f}% of variance is explained by Lambda, while {(1-eta_squared)*100:.1f}% is noise/seed variance)\n")

    
    # --- TEST B: Lost Sales Significance ---
    print("--- TEST B: Lost Sales ~ log(Lambda) ---")
    
    model_log_ls = sm.OLS(df['Lost Sales'], X_log).fit()
    print(f"  Coefficient (beta_1) : {model_log_ls.params['Log_Lambda']:.4e}")
    print(f"  95% CI               : [{model_log_ls.conf_int().loc['Log_Lambda', 0]:.4e}, {model_log_ls.conf_int().loc['Log_Lambda', 1]:.4e}]")
    print(f"  P-value              : {model_log_ls.pvalues['Log_Lambda']:.4e}")
    print(f"  R-squared            : {model_log_ls.rsquared:.4f}")
    print(f"  Adjusted R-squared   : {model_log_ls.rsquared_adj:.4f}\n")
    
    
    # --- TEST C: Economic Profit Significance ---
    print("--- TEST C: Economic Profit ~ log(Lambda) ---")
    
    model_log_prof = sm.OLS(df['Profit'], X_log).fit()
    print(f"  Coefficient (beta_1) : {model_log_prof.params['Log_Lambda']:.4f}")
    print(f"  95% CI               : [{model_log_prof.conf_int().loc['Log_Lambda', 0]:.4f}, {model_log_prof.conf_int().loc['Log_Lambda', 1]:.4f}]")
    print(f"  P-value              : {model_log_prof.pvalues['Log_Lambda']:.4e}")
    print(f"  R-squared            : {model_log_prof.rsquared:.4f}")
    print(f"  Adjusted R-squared   : {model_log_prof.rsquared_adj:.4f}\n")
    
    with open('docs/phase4_figures/extended_regression_results.txt', 'w') as f:
        f.write("Extended Stats Computed.")
        
if __name__ == "__main__":
    run_regression_analysis()
