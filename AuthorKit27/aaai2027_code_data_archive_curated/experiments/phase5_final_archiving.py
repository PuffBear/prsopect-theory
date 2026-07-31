import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

def compute_cohens_f2(rsquared):
    # Cohen's f^2 = R^2 / (1 - R^2)
    return rsquared / (1 - rsquared) if rsquared < 1 else np.inf

def run_archiving_and_stats():
    # 1. Merge the datasets
    df_cent = pd.read_csv('docs/phase4_figures/phase4_raw_data.csv')
    df_cent['Alpha'] = 1.0
    
    df_decent = pd.read_csv('docs/phase4_figures/phase4_5_interaction_data.csv')
    df_decent['Alpha'] = 0.0
    
    # We need to make sure the columns match for concat
    # df_cent has: Lambda, Seed, Mean S, Mean Order, Mean Inventory, Lost Sales, Profit, Bullwhip
    # df_decent has: Lambda, Seed, Mean S, Profit, Lost Sales
    # We will just keep the common metrics or fill NaN.
    
    df_final = pd.concat([df_cent, df_decent], ignore_index=True)
    df_final['Log_Lambda'] = np.log(df_final['Lambda'])
    
    # Save the master dataset
    os.makedirs('docs/final_archive', exist_ok=True)
    df_final.to_csv('docs/final_archive/master_seed_dataset.csv', index=False)
    print("Saved master dataset to docs/final_archive/master_seed_dataset.csv")
    
    print("\n=======================================================")
    print("FINAL FACTORIAL REGRESSION & EFFECT SIZES")
    print("=======================================================\n")
    
    # We want: Y = beta_0 + beta_1*log(lambda) + beta_2*alpha + beta_3*(log(lambda)*alpha)
    df_final['Interaction'] = df_final['Log_Lambda'] * df_final['Alpha']
    X = sm.add_constant(df_final[['Log_Lambda', 'Alpha', 'Interaction']])
    
    metrics = ['Mean S', 'Profit', 'Lost Sales']
    
    with open('docs/final_archive/final_statistics.txt', 'w') as f:
        f.write("FINAL FACTORIAL REGRESSIONS\n")
        f.write("Model: Y = b0 + b1*Log(Lambda) + b2*Alpha + b3*(Log(Lambda)*Alpha)\n")
        f.write("="*70 + "\n\n")
        
        for metric in metrics:
            print(f"--- Metric: {metric} ---")
            y = df_final[metric]
            
            # Remove NaNs if any (e.g., if a dataset was missing a metric)
            valid_idx = y.notna()
            y_valid = y[valid_idx]
            X_valid = X[valid_idx]
            
            model = sm.OLS(y_valid, X_valid).fit()
            
            # Effect sizes
            rsquared = model.rsquared
            adj_rsquared = model.rsquared_adj
            f2 = compute_cohens_f2(rsquared)
            
            # Partial Eta Squared using ANOVA
            df_anova = df_final[valid_idx].copy()
            # To do ANOVA, we need categorical Alpha, but continuous Log_Lambda is fine
            df_anova['Alpha_Cat'] = df_anova['Alpha'].astype(str)
            anova_model = ols(f'Q("{metric}") ~ Log_Lambda * Alpha_Cat', data=df_anova).fit()
            anova_table = sm.stats.anova_lm(anova_model, typ=2)
            
            # Calculate Partial Eta Squared for the interaction term
            ss_interaction = anova_table.loc['Log_Lambda:Alpha_Cat', 'sum_sq']
            ss_residual = anova_table.loc['Residual', 'sum_sq']
            partial_eta2 = ss_interaction / (ss_interaction + ss_residual) if (ss_interaction + ss_residual) > 0 else 0
            
            print(model.summary().tables[1])
            print(f"R-squared: {rsquared:.4f}")
            print(f"Adjusted R-squared: {adj_rsquared:.4f}")
            print(f"Cohen's f^2: {f2:.4f}")
            print(f"Partial Eta^2 (Interaction): {partial_eta2:.4f}\n")
            
            # Write to file
            f.write(f"--- Metric: {metric} ---\n")
            f.write(model.summary().as_text() + "\n")
            f.write(f"R-squared: {rsquared:.4f}\n")
            f.write(f"Adjusted R-squared: {adj_rsquared:.4f}\n")
            f.write(f"Cohen's f^2: {f2:.4f}\n")
            f.write(f"Partial Eta^2 (Interaction): {partial_eta2:.4f}\n")
            f.write("-" * 50 + "\n\n")

if __name__ == "__main__":
    run_archiving_and_stats()
