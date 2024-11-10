from pathlib import Path

import pandas as pd

RESULTS_DIRECTORY = Path(__file__).parent.parent.joinpath("results").resolve()
file_path = RESULTS_DIRECTORY.joinpath('ANN_derivative/Ann_derivative_test.csv')
df = pd.read_csv(file_path, delimiter=',')
print(df)
best_params = df.loc[df['mae'].idxmin()] if 'mae' in df.columns else None
worst_params = df.loc[df['mae'].idxmax()] if 'mae' in df.columns else None

print("Best Parameters based on MAE:")
print(best_params)


best_params_output_file_path = RESULTS_DIRECTORY.joinpath('ANN_derivative/best_ann_derivative_analyse.csv')
best_params_df = pd.DataFrame({
    'Learning Rate': [best_params['learning_rate']],
    'Activation Function': [best_params['activation_function']]
})
best_params_df.to_csv(best_params_output_file_path, index=False)


worst_params_output_file_path = RESULTS_DIRECTORY.joinpath('ANN_derivative/worst_ann_derivative_analyse.csv')
worst_params_df = pd.DataFrame({
    'Learning Rate': [worst_params['learning_rate']],
    'Activation Function': [worst_params['activation_function']]
})
worst_params_df.to_csv(worst_params_output_file_path, index=False)
print("Worst Parameters based on MAE:")
print(worst_params)


