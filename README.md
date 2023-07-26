### Model_Evaluation is a collection of analysis notebooks whose purpose is to visualize data, compare DIC, TA and TA-DIC across models and observations. and to calculate bias, RMSE, and WSS.

Author: Mai Hong

List of subfolders:

#### multiple_models_obs
- Multi-model and observation comparisons
- Thirty-year regionally weighted average of all data (1990-2020)
- PLOTS: y - Region, x - DIC; TA; TA-DIC

#### single_models_obs
- A function with command line arguments (ModelEvaluation.py)
- Single Model and observation comparisons
- Save at the specified address. csv files and plots
- Observed and modeled values for global DIC and TA after matching, and bias, RMSE, and WSS
- Observed and modeled values for subregion-weighted DIC and TA after matching, and sub-regional bias, RMSE, and WSS.
- The default start year: 1990 & the end year: 2020.

#### evaluation_metrics
- Multiple-model Bias, RMSE, WSS sub-area comparison plots

#### other
- GLODAP (Data distribu+on, global, depth)
- Salinity plots (DIC-SAL, TA-SAL, (TA-DIC)-SAL)
- Model region plots (DIC vs. TA-DIC)
- Time series (Trends in modelled and observed values between 1990 and 2020, subregional)
