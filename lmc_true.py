## the code here will be used to build a icm model
## the code wll not be fundamentally too different from the code 
## we will use to build a lcm model
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import gpytorch
from utils import *

np.random.seed(195)

## here we will start by defining a gaussian process model
n_tasks = 3

class ICMModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ICMModel, self).__init__(train_x, train_y, likelihood)
        ## we have the mean module for the model
        self.mean_module = gpytorch.means.ConstantMean()
        ## build kernel for continuous variables some variables have 
        self.covar_rq = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveKernel(gpytorch.kernels.PeriodicKernel()))
        #self.covar_rbf = gpytorch.kernels.RBFKernel(ard_num_dims = 5, active_dims = [1,3,5,7,9]) 
        #self.composite = gpytorch.kernels.ProductKernel(self.covar_rq, self.covar_rbf)
        self.task_kern = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=1)

    def forward(self, x_continuous, task_ids):
        mean_x = self.mean_module(x_continuous)
        covar_continuous = self.covar_rq(x_continuous)
        covar_task = self.task_kern(task_ids)
        ## here we have a coregionalization matrix mixed with the inputs matrix
        full_covar = covar_continuous.mul(covar_task)
        return gpytorch.distributions.MultivariateNormal(mean=mean_x, covariance_matrix=full_covar)
    
## here we define the likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-2))
likelihood.noise = torch.tensor(1e-2)


def make_task_indices(X, n_tasks, dim=1, dtype=torch.long):
        """
        Returns a list of n_tasks tensors, each of shape (X.shape[0], dim),
        filled with task indices 0, 1, …, n_tasks-1.

        Args:
            X (Tensor or anything with .shape): input whose first dim is batch size
            n_tasks (int): number of tasks
            dim (int): number of “columns” per task-index tensor (default 1)
            dtype (torch.dtype): dtype of the output tensors (default torch.long)

        Returns:
            List[Tensor]: [T0, T1, …, T{n_tasks-1}], each of shape (batch_size, dim)
        """
        batch_size = X.shape[0]
        return [
            torch.full((batch_size, dim), fill_value=i, dtype=dtype)
            for i in range(n_tasks)
        ]




def gp_model(X_train, y_train, train_task, test_df, likelihood, training_iter=100):
    model = ICMModel((X_train, train_task), y_train, likelihood)

    ## here we will start by doing the training loop
    smoke_test = ('CI' in os.environ)
    training_iterations = 2 if smoke_test else training_iter

    ## set the model in training mode 
    model.train()
    likelihood.train()
    mse_global = []
    mean_width_global = []
    mean_interval_global = []
    mspe_interval_global = []
    mspe_width_global    = []
    alpha = 1.96

    ## set up the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()),lr=0.04) 
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 

    for i in range(training_iterations):
        with gpytorch.settings.max_cg_iterations(20000), gpytorch.settings.cholesky_jitter(1e-2):
            optimizer.zero_grad() # Zero gradients from previous iteration
            output = model(X_train, train_task) # Output from model
            loss = -mll(output, y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()


    ## we set our model in eval mode
    model.eval()
    likelihood.eval()

    for i in range(1,n_tasks+1):
        ## created the data for the evaluation
        print(f"X_test, SoC for battery {i}")
        ## we use the provided test dataframe
        ## since we still have the process columns we separate the test data by process
        X_test = torch.tensor(test_df.loc[test_df["battery"]==i].drop(columns=["SoC", "battery"]).values, dtype=torch.float)
        ## do the same for target values
        y_test = torch.tensor(test_df["SoC"].loc[test_df["battery"]==i].values, dtype=torch.float)
        ## we create ids for the test where the id is i-1
        ids = torch.full((X_test.shape[0],1), fill_value=i-1, dtype=torch.long)
        
        with torch.no_grad(),gpytorch.settings.fast_pred_var(True), gpytorch.settings.cholesky_jitter(1e-2):

                    # f_preds is the posterior over the latent function f(x)
            f_pred = model(X_test, ids)  

                    # y_pred is the posterior over the *observations* y*, i.e. includes noise
            y_pred = likelihood(f_pred)   

            lower, upper = f_pred.confidence_region() 
        ## get mse between prediction and true values non robust
        mse = gpytorch.metrics.mean_squared_error(y_pred, y_test)
        ## mean width of the confint for all points
        mean_width = torch.mean(abs(upper-lower))
        ## we get how much time the target falls in the predictive interval
        interval = (y_test >= lower) & (y_test <= upper)
        ## we get the percentage of that
        percentage = interval.float().mean().item() * 100

        ## here we use mspe to construct robust confint
        ## we start by computing bias squared of our model
        bias_sq = ((f_pred.mean - y_test).mean())**2
        ## here we will give the robust interval
        mspe = y_pred.variance + bias_sq

        ## here we will calculate the robust confint
        upper_robust = f_pred.mean + (alpha*torch.sqrt(mspe))
        lower_robust = f_pred.mean - (alpha*torch.sqrt(mspe))
        mean_width_robust = torch.mean(abs(upper_robust-lower_robust))

        ## we get if the test are in the robust interval
        interval_robust = (y_test >= lower_robust) & (y_test <= upper_robust)
        percentage_robust = interval_robust.float().mean().item() * 100

        ##here we add the robust interval
        mspe_interval_global.append(np.float64(percentage_robust))
        mspe_width_global.append(np.float64(mean_width_robust))
        ## the normal confint width and percentage
        mean_width_global.append(np.float64(mean_width))
        mean_interval_global.append(np.float64(percentage))
        ## add them to the list for each process
        ## mse
        mse_global.append(np.float64(mse))
        idx = np.arange(len(y_test))
        plt.figure(figsize=(8,4))
        plt.plot(idx, y_test.numpy(),    label="True SoC")
        plt.plot(idx, y_pred.mean.detach().numpy(),    label="Predicted SoC")
        plt.fill_between(idx,
                            lower.detach().numpy(),
                            upper.detach().numpy(),
                            alpha=0.3,
                            label="95% CI")
        plt.title(f"Battery {i} SoC Predictions")
        plt.xlabel("Test point index")
        plt.ylabel("SoC (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    ## return the result mean and width
    mean_mse = np.mean(np.array(mse_global))
    mean_widths = np.mean(np.array(mean_width_global))
    mean_interval_percentage = np.mean(np.array(mean_interval_global))
    mean_mspe_width = np.mean(np.array(mspe_width_global))
    mean_mspe_percentage = np.mean(np.array(mspe_interval_global))
            


    ## here we return the mean mse and mean width
    return mean_mse, mean_widths, mean_interval_percentage, mean_mspe_width, mean_mspe_percentage




train_dfs = []
test_dfs  = []

mse_means = []
width_means = []
interval_means = []
mspe_percentage = []
mspe_widths= []

sample_size = [350,400,500]

for i in sample_size:

    ## here we will get three different df for each battery and later combine them to perform the analysis
    ## we will get the battery in alphabetical order starting with A123, CX2 and INR
    ## the nominal capacity is give here https://calce.umd.edu/battery-data#INR for each battery type
    df1 = pd.read_csv("data_battery/A123.csv")
    df1 = battery_cleaning(df=df1, battery_index=1, capacity_nominal=1.1, columns_drop=["Data_Point","Test_Time(s)","Date_Time","Step_Time(s)","Step_Index","Cycle_Index","dV/dt(V/s)","Is_FC_Data","AC_Impedance(Ohm)","ACI_Phase_Angle(Deg)"])
    ## reduce number of rows using cubic spline interpolation
    df1 = reduce_time_series(df=df1, n_points=i)

    df2 = pd.read_csv("data_battery/CX2.csv")
    df2 = battery_cleaning(df=df2, battery_index=2, capacity_nominal=1.35, columns_drop=["Data_Point","Test_Time(s)","Date_Time","Step_Time(s)","Step_Index","Cycle_Index","dV/dt(V/s)","Is_FC_Data","AC_Impedance(Ohm)","ACI_Phase_Angle(Deg)"])
    ## we reduce the number of rows using cubic splines interpolation
    df2 = reduce_time_series(df=df2, n_points=i)

    df3 = pd.read_csv("data_battery/INR.csv")
    df3 = battery_cleaning(df=df3, battery_index=3, capacity_nominal=2.0, columns_drop=["Data_Point","Test_Time(s)","Date_Time","Step_Time(s)","Step_Index","Cycle_Index","dV/dt(V/s)","Is_FC_Data","AC_Impedance(Ohm)","ACI_Phase_Angle(Deg)"])
    ## reduce the number of rows using cubic spline interpolation
    df3 = reduce_time_series(df=df3, n_points=i)


    frames = [df1, df2, df3]
    ## we join the dataframe as one dataframe
    df = pd.concat(frames, ignore_index=True)
            
    for proc_id, g in df.groupby("battery", sort=False):
                # g is already in time order if your CSV is grouped by process & time
        n       = len(g)
        cutoff  = int(n * 0.8)          # 80% point, e.g. int(501*0.8)=400
        train_dfs.append(g.iloc[:cutoff])
        test_dfs.append( g.iloc[cutoff:])
            ## we get a train and test df
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df  = pd.concat(test_dfs,  ignore_index=True)
            ## we get the parameters to pass in the df to tensors function


    feature_cols = list(df.columns[0:5])
    task_col = ["battery"]

    scalers = {}
    for p in train_df["battery"].unique():
                # mask per process
        idx_train = train_df["battery"] == p
        idx_test  = test_df["battery"]  == p

                # fit scaler on train-y of process p
        scaler = StandardScaler().fit(train_df.loc[idx_train, ["SoC"]])
        scalers[p] = scaler

                # overwrite y with scaled y
        train_df.loc[idx_train, "SoC"] = scaler.transform(train_df.loc[idx_train, ["SoC"]])
        test_df.loc[idx_test,   "SoC"] = scaler.transform(test_df.loc[idx_test,  ["SoC"]])



    X_train, y_train, task_train = df_to_tensors_soc(train_df, feature_cols, task_col)
            ## we pass the create parameter in the model
    mse_mean, width_mean, mean_interval, mspe_width, mspe_percent  = gp_model(X_train=X_train, y_train=y_train, train_task=task_train, test_df=test_df, likelihood=likelihood, training_iter=130)
            
    mse_means.append(mse_mean)
    width_means.append(width_mean)
    interval_means.append(mean_interval)
    mspe_percentage.append(mspe_percent)
    mspe_widths.append(mspe_width)


metrics = {
    "MSE":          np.array(mse_means),
    "Width":        np.array(width_means),
    "Coverage%":    np.array(interval_means),
    "Robust W":     np.array(mspe_widths),
    "Robust Cov%":  np.array(mspe_percentage),
}

print("\nMetric results over", 3, "runs:\n")
for name, arr in metrics.items():
    mean = arr.mean()
    std  = arr.std(ddof=1)
    sem  = std / np.sqrt(3)
    print(f"{name:10s}: {mean:.3f} ± {sem:.3f}   (σ={std:.3f})")

df = pd.DataFrame(metrics)
df["model"] = "ICM"   # change per‐script
df.to_csv("metrics_ICM_true.csv", index=False)
