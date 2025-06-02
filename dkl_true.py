import time
import gpytorch.constraints
import pandas as pd
import numpy as np
import gpytorch
from sklearn.discriminant_analysis import StandardScaler
from sympy import GreaterThan
import torch
import matplotlib.pyplot as plt
from utils import *
import os

np.random.seed(195)
## then we start by building the model
## it needs to be noted that the model is made of three inside model

## define the autoencoder class
## the number of tasks, latent dimensions, embeddings dimension and input dimensions are defined globally here
n_tasks = 3
embedding_dim = 3 
input_dim = 10
latent_dim = 4

## here we define the autoencoder class to be passed in the embedding class
class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FeatureExtractor, self).__init__()
        ## here we define the encoder to go from the data dim to the latent space
        ## we also use leaky relu instead of normal relu to avoid sleepy neurons problem
        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 4),
                torch.nn.BatchNorm1d(4),
                torch.nn.LeakyReLU(negative_slope=0.3),
                torch.nn.Linear(4, 3),
                torch.nn.BatchNorm1d(3),
                torch.nn.LeakyReLU(negative_slope=0.3),
                torch.nn.Linear(3, latent_dim)
        )

class CosineSimilarityKernel(gpytorch.kernels.Kernel):
        """
        k(x, x') = x · x' / (||x|| · ||x'||)
        """

        is_stationary = False   # cosine‐similarity is not a stationary kernel

        def __init__(self, eps: float = 1e-8, **kwargs):
            """
            Args:
                eps (float): small value to avoid division by zero when normalizing.
                **kwargs: passed to gpytorch.kernels.Kernel (e.g. active_dims, batch_shape).
            """
            super().__init__(**kwargs)
            self.eps = eps

        def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
            # If only the diagonal is requested, cosine similarity of each row with itself is 1
            if diag:
                return x1.new_ones(x1.size(-2))

            # Compute norms
            x1_norm = x1.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            x2_norm = x2.norm(dim=-1, keepdim=True).clamp_min(self.eps)

            # Normalize
            x1_normalized = x1 / x1_norm
            x2_normalized = x2 / x2_norm

            # Dot‐product of unit vectors = cosine similarity
            cov = x1_normalized @ x2_normalized.transpose(-2, -1)
            return cov
  



## here we define our own custom kernel 
## the cosine similarity kernel

class MultiTaskDKL(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(MultiTaskDKL, self).__init__(train_inputs, train_targets, likelihood)
        ## here we will put in the different values we need for the different neural networks
        ## so we will need to define those variables globally
        n_tasks = 3
        embedding_dim = 3 
        input_dim = 5
        latent_dim = 3
        ## here we define the mean module
        self.mean_module = gpytorch.means.ConstantMean()
        ## here we define the continuous as well as embedding task covar module
        self.continuous_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        self.task_kern = gpytorch.kernels.IndexKernel(num_tasks=n_tasks, rank=1)


        ## here we will build the various neural network
        ## we start by the auto encoder
        self.ae = FeatureExtractor(input_dim= input_dim, latent_dim= latent_dim)
        ## here we build the embedder
        


        ## after building all of those we just build the forward function
    def forward(self, x_continuous, tasks_ids):
        
        encoded = self.ae.encoder(x_continuous)
             # if encoded has extra dims, flatten them:
        encoded = encoded.view(encoded.size(0), -1)

        mean_x = self.mean_module(encoded)
        covar_continuous = self.continuous_covar(x_continuous)
        covar_task = self.task_kern(tasks_ids)
        full_covar = covar_continuous.mul(covar_task)
        return gpytorch.distributions.MultivariateNormal(mean=mean_x, covariance_matrix=full_covar)      






    
## here we define the likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-2))
likelihood.noise = torch.tensor(1e-2)



def gp_model(X_train, y_train, train_task, test_df, likelihood, training_iter=50, learning_rate = 0.04):
        ## we start by defining the model
        model = MultiTaskDKL(train_inputs=(X_train, train_task), train_targets=y_train, likelihood=likelihood)
        ## we set up the training loop

        smoketest = ('CI' in os.environ)
        ## define the number of training iterations
        training_iterations = 2 if smoketest else training_iter

        ## set model and likelihood in training mode
        model.train()
        likelihood.train()

        all_params = list(model.parameters()) + list(likelihood.parameters())
        seen = set()
        unique_params = []
        for p in all_params:
            if id(p) not in seen:
                unique_params.append(p)
                seen.add(id(p))

        ## here we define the optimizer to uniquely optimize the parameters
        optimizer = torch.optim.Adam(unique_params, lr=learning_rate)
        ## the loss is the marginal loglikelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        ## we increase conjugate gradient for optimization
        with gpytorch.settings.max_cg_iterations(20000), gpytorch.settings.cholesky_jitter(1e-2):


            for i in range(training_iterations):
                optimizer.zero_grad()
                output_gp = model(X_train, train_task)
                loss = -mll(output_gp, y_train)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                optimizer.step()

        mse_global = []
        mean_width_global = []
        mean_interval_global = []
        mspe_interval_global = []
        mspe_width_global    = []
        alpha = 1.96
        ## we set our model in eval mode
        model.eval()
        likelihood.eval()

        for i in range(1,len(frames)+1):
            ## created the data for the evaluation
            print(f"X_test, y_test for process {i}")
            ## we use the provided test dataframe
            ## since we still have the process columns we separate the test data by process
            X_test = torch.tensor(test_df.loc[test_df["battery"]==i].drop(columns=["SoC", "battery"]).values, dtype=torch.float)
            ## do the same for target values
            y_test = torch.tensor(test_df["SoC"].loc[test_df["battery"]==i].values, dtype=torch.float)
            ## we create ids for the test where the id is i-1
            ids = torch.full((X_test.shape[0],1), fill_value=i-1, dtype=torch.long)
            
            with gpytorch.settings.fast_pred_var(True), gpytorch.settings.cholesky_jitter(1e-9):

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

            
            ## return the result mean and width
            mean_mse = np.mean(np.array(mse_global))
            mean_widths = np.mean(np.array(mean_width_global))
            mean_interval_percentage = np.mean(np.array(mean_interval_global))
            mean_mspe_width = np.mean(np.array(mspe_width_global))
            mean_mspe_percentage = np.mean(np.array(mspe_interval_global))

        ## here we return the mean mse and mean width
        

            # Plot true vs predicted
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
            


        ## here we return the mean mse and mean width
        return mean_mse, mean_widths, mean_interval_percentage, mean_mspe_width, mean_mspe_percentage

mse_means = []
width_means = []
interval_means = []
training_times = []
mspe_percentage = []
mspe_widths= []

sample_size = [350,400,500]



for i in sample_size:

    ## here we import our dataset
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

    ## then we start by building the model
    ## it needs to be noted that the model is made of three inside model

    ## define the autoencoder class
    ## the number of tasks, latent dimensions, embeddings dimension and input dimensions are defined globally here






    ## we construct the dataframe for both train and test

    train_dfs = []
    test_dfs  = []
    ## metrics

            
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


    mse_mean, width_mean, mean_interval, mspe_width, mspe_percent  = gp_model(X_train=X_train, y_train=y_train, train_task=task_train, test_df=test_df, likelihood=likelihood, training_iter=130, learning_rate=0.04)
            ## we append the means after each iteration
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

print("\nMetric results over", len(sample_size), "runs:\n")
for name, arr in metrics.items():
        mean = arr.mean()
        std  = arr.std(ddof=1)
        sem  = std / np.sqrt(len(arr))
        print(f"{name:10s}: {mean:.3f} ± {sem:.3f}   (σ={std:.3f})")


df = pd.DataFrame(metrics)
df["model"] = "dklICM"   # change per‐script
df.to_csv("metrics_dklICM_true.csv", index=False)