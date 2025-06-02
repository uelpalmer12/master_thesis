## the code here will be used to build a icm model
## the code wll not be fundamentally too different from the code 
## we will use to build a lcm model
import os
import time
import gpytorch.constraints
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import gpytorch
from utils import *
import matplotlib.pyplot as plt


## here we will start by defining a gaussian process model
n_tasks = 3

class MultiTaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(MultiTaskGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ## here we will build the covariance for the model
        self.covar_rq = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.RQKernel(ard_num_dims = 5, active_dims = [0,2,4,6,8]),gpytorch.kernels.LinearKernel(ard_num_dims = 5, active_dims = [0,2,4,6,8]))
        self.covar_rbf = gpytorch.kernels.RBFKernel(ard_num_dims = 5, active_dims = [1,3,5,7,9]) 
        self.composite = gpytorch.kernels.ProductKernel(self.covar_rq, self.covar_rbf)
        self.task_covar = gpytorch.kernels.IndexKernel(num_tasks=3, rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.composite(x)
        covar_task = self.task_covar(i)
        covar = (covar_x).mul(covar_task)
        return  gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-1))
likelihood.noise = torch.tensor(1e-1)


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




def gp_model(X_train, y_train, train_task, test_df, likelihood, training_iter=50, learning_rate=0.04):
    model = MultiTaskGP((X_train, train_task), y_train, likelihood)

    ## here we will start by doing the training loop
    smoke_test = ('CI' in os.environ)
    training_iterations = 2 if smoke_test else training_iter

    ## set the model in training mode 
    model.train()
    likelihood.train()

    ## set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 
    # start timing
    start = time.time()
    for i in range(training_iterations):
        with gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.cholesky_jitter(1e-2):
            optimizer.zero_grad() # Zero gradients from previous iteration
            output = model(X_train, train_task) # Output from model
            loss = -mll(output, y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

    # end timing
    end = time.time()
    train_elapsed = end - start
    ## before evaluation we process the test_df so that each test df is separated
    #test_dfs = {}
    ## we init the list 
    mse_global = []
    mean_width_global = []
    mean_interval_global = []
    mspe_interval_global = []
    mspe_width_global    = []
    alpha = 1.96


    ## we set our model in eval mode
    model.eval()
    likelihood.eval()

    for i in range(1,n_tasks+1):
        ## created the data for the evaluation
        print(f"X_test, y_test for process {i}")
        ## we use the provided test dataframe
        ## since we still have the process columns we separate the test data by process
        X_test = torch.tensor(test_df.loc[test_df["process"]==i].drop(columns=["y", "process"]).values, dtype=torch.float)
        ## do the same for target values
        y_test = torch.tensor(test_df["y"].loc[test_df["process"]==i].values, dtype=torch.float)
        ## we create ids for the test where the id is i-1
        ids = torch.full((X_test.shape[0],1), fill_value=i-1, dtype=torch.long)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(1e-3):

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
    return mean_mse, mean_widths, mean_interval_percentage, mean_mspe_width, mean_mspe_percentage






## number of simulation studies
m = 50
## time series points
t = [50, 100, 500]
## initialize the mean mse list and mean width list
mse_means = []
width_means = []
interval_means = []
training_times = []
mspe_percentage = []
mspe_widths= []
## we focus on replication for now
for i in range(0, m):
        print(f"iteration number {i+1} for {200} sample size")
    #for j in t:
        ## here for each iteration we basically have a neew seed so that the data used for the model will be different each time
        ## the number for the random seed is made for both the iteration and sample size
        rng = np.random.default_rng(195 * i * 200)

        x_bases = [                     # same 10 baselines
            rng.uniform(6, 7), rng.poisson(9, 1), rng.gamma(9, 3),
            rng.uniform(30, 31), rng.uniform(3, 4), rng.normal(1, 1),
            rng.uniform(6, 7), rng.beta(6, 7), rng.normal(6, 1),
            rng.uniform(20, 24),
        ]

        specs = [
            dict(y0=rng.uniform(4, 5),
                coeffs=[0.5, 0.1, -0.2, 0.2, 1.2, 0.43, 0.4, 1.2, -0.6, -0.4],
                lag=-0.4),
            dict(y0=rng.uniform(7, 8),
                coeffs=[3.7, 5.1, -7.2, 0.8, 1.2, 3.3, 0.5, 1.2, -0.6, -3.9],
                lag=0.1),
            dict(y0=rng.uniform(7, 8),
                coeffs=[1.7, 0.1, -6.2, 3.4, 2.2, 5.3, 1.5, 3.2, -0.6, -1.9],
                lag=0.7),
        ]
        ## with the specification and each dataframe we create a big dataframe for all the processes
        frames = []
        for k, s in enumerate(specs, 1):
            df = simulate_process(**s, x_bases=x_bases, T=200, burnin=200, rng=rng)
            df["process"] = k
            frames.append(df)
        ## dataframe concatenated 
        df = pd.concat(frames, ignore_index=True)
        ## we will separate them in train and test
        train_dfs = []
        test_dfs  = []
        
        for proc_id, g in df.groupby("process", sort=False):
            # g is already in time order if your CSV is grouped by process & time
            n       = len(g)
            cutoff  = int(n * 0.8)          # 80% point, e.g. int(501*0.8)=400
            train_dfs.append(g.iloc[:cutoff])
            test_dfs.append( g.iloc[cutoff:])
        ## we get a train and test df
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df  = pd.concat(test_dfs,  ignore_index=True)

        ## here we will fit the scaler separately for each process on the y variable


        scalers = {}
        for p in train_df["process"].unique():
            # mask per process
            idx_train = train_df["process"] == p
            idx_test  = test_df["process"]  == p

            # fit scaler on train-y of process p
            scaler = StandardScaler().fit(train_df.loc[idx_train, ["y"]])
            scalers[p] = scaler

            # overwrite y with scaled y
            train_df.loc[idx_train, "y"] = scaler.transform(train_df.loc[idx_train, ["y"]])
            test_df.loc[idx_test,   "y"] = scaler.transform(test_df.loc[idx_test,  ["y"]])

        
        ## we get the parameters to pass in the df to tensors function
        feature_cols = [f"v{i}" for i in range(1,11)]
        task_col = ["process"]


        X_train, y_train, task_train = df_to_tensors(train_df, feature_cols, task_col)
        ## we pass the create parameter in the model
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

print("\nMetric results over", m, "runs:\n")
for name, arr in metrics.items():
    mean = arr.mean()
    std  = arr.std(ddof=1)
    sem  = std / np.sqrt(m)
    print(f"{name:10s}: {mean:.3f} ± {sem:.3f}   (σ={std:.3f})")




