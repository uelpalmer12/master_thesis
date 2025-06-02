## here we build the proposed model in our master thesis
## it is a combination of deep kernel learning, entity embedding to perform multitask learning

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

## then we start by building the model
## it needs to be noted that the model is made of three inside model

## define the autoencoder class
## the number of tasks, latent dimensions, embeddings dimension and input dimensions are defined globally here
n_tasks = 3
embedding_dim = 3 
input_dim = 10
latent_dim = 4

## here we define the autoencoder class to be passed in the embedding class
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        ## here we define the encoder to go from the data dim to the latent space
        ## we also use leaky relu instead of normal relu to avoid sleepy neurons problem
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.LeakyReLU(negative_slope=0.3),
            torch.nn.Linear(8, 6),
            torch.nn.BatchNorm1d(6),
            torch.nn.LeakyReLU(negative_slope=0.3),
            torch.nn.Linear(6, latent_dim)
        )
        ## we define the decoder to go from the latent space to the data dim
        self.decoder = torch.nn.Sequential(
           torch.nn.Linear(latent_dim, 6),
           torch.nn.BatchNorm1d(6),
           torch.nn.LeakyReLU(negative_slope=0.3),
           torch.nn.Linear(6, 8),
           torch.nn.BatchNorm1d(8),
           torch.nn.LeakyReLU(negative_slope=0.3),
           torch.nn.Linear(8,input_dim)
        )



## here we define our own custom kernel 
## the cosine similarity kernel

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
## here we define our model made of the autoencoder network and the embedding head
## all the parameters are trained end to end using marginal log likelihood

class MultiTaskDKL(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(MultiTaskDKL, self).__init__(train_inputs, train_targets, likelihood)
        ## here we will put in the different values we need for the different neural networks
        ## so we will need to define those variables globally
        n_tasks = 3
        embedding_dim = 3
        input_dim = 10
        latent_dim = 4
        ## here we define the mean module
        self.mean_module = gpytorch.means.ConstantMean()
        grid_size = gpytorch.utils.grid.choose_grid_size(train_inputs[0],1.0)
        ## here we define the continuous as well as embedding task covar module
        self.continuous_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 4, active_dims=[0,1,2,3]))
        self.embed_covar = (CosineSimilarityKernel(ard_num_dims = 3, active_dims = [4,5,6]))
        self.composite = (gpytorch.kernels.ProductKernel(self.continuous_covar, self.embed_covar))
        ## here we will build the various neural network
        ## we start by the auto encoder
        self.ae = AutoEncoder(input_dim= input_dim, latent_dim= latent_dim)
        ## here we build the embedder
        self.embedder = torch.nn.Embedding(num_embeddings=n_tasks, embedding_dim=embedding_dim)
        


        ## after building all of those we just build the forward function
    def forward(self, x_continuous, tasks_ids):
        
        encoded = self.ae.encoder(x_continuous)
             # if encoded has extra dims, flatten them:
        encoded = encoded.view(encoded.size(0), -1)
            ## the decoder reconstruction
        self.decoded = self.ae.decoder(encoded)

            ## then we build the embeddings
        embeddings = self.embedder(tasks_ids)
            ## once the embeddings are build
            # if embedded has a singleton middle dim:
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)
            ## here before performing gaussian process we want to train the model to have good encoding
            ## we also want to train to recognize the class
        inputs = torch.cat([encoded, embeddings], dim=1)
        
            ## here we finalize by have the mean and covar 
        mean_x = self.mean_module(encoded)
            ## the covariance take task embedding and the 
        covar_x = self.composite(inputs)
        return gpytorch.distributions.MultivariateNormal(mean=mean_x, covariance_matrix=covar_x)      






    
## here we define the likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-2))
likelihood.noise = torch.tensor(1e-2)
## after defining the likelihood we will define the function that will have the training and evaluation loops

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
        # start timing
    start = time.time()
    with gpytorch.settings.max_cg_iterations(50000), gpytorch.settings.cholesky_jitter(1e-8):


        for i in range(training_iter):
            optimizer.zero_grad()
            output_gp = model(X_train, train_task)
            recon = model.decoded
            mse = torch.nn.functional.mse_loss(recon, X_train)
            loss = -mll(output_gp, y_train) + mse
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()
    # end timing
    end = time.time()
    train_elapsed = end - start

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
        
        with gpytorch.settings.fast_pred_var(False):

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
    return mean_mse, mean_widths, mean_interval_percentage, train_elapsed, mean_mspe_width, mean_mspe_percentage



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
        print(f"iteration number {i+1} for {200} samples")
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
        ## we get the parameters to pass in the df to tensors function



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
        mse_mean, width_mean, mean_interval, t_elapsed, mspe_width, mspe_percent  = gp_model(X_train=X_train, y_train=y_train, train_task=task_train, test_df=test_df, likelihood=likelihood, training_iter=130, learning_rate=0.04)
        ## we append the means after each iteration
        mse_means.append(mse_mean)
        width_means.append(width_mean)
        interval_means.append(mean_interval)
        training_times.append(t_elapsed)
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

# plt.figure(figsize=(8,4))
# plt.plot(range(1, m+1), training_times, marker='o')
# plt.xlabel('Simulation run')
# plt.ylabel('Training time (s)')
# plt.title('GP Training Time per Simulation')
# plt.grid(True)
# plt.show()




