import glob
import os
import pickle as pkl
import re
import struct
from copy import copy

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from jax import grad, jit, lax, random, value_and_grad, vmap
from jax.scipy import special as jspec
from scipy import linalg as sla
from scipy import optimize as sopt
from scipy import stats
from tqdm import tqdm

matplotlib.use('Agg')
import pylab as pl
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set()

plt.style.use('./theeconomist.mplstyle.txt')
SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class circular_functions:
    def __init__(self):
        pass

    def _wrap(self,x):
        return jnp.mod(x + 3 * jnp.pi,2 * jnp.pi) - jnp.pi

    def circular_mean(self,samples):
        real = jnp.mean(jnp.cos(samples))
        imag = jnp.mean(jnp.sin(samples))
        return jnp.arctan2(imag,real)

    def linearise(self,data,mid_point):
        return self._wrap(data - mid_point)

class em_functions(circular_functions):
    def __init__(self):
        super().__init__()
        self.e          = jit(self.E)
        self.e_grad_pi1 = jit(grad(self.E,3))
        self.e_grad_pi2 = jit(grad(self.E,4))
        self.e_grad_mu1 = jit(grad(self.E,5))
        self.e_grad_mu2 = jit(grad(self.E,6))
        self.e_grad_sig1= jit(grad(self.E,7))
        self.e_grad_sig2= jit(grad(self.E,8))
        self.ne          = jit(self.naive_E)
        self.ne_grad_pi1 = jit(grad(self.naive_E,3))
        self.ne_grad_pi2 = jit(grad(self.naive_E,4))
        self.ne_grad_mu1 = jit(grad(self.naive_E,5))
        self.ne_grad_mu2 = jit(grad(self.naive_E,6))
        self.ne_grad_sig1= jit(grad(self.naive_E,7))
        self.ne_grad_sig2= jit(grad(self.naive_E,8))
        self.kappa      = 10
        self.max_iter   = 100000
        self.min_update = 1e-7
        self.lr         = 1e-3
            
    def calc_posteriors(self,data,pi1,mu1,mu2,sig1,sig2):
        pi2 = 1 - pi1
        resp1 = pi1/jspec.i0(sig1) * jnp.exp(sig1 * jnp.cos(data - mu1))
        resp2 = pi2/jspec.i0(sig2) * jnp.exp(sig2 * jnp.cos(data - mu2))
        # Normalise
        total_resp = resp1 + resp2
        resp1 /= total_resp
        resp2 /= total_resp
        return resp1,resp2
    
    def E(self,data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,kappa):
        pi2_t = 1 - pi1_t
        comp1 = resp1 @ (jnp.log(pi1) + sig1 * jnp.cos(data - mu1) - jnp.log(jspec.i0(sig1)))
        comp2 = resp2 @ (jnp.log(pi2) + sig2 * jnp.cos(data - mu2) - jnp.log(jspec.i0(sig2)))
        log_p_mu1 = self.log_p_mu(mu1,mu1_t,sig1_t,kappa)
        log_p_mu2 = self.log_p_mu(mu2,mu2_t,sig2_t,kappa)
        log_p_pi1 = self.log_p_pi(pi1,pi1_t)
        log_p_pi2 = self.log_p_pi(pi2,pi2_t)
        log_p_sig1= self.log_p_sig(sig1,sig1_t)
        log_p_sig2= self.log_p_sig(sig2,sig2_t)
        return comp1 + comp2 + log_p_mu1 + 10 * log_p_mu2 + log_p_pi1 + log_p_pi2 + log_p_sig1 + log_p_sig2

    def naive_E(self,data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2):
        comp1 = resp1 @ (jnp.log(pi1) + sig1 * jnp.cos(data - mu1) - jnp.log(jspec.i0(sig1)))
        comp2 = resp2 @ (jnp.log(pi2) + sig2 * jnp.cos(data - mu2) - jnp.log(jspec.i0(sig2)))
        return comp1 + comp2

    def log_p_mu(self,mu,mu_t,sig_t,kappa):
        return kappa * sig_t * jnp.cos(mu - mu_t) - jnp.log(2 * jnp.pi) - jnp.log(jspec.i0(kappa * sig_t))
    
    def log_p_pi(self,pi,pi_t):
        #alpha = 2
        #beta  = 1/pi_t
        alpha = 5
        beta = 1
        return (alpha-1)*jnp.log(pi) + (beta-1)*jnp.log(1-pi) - (jspec.gammaln(alpha) + jspec.gammaln(beta) - jspec.gammaln(alpha + beta))
    
    def log_p_sig(self,sig,sig_t):
        std = jnp.sqrt(1 - jspec.i1(sig_t) / jspec.i0(sig_t))
        alpha = 2
        beta = 1/std
        return alpha * jnp.log(beta) + (alpha - 1)*jnp.log(sig) - beta * sig - jspec.gammaln(alpha)
    
    def __norm_M_step(self,data,resp1,resp2,pi1,mu1,mu2,sig1,sig2):
        resp1_tot = jnp.sum(resp1)
        resp2_tot = jnp.sum(resp2)
        new_pi1 = 1/resp1.shape[0] * resp1_tot
        new_pi2 = 1/resp2.shape[0] * resp2_tot
        tot_pi = new_pi1 + new_pi2
        new_pi1 /= tot_pi
        new_mu1 = (resp1 @ data)/resp1_tot
        new_mu2 = (resp2 @ data)/resp2_tot
        new_sig1= jnp.sqrt((resp1 @ (data - mu1)**2)/resp1_tot)
        new_sig2= jnp.sqrt((resp2 @ (data - mu2)**2)/resp2_tot)
        return new_pi1,new_mu1,new_mu2,new_sig1,new_sig2
    
    def prior_em(self,data,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,min_update=1e-10,max_iter = 50000):
        resp1 = jnp.full_like(data,0.5)
        resp2 = jnp.full_like(data,0.5)
        cur_update = 1.0
        counter = 0
        # initialise parameters as old parameters
        pi1,mu1,mu2,sig1,sig2 = pi1_t,mu1_t,mu2_t,sig1_t,sig2_t
        while (cur_update > min_update) & (counter < max_iter):
            new_resp1,new_resp2 = self.calc_posteriors(data,pi1,mu1,mu2,sig1,sig2)
            cur_update = jnp.sum((new_resp1 - resp1) ** 2)
            pi1,mu1,mu2,sig1,sig2 = self.grad_M_step(data,new_resp1,new_resp2,pi1,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t)
            resp1 = new_resp1; resp2 = new_resp2
            counter += 1
        return resp1,resp2,pi1,mu1,mu2,sig1,sig2
          
    def grad_M_step(self,data,resp1,resp2,pi1,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t):
        counter = 0
        update = 1.0
        pi2 = 1 - pi1
        old_e = self.e(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
        while (counter < self.max_iter) & (update > self.min_update):
            grad_pi1 = self.e_grad_pi1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            grad_pi2 = self.e_grad_pi2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            grad_mu1 = self.e_grad_mu1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            grad_mu2 = self.e_grad_mu2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            grad_sig1= self.e_grad_sig1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            grad_sig2= self.e_grad_sig2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            pi1 += self.lr * grad_pi1; pi2 += self.lr * grad_pi2; tot = pi1 + pi2; pi1 /= tot; pi2 /= tot
            mu1 += self.lr * grad_mu1; mu2 += self.lr * grad_mu2
            sig1+= self.lr * grad_sig1;sig2+= self.lr * grad_sig2
            e_value = self.e(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2,pi1_t,mu1_t,mu2_t,sig1_t,sig2_t,self.kappa)
            update = (e_value - old_e) ** 2
            old_e = e_value
            counter += 1
            mu1 = self._wrap(mu1)
            mu2 = self._wrap(mu2)
        return pi1,mu1,mu2,sig1,sig2

    def grad_M_step_naive(self,data,resp1,resp2,pi1,mu1,mu2,sig1,sig2):
        counter = 0
        update = 1.0
        pi2 = 1 - pi1
        old_e = self.ne(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
        while (counter < self.max_iter) & (update > self.min_update):
            grad_pi1 = self.ne_grad_pi1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            grad_pi2 = self.ne_grad_pi2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            grad_mu1 = self.ne_grad_mu1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            grad_mu2 = self.ne_grad_mu2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            grad_sig1= self.ne_grad_sig1(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            grad_sig2= self.ne_grad_sig2(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            pi1 += self.lr * grad_pi1; pi2 += self.lr * grad_pi2; tot = pi1 + pi2; pi1 /= tot; pi2 /= tot
            mu1 += self.lr * grad_mu1; mu2 += self.lr * grad_mu2
            sig1+= self.lr * grad_sig1;sig2+= self.lr * grad_sig2
            e_value = self.ne(data,resp1,resp2,pi1,pi2,mu1,mu2,sig1,sig2)
            update = (e_value - old_e) ** 2
            old_e = e_value
            counter += 1
            mu1 = self._wrap(mu1)
            mu2 = self._wrap(mu2)
        return pi1,mu1,mu2,sig1,sig2

    def fresh_em(self,data,min_update = 1e-10,max_iter = 10000,pi1=0.5,mu1=0.1,mu2=-0.1,sig1=1.0,sig2=1.0):
        resp1 = jnp.full_like(data,0.5)
        resp2 = jnp.full_like(data,0.5)
        cur_update = 1.0
        counter = 0
        while (cur_update > min_update) & (counter < max_iter):
            new_resp1,new_resp2 = self.calc_posteriors(data,pi1,mu1,mu2,sig1,sig2)
            cur_update = jnp.sum((new_resp1 - resp1) ** 2)
            pi1,mu1,mu2,sig1,sig2 = self.grad_M_step_naive(data,new_resp1,new_resp2,pi1,mu1,mu2,sig1,sig2)
            resp1 = new_resp1; resp2 = new_resp2
            counter += 1
        return resp1,resp2,pi1,mu1,mu2,sig1,sig2
    
    def fit_posterior(self,resps,samps,start = 0.75,num_bins = 26,min_update = 1e-5,pi1=0.5,mu1=0.1,mu2=-0.1,sig1=1.0,sig2=1.0,seed=1996):
        bins = jnp.linspace(-jnp.pi,jnp.pi,num_bins + 1)
        # Get the bin mids
        bin_mids = jnp.empty(num_bins)
        for cur_bin in range(num_bins + 1):
            cur_inds = (samps > bins[cur_bin]) & (samps < bins[cur_bin + 1])
            bin_mids = bin_mids.at[cur_bin].set(self.circular_mean(samps[cur_inds]))
        start_bin = int(jnp.floor(start * num_bins))
        # Set up the order for cycling through the data
        cycle_order = jnp.empty(num_bins,dtype=jnp.int16)
        cycle_order = cycle_order.at[0:(num_bins - start_bin -1)].set(jnp.arange(num_bins - start_bin -1 ) + start_bin + 1)
        cycle_order = cycle_order.at[(num_bins - start_bin - 1):].set(jnp.arange(start_bin + 1))
        # Get the initial data
        start_data = (samps > bins[start_bin]) & (samps < bins[start_bin + 1])
        start_samp = bin_mids[start_bin]
        post_store = jnp.empty(resps.shape[0])
        pi_store   = jnp.empty(num_bins)
        mu1_store  = jnp.empty(num_bins); mu2_store = jnp.empty(num_bins)
        sig1_store = jnp.empty(num_bins); sig2_store= jnp.empty(num_bins)
        resp1,_,pi1,mu1,mu2,sig1,sig2 = self.fresh_em(resps[start_data],pi1=0.5,mu1=start_samp,mu2=-start_samp,sig1=5.0,sig2=5.0)
        post_store = post_store.at[start_data].set(resp1)
        pi_store   = pi_store.at[start_bin].set(pi1)
        mu1_store  = mu1_store.at[start_bin].set(mu1);   mu2_store  = mu2_store.at[start_bin].set(mu2)
        sig1_store = sig1_store.at[start_bin].set(sig1); sig2_store = sig2_store.at[start_bin].set(sig2)
        # Now we have the initialisation, run through the rest of the steps
        not_converged = True
        while not_converged:
            for cur_bin in tqdm(cycle_order):
                cur_inds = (samps > bins[cur_bin]) & (samps < bins[cur_bin + 1])
                cur_samp = bin_mids[cur_bin]
                resp1,_,pi1,mu1,mu2,sig1,sig2 = self.prior_em(resps[cur_inds],pi1,mu1_t=cur_samp,mu2_t=-cur_samp,sig1_t=sig1,sig2_t=sig2)
                post_store = post_store.at[cur_inds].set(resp1)
                pi_store   = pi_store.at[cur_bin].set(pi1)
                mu1_store  = mu1_store.at[cur_bin].set(mu1);   mu2_store  = mu2_store.at[cur_bin].set(mu2)
                sig1_store = sig1_store.at[cur_bin].set(sig1); sig2_store = sig2_store.at[cur_bin].set(sig2)
            not_converged = False
        """
        key = random.PRNGKey(seed)
        rand_samples = random.bernoulli(key,p=1-post_store)
        """
        rect_resp    = jnp.array(resps)
        rand_samples = post_store < 0.5
        rect_resp    = rect_resp.at[rand_samples].set(-1 * rect_resp[rand_samples])
        fig = self.display_posteriors(samps,resps,post_store,pi_store,mu1_store,mu2_store,sig1_store,sig2_store,bin_mids)
        return fig,samps,rect_resp
            
    def disp_single_fit(self,responses,post1,pi1,mu1,mu2,sig1,sig2):
        eval_pts= jnp.linspace(-jnp.pi,jnp.pi,100)
        pdf1 = jnp.exp(sig1 * jnp.cos(eval_pts - mu1))/(2 * jnp.pi * jspec.i0(sig1))
        pdf2 = jnp.exp(sig2 * jnp.cos(eval_pts - mu2))/(2 * jnp.pi * jspec.i0(sig2))
        fig,axs = plt.subplots(1,1,figsize=(16,8))
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        axs.plot(eval_pts,pdf1,lw=5 * pi1,alpha=0.25,c='k')
        axs.plot(eval_pts,pdf2,lw=5 * (1 - pi1),alpha=0.25,c='k')
        im = axs.scatter(responses,jnp.zeros(responses.shape[0]),s=200,c=post1,alpha=0.5,vmax=1,vmin=0)
        fig.colorbar(im,cax=cax,orientation='vertical')
        return fig
    
    def display_posteriors(self,samples,responses,post_store,pi_store,mu1_store,mu2_store,sig1_store,sig2_store,bin_mids):
        var1 = 1 - jspec.i1(sig1_store)/jspec.i0(sig1_store)
        var2 = 1 - jspec.i1(sig2_store)/jspec.i0(sig2_store)
        mu1_up = self._wrap(mu1_store + var1); mu1_low = self._wrap(mu1_store - var1)
        mu2_up = self._wrap(mu2_store + var2); mu2_low = self._wrap(mu2_store - var2)
        fig,axs = plt.subplots(1,1,figsize=(16,16))
        axs.scatter(samples,responses,s=50,c=post_store,alpha=0.5)
        axs.scatter(bin_mids,mu1_store,s=100,marker='x',c='r',label=r"$\mu_1$")
        axs.scatter(bin_mids,mu2_store,s=100,marker='x',c='g',label="Dist 2")
        axs.scatter(bin_mids,mu1_up,s=50,marker='v',c='r',label=r"Var[$\mu_1$]")
        axs.scatter(bin_mids,mu1_low,s=50,marker='^',c='r',label=r"Var[$\mu_1$]")
        axs.scatter(bin_mids,mu2_up,s=50,marker='v',c='g',label=r"Var[$\mu_2$]")
        axs.scatter(bin_mids,mu2_low,s=50,marker='^',c='g',label=r"Var[$\mu_2$]")
        return fig

class summary_plots(circular_functions):

    sample_rate = 44100 # This should pretty much be constant.

    def __init__(self):
        super().__init__()
        self.grad_prior_mean_cost = jit(value_and_grad(self.prior_mean_cost,1))
        self.jit_row_means_modes = jit(self.row_means_modes)

    def split_data2(self,df,seg1):
        df1 = df.iloc[:seg1]
        df2 = df.iloc[seg1:]
        return df1,df2

    def scatter_trials(self,summary,size_scale = 200):
        sample_density = summary["SampleDensity"].to_numpy()
        response_density=summary["EndMatchDensity"].to_numpy()
        match_time = size_scale * (summary["MatchEndTime"] - summary["MatchStartTime"]).to_numpy()
        fig,ax = plt.subplots(1,1,figsize=(16,16))
        min_sample = jnp.min(sample_density); max_sample = jnp.max(sample_density)
        ax.plot((min_sample,max_sample),(min_sample,max_sample),lw=2,c='k')
        ax.plot((min_sample,max_sample),(-min_sample,-max_sample),lw=2,c='k',ls='--')
        ax.scatter(sample_density,response_density,jnp.sqrt(match_time),alpha=0.5)
        ax.set_title(f"Results scatter plot")
        ax.set_ylabel("Response density")
        ax.set_xlabel("Sample density")
        return fig

    def gen_sum_hist(self,num_bins = 51):
        x = self.data["SampleDensity"].to_numpy()
        y = self.data["EndMatchDensity"].to_numpy()
        fig,_ = self.alt_hist_results(x,y,"Samples","Responses","Total histogram",num_bins = num_bins)
        return fig,_

    def compare_hist(self,df1,df2,num_bins=26):
        fig,axs = plt.subplots(1,2,figsize=(16,8),sharey=True)
        x1 = df1["SampleDensity"].to_numpy()
        x2 = df2["SampleDensity"].to_numpy()
        y1 = df1["EndMatchDensity"].to_numpy()
        y2 = df2["EndMatchDensity"].to_numpy()
        bins = jnp.linspace(-1.7,1.7,num_bins + 1)
        axs[0].hist2d(x1,y1,bins,density=True)
        axs[1].hist2d(x2,y2,bins,density=True)
        axs[0].set_ylabel("Response density")
        axs[0].set_xlabel("Sample density")
        axs[1].set_xlabel("Sample density")
        axs[0].set_title("Data segment 1")
        axs[1].set_title("Data segment 2")
        fig.tight_layout()
        return fig


    def gen_match_resp_hist(self,num_bins=51):
        x = self.data["StartMatchDensity"].to_numpy()
        y = self.data["EndMatchDensity"].to_numpy()
        fig,_ = self.alt_hist_results(x,y,"Match","Responses","Total histogram",num_bins = num_bins)
        return fig,_

    def alt_hist_results(self,x_density,y_density,x_label,y_label,title,num_bins=101,diag=False):
        # Setup for a circle
        x_density = (x_density / 1.7) * np.pi
        y_density = (y_density / 1.7) * np.pi
        # Set up the figure structure
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        #ax.axhline(0,c='b',linewidth=4)
        #ax.axvline(0,c='y',linewidth=4)
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        # Construct the elements
        diag_min = np.min(np.concatenate([x_density,y_density]))
        diag_max = np.max(np.concatenate([x_density,y_density]))
        diagonal = np.linspace(diag_min,diag_max,180)
        #ax.plot(diagonal,diagonal,c='k',ls='--',lw=2,alpha = 0.5)
        bins = np.linspace(-np.pi,np.pi,num_bins+1,endpoint=True)
        max_hist = diag_max
        data,_,_,_ = ax.hist2d(x_density,y_density,bins)
        x_hist,x_bins = np.histogram(x_density,bins)
        x_hist = x_hist * (max_hist / np.max(x_hist))
        y_hist,y_bins = np.histogram(y_density,bins)
        y_hist = y_hist * (max_hist / np.max(y_hist))
        ax_histx.bar(x_bins[:-1], x_hist, align="edge",color='b',linewidth=0,label=x_label,width = np.diff(x_bins))
        ax_histy.barh(y_bins[:-1], y_hist, height = np.diff(y_bins), align="edge",color='y',linewidth=0,label=y_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax_histx.set_title(title)
        # Make things look a little prettier
        
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['left'].set_visible(False)
        ax_histx.spines['bottom'].set_linewidth(2)
        ax_histx.get_yaxis().set_ticks([])
        
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['bottom'].set_visible(False)
        ax_histy.spines['left'].set_linewidth(2)
        ax_histy.get_xaxis().set_ticks([])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        

        x_values = np.empty(num_bins); y_values = np.empty(num_bins)
        for bin_num in range(num_bins):
            x_vals = np.logical_and(x_density>bins[bin_num],x_density<=bins[bin_num+1])
            cur_x = x_density[x_vals]
            
            cur_x_real = np.mean(np.cos(cur_x))
            cur_x_imag = np.mean(np.sin(cur_x))
            x_values[bin_num] = np.arctan2(cur_x_imag,cur_x_real)
            cur_y = y_density[x_vals]
            cur_y_real = np.mean(np.cos(cur_y))
            cur_y_imag = np.mean(np.sin(cur_y))
            y_values[bin_num] = np.arctan2(cur_y_imag,cur_y_real)
        #ax.plot(diagonal,diagonal,c='r',lw=2,alpha=1,ls='--',label="Identity")
        ax.plot(x_values,y_values,c='w',lw=2,alpha=1,ls='--',label="Mean response")
        ax.set_xticks([np.pi,-np.pi,2*np.pi/3,-2*np.pi/3,2*np.pi/6,-2*np.pi/6,0])
        ax.set_yticks([np.pi,-np.pi,2*np.pi/3,-2*np.pi/3,2*np.pi/6,-2*np.pi/6,0])
        #ax.legend()
        print(f"data_max = {np.max(data)}")
        data = data / np.sum(data)
        max_prob = np.max(data); min_prob = np.min(data)
        col_fig = pl.figure(figsize=(9,1.5))
        img = pl.imshow(data)
        pl.gca().set_visible(False)
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])
        pl.colorbar(orientation="horizontal", cax=cax,ticks=[min_prob,max_prob])
        return fig, col_fig


    def circular_error(self,val1,val2):
        abs_dif = val1 - val2
        return self._wrap(abs_dif)

    

    def dist_vs_error(self,df):
        match = df["StartMatchDensity"].to_numpy()
        samps = df["SampleDensity"].to_numpy()
        resps = df["EndMatchDensity"].to_numpy()
        start_dist = self.circular_error(samps,match)
        error      = self.circular_error(samps,resps)
        fig1,axs = plt.subplots(1,1,figsize = (16,16))
        axs.plot([-jnp.pi,jnp.pi],[-jnp.pi,jnp.pi],ls='--',c='k')
        axs.hist2d(start_dist,error,bins=21)
        axs.set_title("Start distance vs error")
        axs.set_ylabel("|Sample - Response|")
        axs.set_xlabel("|Sample - Match|")
        return fig1

    def _extents(self,f):
        delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]

    def performance(self,df,num_bins=11):
        samps  = df["SampleDensity"].to_numpy() / 1.7 * jnp.pi
        match  = df["StartMatchDensity"].to_numpy() / 1.7 * jnp.pi
        resps  = df["EndMatchDensity"].to_numpy() / 1.7 * jnp.pi
        bins   = jnp.linspace(-1.7,1.7,num_bins + 1)
        perf_mat = jnp.empty((num_bins,num_bins))
        for s_bin in range(num_bins):
            cur_s_ind = (samps > bins[s_bin]) & (samps < bins[s_bin + 1])
            for m_bin in range(num_bins):
                cur_m_ind = (match > bins[m_bin]) & (match < bins[m_bin + 1])
                cur_inds = cur_s_ind & cur_m_ind
                cur_samps = samps[cur_inds]
                cur_resps = resps[cur_inds]
                cur_errors = self.circular_error(cur_resps,cur_samps)
                perf_mat = perf_mat.at[s_bin,m_bin].set(self.circular_mean(cur_errors))
        fig,axs = plt.subplots(1,1,figsize=(16,16))
        axs.imshow(perf_mat,origin="lower")
        axs.set_ylabel("match")
        axs.set_xlabel("sample")
        return fig


    def rectify_density(self,df,num_bins = 51):
        x,y_rect,cols,_ = self._rect_data(df)
        fig1,ax1  = plt.subplots(1,1,figsize=(16,16))
        ax1.scatter(x,df["EndMatchDensity"].to_numpy(),s=200,c=cols)
        fig2,_    = self.alt_hist_results(x,y_rect,"Sample density","Response density","Rectified responses",num_bins=num_bins)
        return fig1,fig2
    
    def _rect_data(self,df):
        x = df["SampleDensity"].to_numpy()
        y_hat = df["EndMatchDensity"].to_numpy(); y_rect = copy(y_hat)
        inds  = jnp.zeros(y_hat.shape,dtype=bool)
        main_dist = jnp.abs(y_hat - x)
        off_dist  = jnp.abs(x + y_hat)
        to_flip   = off_dist < main_dist
        y_rect[to_flip] = y_rect[to_flip] * -1
        inds      = inds.at[to_flip].set(True)
        new_df = copy(df)
        new_df["EndMatchDensity"] = y_rect

        return x,y_rect,inds,new_df
    
    def remove_flipped(self,df):
        _,_,inds,new_df = self._rect_data(df)
        reset_df = df.reset_index()
        return reset_df.drop(reset_df.index[jnp.where(inds)[0]],inplace=False)

    def _update_df(self,df):
        self.data = df

    def row_means(self,df,num_bins = 51):
        samps = jnp.array(df["SampleDensity"].to_numpy()) / 1.7 * jnp.pi
        resps = jnp.array(df["EndMatchDensity"].to_numpy()) / 1.7 * jnp.pi
        mean_resps = jnp.empty(num_bins)
        mean_samps = jnp.empty(num_bins)
        bin_edges  = jnp.linspace(-jnp.pi,jnp.pi,num_bins + 1)
        bin_mids   = (bin_edges[:-1] + bin_edges[1:])/2
        for bin_val in range(num_bins):
            cur_inds = (resps > bin_edges[bin_val]) & (resps < bin_edges[bin_val + 1])
            cur_resps = resps[cur_inds]
            cur_samps = samps[cur_inds]
            mean_resps = mean_resps.at[bin_val].set(self.circular_mean(cur_resps))
            mean_samps = mean_samps.at[bin_val].set(self.circular_mean(cur_samps))
        error = mean_samps - bin_mids
        fig,axs = plt.subplots(1,1,figsize=(16,8))
        axs.axvline(0,lw=2,ls='--',c='k')
        axs.axhline(0,lw=2,ls='--',c='k')
        axs.plot(mean_resps,error,lw=2,c='y')
        
        return fig
    
    def get_bootstraps(self,df,num_bins = 51,kernel=10,cycles = 1000,seed=1996,alpha=0.05):
        samps = jnp.array(df["SampleDensity"].to_numpy()) / 1.7 * jnp.pi
        resps = jnp.array(df["EndMatchDensity"].to_numpy()) / 1.7 * jnp.pi
        key = random.PRNGKey(seed)
        row_means = jnp.empty((cycles,num_bins))
        bins = jnp.linspace(-jnp.pi,jnp.pi,num_bins + 1,endpoint=True)
        row_vals = (bins[:-1]+bins[1:])/2
        row_modes = jnp.empty((cycles,num_bins))
        cur_row_means = jnp.empty(num_bins)
        cur_row_modes = jnp.empty(num_bins)
        grid_spec = jnp.linspace(-jnp.pi,jnp.pi,1001,endpoint=True)
        grid = (grid_spec[:-1] + grid_spec[1:])/2
        for i in tqdm(range(cycles)):
            key,subkey = random.split(key)
            cur_samps  = random.choice(subkey,samps,(samps.shape[0],))
            cur_resps  = random.choice(subkey,resps,(resps.shape[0],))
            cur_means,cur_modes = self.jit_row_means_modes(cur_resps,cur_samps,bins,cur_row_means,cur_row_modes,kernel,row_vals,grid)
            row_means = row_means.at[i,:].set(cur_means)
            row_modes = row_modes.at[i,:].set(cur_modes)
        means,modes = self.jit_row_means_modes(resps,samps,bins,cur_row_means,cur_row_modes,kernel,row_vals,grid)
        return row_means,means,row_modes,modes,row_vals

    def row_means_modes(self,cur_resps,cur_samps,bins,mean_errors,mode_errors,kernel,row_vals,grid):
        real_resps = jnp.cos(cur_resps)
        imag_resps = jnp.sin(cur_resps)
        real_samps = jnp.cos(cur_samps)
        imag_samps = jnp.sin(cur_samps)
        mat = jnp.exp(kernel * (jnp.cos(jnp.expand_dims(cur_samps,0) - jnp.expand_dims(grid,1))-1))
        for cur_bin in range(bins.shape[0]-1):
            index = (cur_resps < bins[cur_bin+1]) & (cur_resps > bins[cur_bin])
            cur_real_resps = (real_resps @ index)
            cur_imag_resps = (imag_resps @ index)
            cur_real_samps = (real_samps @ index)
            cur_imag_samps = (imag_samps @ index)
            mean_errors = mean_errors.at[cur_bin].set(self._wrap(jnp.arctan2(cur_imag_samps,cur_real_samps) - jnp.arctan2(cur_imag_resps,cur_real_resps)))
            curve       = mat @ index
            mode_errors = mode_errors.at[cur_bin].set(self._wrap(grid[jnp.argmax(curve)] - row_vals[cur_bin]))
        return mean_errors, mode_errors

    def disp_bootstrap(self,mean_iters,mean,mode_iters,mode,alpha,row_vals,limit=[-jnp.pi/6,jnp.pi/6]):
        mean_low,mean_high = self.get_CI(mean_iters,alpha)
        mode_low,mode_high = self.get_CI(mode_iters,alpha)
        all_fig = plt.figure(figsize=(16,4))
        all_ax  = all_fig.add_subplot(111)
        ci_fig  = plt.figure(figsize=(16,4))
        ci_ax   = ci_fig.add_subplot(111)
        all_ax.plot(row_vals,mean_iters.T,lw=0.5,alpha=0.1,c='y')
        all_ax.plot(row_vals,mean,lw=5,alpha=0.5,c='y')
        all_ax.plot(row_vals,mode_iters.T,lw=0.5,alpha=0.1,c='b')
        all_ax.plot(row_vals,mode,lw=5,alpha=0.5,c='b')

        ci_ax.plot(row_vals,mean,lw=5,alpha=1,c='y')
        ci_ax.plot(row_vals,mode,lw=5,alpha=1,c='b')
        ci_ax.fill_between(row_vals,mode_high,mode_low,color='b',alpha=0.5)
        ci_ax.fill_between(row_vals,mean_high,mean_low,color='y',alpha=0.5)
        ci_ax.set_xticks([-jnp.pi,-2*jnp.pi/3,-2*jnp.pi/6,0,2*jnp.pi/3,2*jnp.pi/6,jnp.pi])
        ci_ax.set_yticks([-jnp.pi/6,0,jnp.pi/6])
        all_ax.set_xticks([-jnp.pi,-2*jnp.pi/3,-2*jnp.pi/6,0,2*jnp.pi/3,2*jnp.pi/6,jnp.pi])
        all_ax.set_yticks([-jnp.pi/6,0,jnp.pi/6])
        ci_ax.set_ylim(limit)
        all_ax.set_ylim(limit)
        return all_fig,ci_fig
    
    def get_CI(self,iters,alpha):
        num_iters = iters.shape[0]
        num_bins  = iters.shape[1]
        low_ind = int((alpha/2) * num_iters)
        high_ind= int((1 - alpha/2) * num_iters)
        high = jnp.empty(num_bins)
        low  = jnp.empty(num_bins)
        for bin_num in range(num_bins):
            cur_data = jnp.sort(iters[:,bin_num])
            low = low.at[bin_num].set(cur_data[low_ind])
            high= high.at[bin_num].set(cur_data[high_ind])
        return low,high
    
    def comp_mean_mode(self,mean_errors,mode_errors,permutes,seed=1996):
        key = random.PRNGKey(seed)
        overall_mean = mean_errors @ mean_errors.T
        overall_mode = mode_errors @ mode_errors.T
        mean_mode    = jnp.empty(permutes)
        cur_mode_mix = jnp.empty(mean_errors.shape[0])
        for ind in tqdm(range(permutes)):
            key,subkey = random.split(key)
            selection = random.bernoulli(subkey,p=0.5,shape=(mean_errors.shape[0],))
            cur_mode_mix = cur_mode_mix.at[selection].set(mean_errors[selection])
            cur_mode_mix = cur_mode_mix.at[~selection].set(mode_errors[~selection])
            mean_mode = mean_mode.at[ind].set(cur_mode_mix @ cur_mode_mix.T)
        fig,axs = plt.subplots(1,1,figsize=(16,8))
        axs.axvline(overall_mean,lw=2,c='y')
        axs.axvline(overall_mode,lw=2,c='b')
        mean_mode = jnp.sort(mean_mode)
        mode_low   = mean_mode[int(0.05 * permutes)]
        axs.axvline(mode_low,lw=2,ls='--',c='b')
        _,mode_bins = jnp.histogram(mean_mode,25)
        log_mode_bins = jnp.logspace(jnp.log10(mode_bins[0]),jnp.log10(mode_bins[-1]),len(mode_bins))
        axs.hist(np.array(mean_mode),log_mode_bins,facecolor='b')
        axs.set_xscale('log')
        return fig

    def gen_llh(self,df,num_bins=21):
        samps = df["SampleDensity"].to_numpy() / 1.7 * jnp.pi
        resps = df["EndMatchDensity"].to_numpy() / 1.7 * jnp.pi
        bins  = jnp.linspace(-jnp.pi,jnp.pi,num_bins + 1)
        hist,_,_ = jnp.histogram2d(samps,resps,bins)
        hist = hist.T
        s_vals = jnp.sum(hist,axis=0)
        llh_hist = hist @ jnp.diag(1/s_vals)
        return llh_hist
    
    def col_norm(self,mat):
        col_sums = jnp.sum(mat,axis=0)
        return mat @ jnp.diag(1/col_sums)
    
    def row_norm(self,mat):
        row_sums = jnp.sum(mat,axis=1)
        return jnp.diag(1/row_sums) @ mat
    
    def gen_posterior(self,llh,prior):
        joint = llh @ jnp.diag(prior)
        return self.row_norm(joint)
    
    def prior_mean_cost(self,llh,log_prior,s_vals):
        prior = jnp.exp(log_prior)
        posterior = self.gen_posterior(llh,prior)
        return self.post_mean_error(posterior,s_vals)
    
    def posterior_means(self,posterior,s_vals):
        real = posterior @ jnp.cos(s_vals)
        imag = posterior @ jnp.sin(s_vals)
        return jnp.arctan2(imag,real)
    
    def llh_means(self,llh,s_vals):
        norm_llh = self.row_norm(llh)
        return self.posterior_means(norm_llh,s_vals)

    def post_mean_error(self,posterior,s_vals):
        post_means = self.posterior_means(posterior,s_vals)
        dif = post_means - s_vals
        return dif @ dif.T
    
    def opt_prior_mean(self,df,num_bins = 21,lr=1e-6,max_iter = 1e6,report = 100,min_error = 1e-6,min_step = 5e-6):
        bins = jnp.linspace(-jnp.pi,jnp.pi,num = num_bins + 1)
        s_vals = (bins[:-1] + bins[1:])/2
        counter = 0
        llh = self.gen_llh(df,num_bins)
        start_prior = jnp.full(num_bins,1/num_bins)
        log_prior = jnp.log(start_prior)
        error = 1.0
        error_store = []
        iter_store  = []
        step = 1.0
        while (counter < max_iter) & (error > min_error) & (step > min_step):
            error,grad = self.grad_prior_mean_cost(llh,log_prior,s_vals)
            log_prior -= lr * grad
            step = grad @ grad.T
            if jnp.mod(counter,report) == 0:
                print(f"Iteration - {counter}, error - {error:.5f}, step-size - {step:.5f}")
                error_store.append(error)
                iter_store.append(counter)
            counter += 1
            

        fig,axs = plt.subplots(2,2,figsize = (16,16))
        axs[0,0].plot(s_vals,start_prior,lw=2,c='r',label="start prior")
        axs[0,0].plot(s_vals,jnp.exp(log_prior),lw=2,c='b',label="final prior")
        axs[0,0].legend()
        axs[1,0].plot(iter_store,error_store,lw=2,c='k')
        axs[1,0].set_xlabel("Iteration")
        posterior = self.gen_posterior(llh,jnp.exp(log_prior))
        axs[0,1].imshow(posterior,origin="lower")
        start_posterior = self.gen_posterior(llh,start_prior)
        start_means  = self.posterior_means(start_posterior,s_vals)
        learnt_means = self.posterior_means(posterior,s_vals)
        axs[1,1].plot(s_vals,start_means - s_vals,lw=2,c='r',label="Original means")
        axs[1,1].plot(s_vals,learnt_means - s_vals,lw=2,c='b',label='Learnt means')
        axs[1,1].legend()
        axs[1,1].set_xlabel("Response value")
        axs[1,1].set_ylabel(r"$\hat{R} - R$")
        return fig,jnp.exp(log_prior)
            
class pt_handler(summary_plots,em_functions):

    def __init__(self,subject,path="../Data/",lfl=8,lfh=12):
        super().__init__()
        self.subject = subject
        self.path = path
        self.full_path = os.path.join(self.path,self.subject)
        self.summary_data,self.file_counts = self._load_summary(self.full_path)
        self.data = self.summary_data[0]
        self.lfl = lfl # Log-Frequency-Low
        self.lfh = lfh # Log-Frequency-High
        # Jax-ed functions
        self.map_const_env = vmap(self._construct_env,[0,0,0,None,None])
        self.flatten_buffers = jit(self._flatten_buffers)
        
    def _load_summary(self,path):
        summary_files = glob.glob(os.path.join(self.full_path,"summary*.csv"))
        summary_data = []
        file_counts  = []
        for summary in summary_files:
            data = pd.read_csv(summary)
            summary_data.append(data)
            index = int(summary.split("summary")[-1].split(".csv")[0],base=10)
            file_counts.append(index)
        return summary_data,file_counts
    
    def _construct_env(self,time_offset,density,drift,time_ind,freq_inds):
        time = time_ind + time_offset
        x    = (self.lfh - self.lfl) * (freq_inds / freq_inds[-1,-1] + 1)
        w_p  = drift * time
        a    = 1 + jnp.sin(2 * jnp.pi * (w_p + density * x) + jnp.pi/2)
        return a
    
    def construct_trial(self,trial_num,summary_file,file_count):
        # Extract statistics
        cur_trial = summary_file.iloc[trial_num]
        cur_sample_start = cur_trial["SampleStartTime"]
        cur_sample_end   = cur_trial["SampleEndTime"]
        cur_match_start  = cur_trial["MatchStartTime"]
        cur_match_end    = cur_trial["MatchEndTime"]
        # Construct the envelopes
        buffer_info = pd.read_csv(os.path.join(self.full_path,f"WavParams{file_count}.csv"))
        buffer_times = buffer_info["CommonTime"]
        cur_sample_inds = buffer_info[(buffer_info["CommonTime"] >= cur_sample_start) & (buffer_info["CommonTime"] <= cur_sample_end)]
        cur_match_inds  = buffer_info[(buffer_info["CommonTime"] >= cur_match_start) & (buffer_info["CommonTime"] <= cur_match_end)]
        cur_sample_buffers,sample_spectrum = self._gen_envs(cur_sample_inds)
        cur_match_buffers,match_spectrum  = self._gen_envs(cur_match_inds)
        sample_spectrum = jnp.reshape(sample_spectrum,(sample_spectrum.shape[0] * sample_spectrum.shape[1]))
        match_spectrum = jnp.reshape(match_spectrum,(match_spectrum.shape[0] * match_spectrum.shape[1]))
        cur_sample = self.flatten_buffers(cur_sample_buffers,jnp.empty((cur_sample_buffers.shape[1],cur_sample_buffers.shape[0]*cur_sample_buffers.shape[-1])))
        cur_match  = self.flatten_buffers(cur_match_buffers,jnp.empty((cur_match_buffers.shape[1],cur_match_buffers.shape[0]*cur_match_buffers.shape[-1])))
        # Extract the waveforms
        wavefile = os.path.join(self.full_path,f"waveform{file_count}.bin")
        sample_wave = self._extract_waveform(wavefile,cur_sample_inds)
        match_wave  = self._extract_waveform(wavefile,cur_match_inds)
        return cur_sample,cur_match,sample_wave,match_wave,sample_spectrum,match_spectrum

    def display_env(self,sample_env,match_env,sample_wave,match_wave,sample_spectrum,match_spectrum):
        time_ratio = match_wave.shape[0] / sample_wave.shape[0]
        fig,axs = plt.subplots(1,2,figsize = (4*time_ratio,2),gridspec_kw = {'width_ratios':[1, time_ratio]})
        freqs = jnp.linspace(self.lfl,self.lfh,sample_env.shape[0])
        sample_time = jnp.arange(sample_wave.shape[0])
        match_time  = jnp.arange(match_wave.shape[0])
        axs[0].imshow(sample_env,aspect='auto',interpolation='none',extent = self._extents(sample_time) + self._extents(freqs),origin="lower")
        axs[1].imshow(match_env,aspect='auto',interpolation='none',extent = self._extents(match_time) + self._extents(freqs),origin="lower")
        axs[0].set_ylabel("Log frequency")
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        fig.tight_layout()
        return fig

    def display_trial(self,sample_env,match_env,sample_wave,match_wave,sample_spectrum,match_spectrum):
        time_ratio = match_wave.shape[0] / sample_wave.shape[0]
        fig,axs = plt.subplots(3,2,figsize = (4*time_ratio,6),gridspec_kw = {'width_ratios':[1, time_ratio]})
        freqs = jnp.linspace(self.lfl,self.lfh,sample_env.shape[0])
        sample_time = jnp.arange(sample_wave.shape[0])
        match_time  = jnp.arange(match_wave.shape[0])
        axs[0,0].plot(sample_time,sample_spectrum,lw=2,c='r')
        axs[0,1].plot(match_time,match_spectrum,lw=2,c='r')
        axs[0,1].axhline(0,lw=2,c='k'); axs[0,1].axhline(-1.7,lw=2,c='k'); axs[0,1].axhline(1.7,lw=2,c='k')
        axs[1,0].imshow(sample_env,aspect='auto',interpolation='none',extent = self._extents(sample_time) + self._extents(freqs),origin="lower")
        axs[2,0].plot(sample_time,sample_wave,lw=0.2)
        axs[1,1].imshow(match_env,aspect='auto',interpolation='none',extent = self._extents(match_time) + self._extents(freqs),origin="lower")
        axs[2,1].plot(match_time,match_wave,lw=0.2)
        axs[0,0].set_ylabel("Density")
        axs[1,0].set_ylabel("Log frequency")
        axs[1,0].set_title("Sample")
        axs[1,1].set_title("Match")
        axs[2,0].set_ylabel("Amplitude")
        axs[2,0].set_xlabel("Samples")
        axs[2,1].set_xlabel("Samples")
        max_amp = jnp.max(jnp.array([jnp.nanmax(sample_wave),jnp.nanmax(match_wave)]))
        min_amp = jnp.max(jnp.array([jnp.nanmin(sample_wave),jnp.nanmin(match_wave)]))
        axs[0,0].set_xlim(0,sample_wave.shape[0]-1)
        axs[0,1].set_xlim(0,match_wave.shape[0]-1)
        axs[0,0].set_ylim(-1.7,1.7)
        axs[0,1].set_ylim(-1.7,1.7)
        axs[2,0].set_xlim(0,sample_wave.shape[0]-1)
        axs[2,1].set_xlim(0,match_wave.shape[0]-1)
        axs[2,0].set_ylim(min_amp,max_amp)
        axs[2,1].set_ylim(min_amp,max_amp)
        axs[1,0].set_xticks([])
        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([])
        axs[0,1].set_yticks([])
        axs[0,0].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,0].set_yticks([-1.7,0,1.7])
        fig.tight_layout()
        return fig

    def _flatten_buffers(self,buffer_array,new_array):
        for i in range(buffer_array.shape[0]):
            new_array = new_array.at[:,i*buffer_array.shape[-1]:(i+1)*buffer_array.shape[-1]].set(buffer_array[i,:,:])
        return new_array
    
    def _extract_waveform(self,wavname,df):
        f = open(wavname,'rb')
        buffer_size = jnp.unique(df["BufferSize"].to_numpy())
        if buffer_size.shape[0] > 1:
            raise ValueError("Changing matrix width")
        buffer_start = df.index[0]
        num_buffers  = len(df)
        wav_store    = jnp.empty((buffer_size[0] * num_buffers))
        for i in range(num_buffers):
            start_byte = (i + buffer_start) * 8 * buffer_size[0]
            f.seek(start_byte,0)
            cur_buffer_bin = f.read(buffer_size[0] * 8)
            wav_store = wav_store.at[i*buffer_size[0] : (i+1)*buffer_size[0]].set(struct.unpack('d'*buffer_size[0],cur_buffer_bin))
        f.close()
        return wav_store 

    def _prep_env_params(self,old_drift,drift,old_density,density,buffer):
        num_buffers = old_drift.shape[0]
        drift_mat   = jnp.empty((num_buffers,buffer))
        density_mat = jnp.empty((num_buffers,buffer))
        for ind,params in enumerate(zip(old_drift,drift,old_density,density)):
            drift_mat = drift_mat.at[ind,:].set(jnp.linspace(params[0],params[1],buffer))
            density_mat = density_mat.at[ind,:].set(jnp.linspace(params[2],params[3],buffer))
        return drift_mat,density_mat

    def _gen_envs(self,df):
        freqs = jnp.unique(df["NumFreqs"].to_numpy())
        if freqs.shape[0] > 1:
            raise ValueError("Changing matrix height")
        buffer_size = jnp.unique(df["BufferSize"].to_numpy())
        if buffer_size.shape[0] > 1:
            raise ValueError("Changing matrix width")
        dur = buffer_size[0] / self.sample_rate
        time_inds = jnp.expand_dims(jnp.linspace(0,dur,buffer_size[0],endpoint=False),0)
        freq_inds = jnp.expand_dims(jnp.arange(freqs[0]),1)
        drift_mat,density_mat = self._prep_env_params(df["OldDrift"].to_numpy(),df["Drift"].to_numpy(),
                                                      df["OldDensity"].to_numpy(),df["Density"].to_numpy(),
                                                      buffer_size[0])
        return self.map_const_env(df["TimeOffset"].to_numpy(),density_mat,drift_mat,time_inds,freq_inds),density_mat

    def gen_plots(self):
        scatter = self.scatter_trials(self.summary_data[0])
        scatter.savefig(f"{self.subject}_summary.png")
        hist,_    = self.gen_sum_hist()
        hist.savefig(f"{self.subject}_summary_hist.png")
        error = self.dist_vs_error(self.summary_data[0])
        error.savefig(f"{self.subject}_dist_vs_error.png")
        match,_ = self.gen_match_resp_hist()
        match.savefig(f"{self.subject}_match_impact_hist.png")
        outputs = self.construct_trial(100,self.summary_data[0],0)
        fig = self.display_trial(*outputs)
        fig.savefig(f"{self.subject}_trial_100.png")

class total_handler(summary_plots,em_functions):
    def __init__(self,path="../Data/",lfl=8,lfh=12):
        super().__init__()
        self.path = path
        self.lfl = lfl
        self.lfh = lfh

    def get_all_df(self,reject = []):
        pt_summary = []
        pt = list(filter(None,[x[0].replace(self.path,'') for x in os.walk(self.path)]))
        for cur_pt in pt:
            if cur_pt in reject:
                continue
            cur_handler = pt_handler(cur_pt,self.path,self.lfl,self.lfh)
            pt_summary.append(cur_handler.summary_data[0])
        self.data = pd.concat(pt_summary)
        return self.data

    def get_all_df_split2(self,seg1_num,reject = []):
        pt1_summary = []
        pt2_summary = []
        pt = list(filter(None,[x[0].replace(self.path,'') for x in os.walk(self.path)]))
        for cur_pt in pt:
            if cur_pt in reject:
                continue
            cur_handler = pt_handler(cur_pt,self.path,self.lfl,self.lfh)
            seg1,seg2   = self.split_data2(cur_handler.summary_data[0],seg1_num)
            pt1_summary.append(seg1)
            pt2_summary.append(seg2)
        self.seg1 = pd.concat(pt1_summary)
        self.seg2 = pd.concat(pt2_summary)
        return self.seg1,self.seg2

     

def test_gauss(mean,std,num_samp,num_bins = 30):
    samps = np.random.normal(mean,std,num_samp)
    # wrap
    samps = np.mod(samps + 3 * 1.7, 2 * 1.7) - 1.7
    fig,axs = plt.subplots(1,1)
    axs.hist(samps,num_bins)
    fig.savefig("Gauss_wrap_test.png")


def main():
    all_data = total_handler()
    df = all_data.get_all_df(reject=  ["pt_NULL"])
    samps  = df["SampleDensity"].to_numpy() / 1.7 * jnp.pi
    resps  = df["EndMatchDensity"].to_numpy() / 1.7 * jnp.pi
    fig,_,resp = all_data.fit_posterior(resps,samps,num_bins = 51)
    fig.savefig("fit_posteriors_tight_mean_prior.png")
    resps = resp / jnp.pi * 1.7
    fig,_ = all_data.alt_hist_results(df["SampleDensity"].to_numpy(),resps,"Samples","Rectified Responses","Rectified Histogram",num_bins = 51)
    fig.savefig("Rectified_histogram_tight_mean_prior.png")
    


    

    
    
if __name__ == "__main__":
    main()
