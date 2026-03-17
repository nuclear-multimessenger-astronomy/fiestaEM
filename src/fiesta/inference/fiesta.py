import copy
import json
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Float, Array, PRNGKeyArray

from fiesta.conversions import mag_app_from_mag_abs
from fiesta.inference.lightcurve_model import LightcurveModel
from fiesta.inference.prior import Prior
from fiesta.inference.likelihood import EMLikelihood
from fiesta.logging import logger
from fiesta.plot import corner_plot, LightcurvePlotter
from fiesta.inference.systematic import setup_systematics_basic, setup_systematic_from_file

from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

try:
    import numpyro
    import numpyro.distributions as npdist
    from numpyro.infer import SVI as NumpyroSVI, Trace_ELBO
    HAS_NUMPYRO = True
except ImportError:
    HAS_NUMPYRO = False

# see https://github.com/kazewong/flowMC/blob/main/src/flowMC/resource_strategy_bundle/RQSpline_MALA.py#L22
# for all the other arguments that can be set to the strategy-resource bundle
default_bundle_hyperparameters = {
        "n_local_steps": 50,
        "n_global_steps": 200,
        "n_training_loops": 20,
        "n_production_loops": 15,
        "n_epochs": 100,
        "rq_spline_n_layers": 4,
        "rq_spline_hidden_units": [64, 64],
        "rq_spline_n_bins": 8,
        "mala_step_size": 2e-3,
        "learning_rate": 4e-4,
        "n_max_examples": 10_000,
        "n_NFproposal_batch_size": 10_000,
        "chain_batch_size": 100,
        "batch_size": 10_000,
        "verbose": True,
        }


class Fiesta(object):
    """
    Master inference class for interfacing with flowMC.

    Args:
        "likelihood": "(EMLikelihood) likelihood object used for the inference",
        "prior": "(Prior) prior object used for the inference. It has to contain the parameters needed to evaluate likelihood.evaluate().",
        "error_budget": "(float) fixed systematic error to use in the inference in mag. Defaults to 0.3 but is ignored when systematics file is provided.",
        "systematics_file": "(str) path to the .yaml file that provides the setup for the systematic uncertainty parameters. Will overwrite error_budget.",
        "seed": "(int) Value of the random seed used.",
        "n_chains": "(int) Number of chains to be run in parallel by the flowMC sampler.",
        "num_layers": "(int) Number of hidden layers of the NF",
        "hidden_size": "List[int, int] Sizes of the hidden layers of the NF",
        "num_bins": "(int) Number of bins used in MaskedCouplingRQSpline",
        "local_sampler_arg": "(dict) Additional arguments to be used in the local sampler",
        "n_walkers_maximize_likelihood": "(int) Number of walkers used in the maximization of the likelihood with the evolutionary optimizer",
        "n_loops_maximize_likelihood": "(int) Number of loops to run the evolutionary optimizer in the maximization of the likelihood",
        "which_local_sampler": "(str) Name of the local sampler to use",
    """
    
    likelihood: EMLikelihood
    prior: Prior

    def __init__(self,
                 likelihood: EMLikelihood,
                 prior: Prior,
                 outdir: str = "./outdir/",
                 error_budget: float = 0.3,
                 systematics_file: str | None = None,
                 seed: int = 42,
                 n_chains: int = 200,
                 sampler: str = "flowmc",
                 svi_num_iter: int = 10_000,
                 svi_step_size: float = 0.001,
                 svi_num_samples: int = 1000,
                 **kwargs):

        self.likelihood = likelihood
        self.prior = prior
        self.sampler_type = sampler

        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        self.rng_key = jax.random.PRNGKey(seed)

        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX...")

        # setup the systematic uncertainty
        if systematics_file is not None:
            self.likelihood, self.prior = setup_systematic_from_file(self.likelihood, self.prior, systematics_file)
        else:
            self.likelihood, self.prior = setup_systematics_basic(self.likelihood, self.prior, error_budget)

        if sampler == "svi":
            if not HAS_NUMPYRO:
                raise ImportError("numpyro is required for sampler='svi'. Install with: pip install numpyro")
            self.svi_num_iter = svi_num_iter
            self.svi_step_size = svi_step_size
            self.svi_num_samples = svi_num_samples
            self.Sampler = None  # no flowMC sampler
            logger.info(f"Using numpyro SVI sampler ({svi_num_iter} iterations, lr={svi_step_size})")
        else:
            # flowMC sampler (existing behavior)
            self.bundle_hyperparameters = default_bundle_hyperparameters
            for key, value in kwargs.items():
                if key in self.bundle_hyperparameters:
                    self.bundle_hyperparameters[key] = value

            self.rng_key, subkey = jax.random.split(self.rng_key)
            bundle = RQSpline_MALA_Bundle(
                rng_key=subkey,
                n_chains=n_chains,
                n_dims=self.prior.n_dim,
                logpdf=self.log_posterior,
                **self.bundle_hyperparameters)

            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.Sampler = Sampler(
                self.prior.n_dim,
                n_chains,
                subkey,
                resource_strategy_bundles=bundle,
            )
        logger.info(f"Initializing Fast Inference of Electromagnetic Transients with JAX... DONE")

    def log_posterior(self, params: Float[Array, "n_dims"], data: dict[str, any]) -> Float:
        prior_params = self.prior.add_name(params.T)
        log_prior = self.prior.log_prob(prior_params)
        log_posterior = self.likelihood.evaluate(self.prior.transform(prior_params), data) + log_prior
        return log_posterior

    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        """
        Starts the sampling algorithm to obtain the posterior. After running, the posterior samples are stored as ``.posterior_samples`` attribute.

        Args:
            key (PRNGKeyArray): Random seed to start sampling.
            initial_guess (Array, optional): Initial posisions of the chains. If empty, will get initial position as random samples from the prior.
        """
        if self.sampler_type == "svi":
            self._sample_svi(key)
            return

        if initial_guess.size == 0:
            initial_guess_named = self.prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([initial_guess_named[key] for key in self.prior.naming]).T

        logger.info(f"Starting sampling.")
        start_time = time.perf_counter()
        self.Sampler.sample(initial_guess, data={"data": jnp.zeros(self.prior.n_dim)}) # the data argument is ignored because data is setup in the likelihood
        end_time = time.perf_counter()
        logger.info(f"Sampling finished. Sampling took {end_time-start_time:.2f} seconds.")

        # setup the production samples
        samples = self.Sampler.resources["positions_production"].data
        log_prob = self.Sampler.resources["log_prob_production"].data

        samples = samples.reshape(-1, self.prior.n_dim).T
        self.posterior_samples = self.prior.add_name(samples)
        self.posterior_samples["log_prob"] = log_prob.reshape(-1,)
        self.posterior_samples["log_likelihood"] = self.posterior_samples["log_prob"] - self.prior.log_prob(self.posterior_samples)

    def _sample_svi(self, key: PRNGKeyArray):
        """Run numpyro SVI to approximate the posterior.

        Constructs a numpyro model that wraps self.log_posterior (reusing the
        existing likelihood + prior) and a diagonal-normal guide with
        per-parameter learned location and scale.  After optimisation,
        ``self.svi_num_samples`` draws are produced and stored in
        ``self.posterior_samples`` exactly as the flowMC path does.
        """
        n_dim = self.prior.n_dim

        # Extract parameter names and bounds from the prior
        # CompositePrior has .priors list; bare Prior wraps itself
        sub_priors = getattr(self.prior, 'priors', [self.prior])
        param_names = []
        prior_mins = []
        prior_maxs = []
        prior_means = []
        prior_stds = []
        for sub_prior in sub_priors:
            for name in sub_prior.naming:
                param_names.append(name)
                xmin = float(sub_prior.xmin) if hasattr(sub_prior, 'xmin') else -10.0
                xmax = float(sub_prior.xmax) if hasattr(sub_prior, 'xmax') else 10.0
                prior_mins.append(xmin)
                prior_maxs.append(xmax)
                prior_means.append(0.5 * (xmin + xmax))
                prior_stds.append(0.25 * (xmax - xmin))

        prior_mins_j = jnp.array(prior_mins)
        prior_maxs_j = jnp.array(prior_maxs)
        init_loc = jnp.array(prior_means)
        init_scale = jnp.array(prior_stds) / 5.0

        # numpyro model: sample params, score with existing log_posterior
        log_posterior_fn = self.log_posterior
        dummy_data = {"data": jnp.zeros(n_dim)}

        def model():
            with numpyro.plate("params", n_dim):
                params = numpyro.sample(
                    "theta",
                    npdist.TruncatedNormal(
                        loc=init_loc, scale=jnp.array(prior_stds),
                        low=prior_mins_j, high=prior_maxs_j,
                    ),
                )
            numpyro.factor(
                "log_posterior",
                log_posterior_fn(params, data=dummy_data),
            )

        def guide():
            loc = numpyro.param(
                "loc", init_loc,
                constraint=npdist.constraints.interval(prior_mins_j, prior_maxs_j),
            )
            scale = numpyro.param(
                "scale", init_scale,
                constraint=npdist.constraints.positive,
            )
            with numpyro.plate("params", n_dim):
                numpyro.sample("theta", npdist.Normal(loc, scale))

        optimizer = numpyro.optim.Adam(self.svi_step_size)
        svi = NumpyroSVI(model, guide, optimizer, loss=Trace_ELBO())

        key, subkey = jax.random.split(key)
        svi_state = svi.init(subkey)

        logger.info(f"Starting SVI ({self.svi_num_iter} iterations)...")
        start_time = time.perf_counter()

        # JIT-compiled scan loop with NaN safety
        @jax.jit
        def run_loop(init_state):
            def body(state, _):
                new_state, loss = svi.update(state)
                safe_state = lax.cond(
                    jnp.isfinite(loss),
                    lambda: new_state,
                    lambda: state,
                )
                return safe_state, loss
            return lax.scan(body, init_state, None, length=self.svi_num_iter)

        svi_state, losses = run_loop(svi_state)

        end_time = time.perf_counter()
        logger.info(f"SVI finished in {end_time-start_time:.2f}s. Final ELBO loss: {float(losses[-1]):.2f}")

        # Extract learned parameters and draw posterior samples
        params = svi.get_params(svi_state)
        loc = params["loc"]
        scale = params["scale"]

        key, subkey = jax.random.split(key)
        draws = np.array(loc) + np.array(
            jax.random.normal(subkey, shape=(self.svi_num_samples, n_dim))
        ) * np.array(scale)

        # Build posterior_samples dict (same format as flowMC path)
        samples = draws.T  # (n_dim, n_samples)
        self.posterior_samples = self.prior.add_name(samples)

        # Add string-keyed access for convenience: map each parameter name
        # to its samples, regardless of whether the dict is keyed by Prior
        # objects (CompositePrior) or strings.
        for sub_prior in sub_priors:
            for name in sub_prior.naming:
                # Try Prior-object key first, then string key
                if sub_prior in self.posterior_samples:
                    self.posterior_samples[name] = self.posterior_samples[sub_prior]
                elif name not in self.posterior_samples:
                    # Parameter index in the flat draw array
                    idx = param_names.index(name) if name in param_names else -1
                    if idx >= 0:
                        self.posterior_samples[name] = draws[:, idx]

        # Compute log_prob for each sample (vectorized)
        log_probs = jax.vmap(
            lambda p: log_posterior_fn(p, data=dummy_data)
        )(jnp.array(draws))
        self.posterior_samples["log_prob"] = np.array(log_probs)
        self.posterior_samples["log_likelihood"] = (
            self.posterior_samples["log_prob"] - self.prior.log_prob(self.posterior_samples)
        )

        # Store SVI-specific attributes for diagnostics
        self.svi_losses = np.array(losses)
        self.svi_loc = np.array(loc)
        self.svi_scale = np.array(scale)

    
    def _get_summary_statistics(self,):
        if self.sampler_type == "svi":
            # SVI has no training/production chains — use posterior_samples directly
            samples = jnp.stack([self.posterior_samples[k] for k in self.prior.naming])
            self.production_chain = samples
            self.production_log_prob = self.posterior_samples["log_prob"]
            self.training_chain = samples
            self.training_log_prob = self.posterior_samples["log_prob"]
            self.training_local_acceptance = jnp.array([1.0])
            self.training_global_acceptance = jnp.array([1.0])
            self.production_local_acceptance = jnp.array([1.0])
            self.production_global_acceptance = jnp.array([1.0])
            self.training_loss = getattr(self, "svi_losses", jnp.array([0.0]))
            return

        resources = self.Sampler.resources

        self.training_chain = resources["positions_training"].data.reshape(-1, self.prior.n_dim).T

        self.training_log_prob = resources["log_prob_training"].data
        training_local_acceptance = resources["local_accs_training"].data
        self.training_local_acceptance = training_local_acceptance[~jnp.isneginf(training_local_acceptance)]
        training_global_acceptance = resources["global_accs_training"].data
        self.training_global_acceptance = training_global_acceptance[~jnp.isneginf(training_global_acceptance)]
        self.training_loss = resources["loss_buffer"].data

        self.production_chain = resources["positions_production"].data.reshape(-1, self.prior.n_dim).T
        self.production_log_prob = resources["log_prob_production"].data
        production_local_acceptance = resources["local_accs_production"].data
        self.production_local_acceptance = production_local_acceptance[~jnp.isneginf(production_local_acceptance)]
        production_global_acceptance = resources["global_accs_production"].data
        self.production_global_acceptance = production_global_acceptance[~jnp.isneginf(production_global_acceptance)]



    def print_summary(self, transform: bool = True):
        """
        Generate summary of the run

        """
        self._get_summary_statistics()

        print("Training summary")
        print("=" * 10)
        training_chain = self.prior.add_name(self.training_chain)
        for key, value in training_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")

        print(
            f"Log probability: {self.training_log_prob.mean():.3f} +/- {self.training_log_prob.std():.3f}"
        )

        training_local_acceptance = jnp.mean(self.training_local_acceptance, axis=0)
        print(
            f"Local acceptance: {training_local_acceptance.mean():.3f} +/- {training_local_acceptance.std():.3f}"
        )
        
        training_global_acceptance = jnp.mean(self.training_global_acceptance, axis=0)
        print(
            f"Global acceptance: {training_global_acceptance.mean():.3f} +/- {training_global_acceptance.std():.3f}"
        )

        print(
            f"Max loss: {self.training_loss.max():.3f}, Min loss: {self.training_loss.min():.3f}"
        )
        
        print("\n \n")

        print("Production summary")
        print("=" * 10)
        production_chain = self.prior.add_name(self.production_chain)
        for key, value in production_chain.items():
            print(f"{key}: {value.mean():.3f} +/- {value.std():.3f}")

        print(
            f"Log probability: {self.production_log_prob.mean():.3f} +/- {self.production_log_prob.std():.3f}"
        )

        production_local_acceptance = jnp.mean(self.production_local_acceptance, axis=0)
        print(
            f"Local acceptance: {production_local_acceptance.mean():.3f} +/- {production_local_acceptance.std():.3f}"
        )

        production_global_acceptance = jnp.mean(self.production_global_acceptance, axis=0)
        print(
            f"Global acceptance: {production_global_acceptance.mean():.3f} +/- {production_global_acceptance.std():.3f}"
        )
        print("=" * 10)
    
    def save_results(self, bestfit_params: bool =True, training_samples: bool=False):
        """
        Saves the poster samples to .npz files in ``outdir``.

        Args:
            bestfit_params (bool): Whether to print an extra file with the best fit parameters and light curves. Defaults to True.
            training_samples (bool): Whether to save the training samples from the normalizing flow training with the acceptance ratios. Defaults to False.
        """
        

        self._get_summary_statistics()
        
        if training_samples:
            # - training phase
            name = os.path.join(self.outdir, f'results_training.npz')
            logger.info(f"Saving training samples to {name}.")
    
            jnp.savez(name, log_prob=self.training_log_prob,
                            chains = self.training_chain,
                            local_accs=jnp.mean(self.training_local_acceptance, axis=0),
                            global_accs=jnp.mean(self.training_global_acceptance, axis=0), 
                            loss_vals=self.training_loss)
            
            #  - production phase
            name = os.path.join(self.outdir, f'results_production.npz')
            logger.info(f"Saving production samples to {name}")
            
            jnp.savez(name, chains=self.production_chain, 
                            log_prob=self.production_log_prob,
                            local_accs=jnp.mean(self.production_local_acceptance, axis=0),
                            global_accs=jnp.mean(self.production_global_acceptance, axis=0)
            )
        
        if bestfit_params:
            # - best fit params
            name = os.path.join(self.outdir, f'bestfit_params.pkl')
            logger.info(f"Saving best fit params to {name}.")

            lc_plotter = LightcurvePlotter(self.posterior_samples,
                                           self.likelihood)
            lc_plotter._get_best_fit_lc()
            chisq_dict = lc_plotter.get_chisquared(per_dof=True)

            with open(name, "wb") as f:
                pickle.dump({"bestfit_params": lc_plotter.best_fit_params,
                             "light_curves": {"times": lc_plotter.t_best_fit, **lc_plotter.best_fit_lc},
                             "chisq": chisq_dict}, f)


        name = os.path.join(self.outdir, f"posterior.npz")
        logger.info(f"Saving posterior samples to {name}.")
        jnp.savez(name, **self.posterior_samples)

    
    def save_hyperparameters(self):

        if self.sampler_type == "svi":
            hyperparameters_dict = {
                "svi": {
                    "num_iter": self.svi_num_iter,
                    "step_size": self.svi_step_size,
                    "num_samples": self.svi_num_samples,
                }
            }
        else:
            hyperparameters_dict = {"flowmc": self.Sampler.hyperparameters}
        
        try:
            name = os.path.join(self.outdir, "hyperparams.json")
            with open(name, 'w') as file:
                json.dump(hyperparameters_dict, file)
        except Exception as e:
            logger.error(f"Error occurred saving jim hyperparameters, are all hyperparams JSON compatible?: {e}")
            

    def plot_lightcurves(self,):
        
        """
        Plot the data and the posterior lightcurves and the best fit lightcurve more visible on top
        """      

        lc_plotter = LightcurvePlotter(self.posterior_samples,
                                       self.likelihood)

        filters = self.likelihood.filters

        ### Plot the data
        height = len(filters) * 2.5
        fig, ax = plt.subplots(nrows = len(filters), ncols = 1, figsize = (8, height))
        
        for cax, filt in zip(ax, filters):

            lc_plotter.plot_data(cax, filt, color="red")
            lc_plotter.plot_best_fit_lc(cax, filt, color="blue")
            lc_plotter.plot_sample_lc(cax, filt)
            
            # Make pretty
            cax.set_ylabel(filt)
            cax.set_xlim(left=np.maximum(self.likelihood.tmin, 1e-4), right=self.likelihood.tmax)
            cax.set_xscale("log")
            ymin = np.min(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) - 2
            ymax = np.max(np.concatenate([lc_plotter.mag_det[filt], lc_plotter.mag_nondet[filt]])) + 2
            cax.set_ylim(ymax, ymin)
        
        ax[-1].set_xlabel("$t$ in days")
        
        # Save
        fig.savefig(os.path.join(self.outdir, "lightcurves.pdf"), bbox_inches = 'tight', dpi=250)
    
    def plot_corner(self, truths: dict = None):

        fig, ax = corner_plot(self.posterior_samples,
                              self.prior.naming,
                              truths=truths)

        if fig is None:
            return

        fig.savefig(os.path.join(self.outdir, "corner.pdf"), dpi=250, bbox_inches='tight')


