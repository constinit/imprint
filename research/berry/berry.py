import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy.special import logit

import inla
import util

"""
The Berry model!

y_i ~ Binomial(theta_i, N_i)
theta_i ~ N(mu, sigma2)
mu ~ N(mu_0, S2)
sigma2 ~ InvGamma(a, b)

mu_0 = -1.34
S2 = 100
a = 0.0005
b = 0.000005
"""

class Berry:
    def __init__(self, sigma2_n_quad, sigma2_bounds):
        """
        sigma2_n_quad: int, the number of quadrature points to use integrating over the sigma2 hyperparameter
        sigma2_bounds: a tuple (a, b) specifying the integration limits in the sigma2 dimension
        """
        self.n_stages = 6
        self.n_arms = 4


        # Final evaluation criterion:
        # Accept the alternative hypo if Pr(p[i] > p0|data) > pfinal_thresh[i]
        # Or in terms of theta: Pr(theta[i] > p0_theta|data) > pfinal_thresh[i]
        self.p0 = np.full(4, 0.1)  # rate of response below this is the null hypothesis
        self.p0_theta = scipy.special.logit(self.p0)
        self.pfinal_thresh = np.full(4, 0.85)


        # Interim success criterion:
        # For some of Berry's calculations (e.g. the interim analysis success
        # criterion in Figure 1/2, the midpoint of p0 and p1 is used.)
        # Pr(theta[i] > pmid_theta|data) > pmid_accept
        # or concretely: Pr(theta[i] > 0.2|data) > 0.9
        self.p1 = np.full(4, 0.3)
        self.pmid = (self.p0 + self.p1) / 2
        self.pmid_theta = scipy.special.logit(self.pmid)
        self.pmid_accept = 0.9

        # Early failure criterion:
        # Pr(theta[i] > pmid_theta|data) < pmid_fail
        self.pmid_fail = 0.05

        # Specify the stopping/success criteria.
        self.suc_thresh = np.empty((self.n_stages, self.n_arms))
        # early stopping condition.
        self.suc_thresh[:5] = self.pmid_theta
        # final success criterion
        self.suc_thresh[5] = self.p0_theta

        self.model = inla.Model(
            self.berry_prior,
            self.log_joint,
            self.log_joint_xonly,
            self.gradx_log_joint,
            self.hessx_log_joint,
        )

        # TODO: move to using the MVN version that has no mu integration.
        self.mu_rule = util.gauss_rule(201, -5, 3)
        self.sigma2_rule = util.log_gauss_rule(sigma2_n_quad, *sigma2_bounds)

    def berry_prior(self, hyper):
        mu = hyper[..., 0]
        # mu prior: N(-2.197, 100)
        mu_prior = scipy.stats.norm.logpdf(mu, self.p0_theta[0], 100)

        # sigma prior: InvGamma(0.0005, 0.000005)
        sigma2 = hyper[..., 1]
        alpha = 0.0005
        beta = 0.000005
        sigma2_prior = scipy.stats.invgamma.logpdf(sigma2, alpha, scale=beta)
        return mu_prior + sigma2_prior


    def log_gaussian_x_diag(x, hyper, include_det):
        """
        Gaussian likelihood for the latent variables x for the situation in which the
        precision matrix for those latent variables is diagonal.

        what we are computing is: -x^T Q x / 2 + log(determinant(Q)) / 2

        Sometimes, we may want to leave out the determinant term because it does not
        depend on x. This is useful for computational efficiency when we are
        optimizing an objective function with respect to x.
        """
        n_rows = x.shape[-1]
        mu = hyper[..., 0]
        sigma2 = hyper[..., 1]
        Qv = 1.0 / sigma2
        quadratic = -0.5 * ((x - mu[..., None]) ** 2) * Qv[..., None]
        out = np.sum(quadratic, axis=-1)
        if include_det:
            # determinant of diagonal matrix = prod(diagonal)
            # log(prod(diagonal)) = sum(log(diagonal))
            out += np.log(Qv) * n_rows / 2
        return out

    # def log_gaussian_x_mvn(x, hyper, include_det):
    #     """

    #     Parameters
    #     ----------
    #     x
    #         The 
    #     hyper
    #         The hyperparameter array
    #     include_det
    #         Should we include the determinant term? 
    #     """
    #     mu0 = 
    #     sigma2 = hyper[..., 1]


    def log_binomial(x, data):
        y = data[..., 0]
        n = data[..., 1]
        return np.sum(x * y - n * np.log(np.exp(x) + 1), axis=-1)


    def log_joint(model, x, data, hyper):
        # There are three terms here:
        # 1) The terms from the Gaussian distribution of the latent variables
        #    (indepdent of the data):
        # 2) The term from the response variable (in this case, binomial)
        # 3) The prior on the hyperparameters
        return (
            log_gaussian_x_diag(x, hyper, True)
            + log_binomial(x, data)
            + model.log_prior(hyper)
        )


    def log_joint_xonly(x, data, hyper):
        # See log_joint, we drop the parts not dependent on x.
        term1 = log_gaussian_x_diag(x, hyper, False)
        term2 = log_binomial(x, data)
        return term1 + term2


    def gradx_log_joint(x, data, hyper):
        y = data[..., 0]
        n = data[..., 1]
        mu = hyper[..., 0]
        Qv = 1.0 / hyper[..., 1]
        term1 = -Qv[..., None] * (x - mu[..., None])
        term2 = y - (n * np.exp(x) / (np.exp(x) + 1))
        return term1 + term2


    def hessx_log_joint(x, data, hyper):
        n = data[..., 1]
        Qv = 1.0 / hyper[..., 1]
        term1 = -n * np.exp(x) / ((np.exp(x) + 1) ** 2)
        term2 = -Qv[..., None]
        return term1 + term2


def figure1_plot(b, title, data, stats):
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title)
    outergs = fig.add_gridspec(2, 3, hspace=0.3)
    for i in range(data.shape[0]):
        innergs = outergs[i].subgridspec(
            2, 1, wspace=0, hspace=0, height_ratios=[0.7, 0.3]
        )
        figure1_subplot(innergs[0], innergs[1], i, b, data, stats)

    plt.show()


def figure1_subplot(gridspec0, gridspec1, i, b, data, stats):
    plt.subplot(gridspec0)
    # expit(mu_post) is the posterior estimate of the mean probability.
    p_post = scipy.special.expit(stats["mu_appx"])

    # two sigma confidence intervals transformed from logit to probability space.
    cilow = scipy.special.expit(stats["mu_appx"] - 2 * stats["sigma_appx"])
    cihigh = scipy.special.expit(stats["mu_appx"] + 2 * stats["sigma_appx"])

    y = data[:, :, 0]
    n = data[:, :, 1]

    # The simple ratio of success to samples. Binomial "p".
    raw_ratio = y / n

    plt.plot(np.arange(4), raw_ratio[i], "kx")
    plt.plot(np.arange(4), p_post[i], "ko", mfc="none")
    plt.plot(np.arange(4), stats["exceedance"][i], "k ", marker=(8, 2, 0))

    plt.vlines(np.arange(4), cilow[i], cihigh[i], color="k", linewidth=1.0)

    if i < 5:
        plt.title(f"Interim Analysis {i+1}")
        plt.hlines([b.pmid_fail, b.pmid_accept], -1, 4, colors=["k"], linestyles=["--"])
        plt.text(-0.1, 0.91, "Early Success", fontsize=7)
        plt.text(2.4, 0.06, "Early Futility", fontsize=7)
    else:
        plt.title("Final Analysis")
        plt.hlines([b.pfinal_thresh[0]], -1, 4, colors=["k"], linestyles=["--"])
        plt.text(-0.1, 0.86, "Final Success", fontsize=7)

    plt.xlim([-0.3, 3.3])
    plt.ylim([0.0, 1.05])
    plt.yticks(np.linspace(0.0, 1.0, 6))
    plt.xlabel("Group")
    plt.ylabel("Probability")

    plt.subplot(gridspec1)
    plt.bar(
        [0, 1, 2, 3],
        n[i],
        tick_label=[str(i) for i in range(4)],
        color=(0.6, 0.6, 0.6, 1.0),
        edgecolor="k",
        zorder=0,
    )
    plt.bar(
        [0, 1, 2, 3],
        y[i],
        color=(0.6, 0.6, 0.6, 1.0),
        hatch="////",
        edgecolor="w",
        lw=1.0,
        zorder=1,
    )
    #         # draw hatch
    # ax1.bar(range(1, 5), range(1, 5), color='none', edgecolor='red', hatch="/", lw=1., zorder = 0)
    # # draw edge
    plt.bar([0, 1, 2, 3], y[i], color="none", edgecolor="k", zorder=2)
    ticks = np.arange(0, 36, 5)
    plt.yticks(ticks, [str(i) if i % 10 == 0 else "" for i in ticks])
    plt.xticks(np.arange(4), ["1", "2", "3", "4"])
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel("Group")
    plt.ylabel("N")