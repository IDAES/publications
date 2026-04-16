"""
Functions for generating, sampling from, and plotting confidence
ellipsoids.
"""


import numpy as np
import scipy as sp
from scipy.optimize import root
from scipy.special import gamma, erf

from pyomo.contrib.pyros import EllipsoidalSet


def prob(r, n):
    """
    Evaluate the standard Gaussian (or chi-square) probability
    that the norm of a random vector in an `n`-dimensional
    Euclidean space is no greater than `r`.

    Parameters
    ----------
    r : float
        Random vector norm of interest.
    n : int
        Dimension of the Euclidean space of interest.

    Returns
    -------
    float
        Desired probability.

    Notes
    -----
    Based on the recursive formula presented in Equations
    (19) and (22) of [1]_.

    References
    ----------
    .. [1] Wang et al. "Confidence analysis of standard deviational
       ellipse and its extension into higher dimensional Euclidean
       space." PloS one 10.3 (2015): e0118537. URL:
       https://doi.org/10.1371/journal.pone.0118537
       (accessed October 24, 2023). DOI: 10.1371/journal.pone.0118537.
    """
    if n == 1:
        return erf(r / np.sqrt(2))
    elif n == 2:
        return 1 - np.exp(-r ** 2 / 2)
    else:
        return (
            prob(r, n - 2)
            - (r / np.sqrt(2)) ** (n - 2)
            * np.exp(-r ** 2 / 2) / gamma(n / 2)
        )


def mag_factor(p, n):
    """
    Determine the value `r` such that that the Gaussian/chi-square
    probability in dimension `n` of a random vector having norm
    no greater than `r` is `p`.

    Parameters
    ----------
    p : float
        Probability value.
    n : int
        Euclidean space dimensionality.

    Returns
    -------
    float
        Desired norm value.

    Notes
    -----
    Based on the procedure of Section 3.2.2. of [1]_, but we do
    not specify the derivative function.

    References
    ----------
    .. [1] Wang et al. "Confidence analysis of standard deviational
       ellipse and its extension into higher dimensional Euclidean
       space." PloS one 10.3 (2015): e0118537. URL:
       https://doi.org/10.1371/journal.pone.0118537
       (accessed October 24, 2023). DOI: 10.1371/journal.pone.0118537.
    """
    assert p >= 0 and p < 1

    def factor_func(r):
        return prob(r, n) - p

    return root(factor_func, np.sqrt(n - 1)).x[0]


def calc_conf_lvl(r, n):
    """
    Determine the value `p` such that that the Gaussian/chi-square
    probability in dimension `n` of a random vector having norm
    no greater than `r` is `p`.

    This function can be viewed as the inverse of
    ``mag_factor(p, n)``.

    Parameters
    ----------
    r : float
        Norm.
    n : int
        Euclidean space dimensionality.

    Returns
    -------
    float
        Desired probability `p`.
    """
    return prob(r, n)


def unit_sphere(n, samples=200):
    """
    Evaluate points on the n-dimensional unit sphere.
    Currently, only applied for `n=2`
    """
    angles = np.linspace(0, 2 * np.pi, samples)
    return np.array([np.cos(angles), np.sin(angles)])


def confidence_ellipsoid(nom_val, cov_mat, level, ax,
                         samples=200, plot_label=None):
    """
    Plot confidence ellipsoid.

    Parameters
    ----------
    nom_val : (N,) array-like
        Nominal, or mean value of the distribution.
    cov_mat : (N, N) array-like
        Covariance matrix.
    level : float or int
        Confidence level. Must be between 0 and 1.
    ax : matplotlib.axes.Axes
        Axes object into which to draw ellipse.
    samples : int, optional
        Number of ellipsoid points to generate for the plot.
        The default is 200.
    plot_label : str, optional
        Label for the ellipsoid plot.
        The default is `str(level)`.
    """
    assert nom_val.size == 2

    # obtain (1) eigenvalues (2) orthonormal matrix of e-vectors
    evals, orthog_mat = np.linalg.eig(cov_mat)
    root_diag = np.diag(evals ** 0.5)

    # determine scaling factor (number standard devs)
    # corresponding to confidence level
    num_std = mag_factor(level, np.size(nom_val))

    # get unit sphere points
    unit_sph = unit_sphere(2, samples=201)

    # perform Mahalanobis transformation to obtain ellipsoid
    ellipsoid_points = (
        nom_val
        + (orthog_mat @ root_diag @ orthog_mat.T @ unit_sph * num_std).T
    )

    if plot_label is None:
        plot_label = f"{100 * level:.2f}%"

    # finally, plot ellipsoid
    return ax.plot(
        ellipsoid_points[:, 0],
        ellipsoid_points[:, 1],
        label=plot_label,
    )


def get_pyros_ellipsoidal_set(mean, cov_mat, level):
    """
    Instantiate PyROS Ellipsoidal set object representing
    a confidence ellipsoid.

    Parameters
    ----------
    mean : (N,) array_like
        Center of the ellipsoid.
    cov_mat : (N, N) array_like
        Covariance (or shape) matrix of the ellipsoid.
    level : float
        Confidence level. Must be a value in (0, 1].

    Returns
    -------
    pyomo.contrib.pyros.uncertainty_sets.EllipsoidalSet
        PyROS ellipsoidal set representation.
    """
    return EllipsoidalSet(
        center=mean,
        shape_matrix=cov_mat,
        scale=mag_factor(level, np.size(mean)) ** 2,
    )


def gaussian_pdf(x, mean, cov_mat):
    """
    Evaluate Gaussian probability density function
    parameterized by a given mean and covariance matrix.
    """
    n = len(mean)
    assert cov_mat.shape == (n, n)

    # convert to arrays
    x = np.array(x)
    mean = np.array(mean)
    cov_mat = np.array(cov_mat)

    # now evaluate the probability
    denom = (2 * np.pi) ** (n / 2) * np.linalg.det(cov_mat) ** (1 / 2)
    return (1 / denom) * np.exp(
        - (1 / 2)
        * (
            (x - mean)[np.newaxis]
            @ np.linalg.inv(cov_mat)
            * (x - mean)
        ).sum(axis=-1)
    )[0]


ellipsoid_probability = gaussian_pdf


def sample_unit_sphere(n, rng, samples=100):
    """
    Uniformly random sample the `(n-1)`-dimensional surface of
    the `n`-dimensional unit ball.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    rng : numpy.random._generator.Generator
        Random number generator for sampling.
    samples : int, optional
        Number of samples to generate.

    Returns
    -------
    (`samples`, `n`) array_like
        Sampled points.

    Notes
    -----
    This method is based on Section 2.1 of [1]_, which performs the
    uniform sampling by independently sampling `n + 1` normally
    distributed variables and morphing the samples to unit vectors.

    References
    ----------
    .. [1] Voelker et al., Centre for Theoretical Neuroscience,
       Technical Report (2017). DOI: 10.13140/RG.2.2.15829.01767/1.
    """
    pts = rng.normal(size=(samples, n))
    radii = np.sqrt(np.sum(pts ** 2, axis=1)).reshape(samples, 1)

    return pts / radii


def sample_unit_ball(n, rng, samples=100):
    """
    Uniformly sample the `n`-dimensional unit ball.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    rng : numpy.random._generator.Generator
        Random number generator for sampling.
    samples : int, optional
        Number of samples to generate.

    Returns
    -------
    (`samples`, `n`) array_like
        Sampled points.

    Notes
    -----
    In accordance with Section 3.1 of [1]_, this sampling is carried
    out through the uniform random sampling of the surface of an
    `n+2`-dimensional unit sphere, then removing the last two entries
    of each sampled point.

    References
    ----------
    .. [1] Voelker et al., Centre for Theoretical Neuroscience,
       Technical Report (2017). DOI: 10.13140/RG.2.2.15829.01767/1.
    """
    return sample_unit_sphere(n + 2, rng, samples=samples)[:, :-2]


def sample_ellipsoid(mean, cov_mat, level, rng, samples=100):
    """
    Uniformly sample an `n`-dimensional ellipsoid
    with a given center, shape matrix, and size.

    Parameters
    ----------
    mean : (N,) array-like
        Center of the ellipsoid.
    cov_mat : (N, N) array-like
        Covariance matrix of the ellipsoid.
    level : float
        Confidence level for the ellipsoid.
    rng : numpy.random._generator.Generator
        Random number generator (such as Numpy's default RNG).
    samples : int, optional
        Number of points to sample from the ellipsoid.

    Returns
    -------
    ellipsoid_points : (`samples`, N) ndarray
        Sampled points.

    Notes
    -----
    The uniform sampling is performed as follows:

    - Uniformly sample the unit `n`-ball
    - Through the linear transformation of Equation 1 of [1]_,
      map the points into the ellipsoid.

    References
    ----------
    .. [1] Gammell, Jonathan D., and Timothy D. Barfoot.
       "The probability density function of a transformation-based
       hyperellipsoid sampling technique."
       arXiv preprint arXiv:1404.1347 (2014).
       URL: https://doi.org/10.48550/arXiv.1404.1347
    """
    unit_ball_pts = sample_unit_ball(mean.size, rng, samples=samples)

    # perform eigen-decomposition of the shape matrix
    # note: we use eigh here since shape matrices passed
    # to this function are presumed to be symmetric
    # and we anticipate more precise results
    evals, orthog_mat = np.linalg.eigh(cov_mat)
    root_diag = np.diag(evals ** 0.5)

    # transform sphere points to ellipsoid points
    ellipsoid_points = (
        mean
        + mag_factor(level, mean.size)
        * np.linalg.multi_dot(
            (orthog_mat, root_diag, orthog_mat.T, unit_ball_pts.T)
        ).T
    )

    return ellipsoid_points


def ellipsoid_hypervolume(mean, cov_mat, lvl):
    """
    Evaluate hypervolume of a confidence ellipsoid.

    Calculation is based on [1]_, Section A.2.

    The method makes use of the fact that the ellipsoid is
    the image of a linear transformation of the unit ball.

    References
    ----------
    .. [1] Friendly, Michael, Georges Monette, and John Fox.
       "Elliptical insights: understanding statistical methods
       through elliptical geometry." Statistical Science (2013): 1-39.
       URL: https://www.jstor.org/stable/43288410
    """
    n = mean.size
    return (
        2 / n * np.pi ** (n / 2) / sp.special.gamma(n / 2)
        * mag_factor(lvl, n) ** n
        * np.sqrt(np.linalg.det(cov_mat))
    )


def sample_ellipsoid_boundary(mean, cov_mat, level, rng, samples=100):
    """
    Uniformly sample the `(n - 1)`-dimensional
    boundary of an `n`-dimensional ellipsoid
    with a given center, shape matrix, and size.

    Parameters
    ----------
    mean : (N,) array-like
        Center of the ellipsoid.
    cov_mat : (N, N) array-like
        Covariance matrix of the ellipsoid.
    level : float
        Confidence level for the ellipsoid.
    rng : numpy.random._generator.Generator
        Random number generator (such as Numpy's default RNG).
    samples : int, optional
        Number of points to sample from the ellipsoid.

    Returns
    -------
    ellipsoid_points : (`samples`, N) ndarray
        Sampled points.
    """
    mean = np.array(mean)
    cov_mat = np.array(cov_mat)

    unit_sphere_pts = sample_unit_sphere(mean.size, rng, samples=samples)

    # perform eigen-decomposition of the shape matrix
    # note: we use eigh here since shape matrices passed
    # to this function are presumed to be symmetric
    # and we anticipate more precise results
    evals, orthog_mat = np.linalg.eigh(cov_mat)
    root_diag = np.diag(evals ** 0.5)

    # transform sphere points to ellipsoid points
    boundary_points = (
        mean
        + mag_factor(level, mean.size)
        * np.linalg.multi_dot(
            (orthog_mat, root_diag, orthog_mat.T, unit_sphere_pts.T)
        ).T
    )

    return boundary_points


def plot_confidence_ellipsoid(
    mean,
    cov_mat,
    level,
    ax,
    samples=200,
    **kwargs,
):
    """
    Plot boundary of a 2D confidence ellipsoid.

    Parameters
    ----------
    mean : (2,) array_like
        Center of the ellipsoid.
    cov_mat : (2, 2) array_like
        Covariance matrix.
    level : float or int
        Confidence level. Must be between 0 and 1.
    ax : matplotlib.axes.Axes
        Axes object into which to draw ellipse.
    samples : int, optional
        Number of ellipsoid points to generate for the plot.
        The default is 200.
    **kwargs : dict, optional
        Keyword arguments to ``matplotlib.pyplot.plot``.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Output of ``matplotlib.plot(...)`` containing plotted data.
    """
    if mean.size != 2:
        raise ValueError("Only 2D ellipsoids supported.")

    # obtain (1) eigenvalues (2) orthonormal matrix of e-vectors
    evals, orthog_mat = np.linalg.eigh(cov_mat)
    root_diag = np.diag(evals**0.5)

    # determine scaling factor (number standard devs)
    # corresponding to confidence level
    num_std = mag_factor(level, np.size(mean))

    angles = np.linspace(0, 2 * np.pi, 2000)
    unit_sphere_surface_pts = np.array([np.cos(angles), np.sin(angles)])

    # perform Mahalanobis transformation to obtain ellipsoid points
    ellipsoid_points = (
        mean
        + (
            orthog_mat
            @ root_diag
            @ orthog_mat.T
            @ unit_sphere_surface_pts
            * num_std
        ).T
    )

    return ax.plot(ellipsoid_points[:, 0], ellipsoid_points[:, 1], **kwargs)


def evaluate_projection_params(mean, cov_mat, conf_lvl, on_axes):
    """
    Evaluate parameters (mean, covariance matrix, confidence level)
    of the projection of an ellipsoid on a coordinate plane.

    Parameters
    ----------
    mean : (N,) array_like
        Center of the ellipsoid.
    cov_mat : (N, N) array_like
        Covariance (shape) matrix of the ellipsoid.
    conf_lvl : float
        Confidence level.
    on_axes : (M,) array_like of int
        Indexes of the dimensions/coordinate axes on which
        to project the ellipsoid.

    Returns
    -------
    new_mean : (M,) numpy.ndarray
        Center of the projection.
    new_cov_mat : (M, M) numpy.ndarray
        Covariance, or shape matrix, of the projection.
    new_conf_lvl : float
        Confidence level of the projection.

    Notes
    -----
    This method is based on Section 13 (pp. 30--31) of [1]_.

    References
    ----------
    .. [1] Pope, Stephen B. "Algorithms for ellipsoids."
       Cornell University Report No. FDA (2008): 08-01.
       URL: https://tcg.mae.cornell.edu/pubs/Pope_FDA_08.pdf
       accessed 2024-01-31.
    """
    mean = np.array(mean)
    cov_mat = np.array(cov_mat)
    on_axes = np.array(on_axes)

    n = mean.size
    m = len(on_axes)

    T = np.zeros((n, m))
    for idx, onax in enumerate(on_axes):
        T[onax, idx] = 1

    scale_factor = mag_factor(conf_lvl, n)

    evals, evecs = np.linalg.eigh(cov_mat)
    sqrt_cov_mat = evecs * np.sqrt(evals) @ np.linalg.inv(evecs)
    LnegT = scale_factor * sqrt_cov_mat
    U, S, V = np.linalg.svd(T.T @ LnegT)

    B = np.linalg.inv(U @ np.diag(S)).T
    LTilde = np.linalg.cholesky(B @ B.T)

    new_cov_mat = np.linalg.inv(LTilde.T) @ np.linalg.inv(LTilde.T).T
    new_conf_lvl = calc_conf_lvl(1, m)
    new_mean = mean[on_axes]

    return new_mean, new_cov_mat, new_conf_lvl


def get_conf_lvl_plot_label(
        conf_lvl,
        n=None,
        decimals=1,
        as_mag_factor=True,
        conf_lvl_suffix="\\% CI",
        mag_factor_suffix="$\\boldsymbol{\\sigma}$",
        ):
    """
    Get plotting label for confidence level.
    """
    if as_mag_factor:
        suffix = mag_factor_suffix
        val = mag_factor(conf_lvl / 100, n)
    else:
        suffix = conf_lvl_suffix
        val = conf_lvl
    return f"{val:.{decimals}f}{suffix}"


def plot_confidence_ellipsoid_projection(
        mean,
        cov_mat,
        level,
        ax,
        plane_idxs,
        samples=200,
        **kwargs,
        ):
    """
    Plot projection of confidence ellipsoid (including uniform
    random samples) onto specified 2-D coordinate plane.

    Parameters
    ----------
    mean : (N,) array_like
        Center of the ellipsoid.
    cov_mat : (N, N) array_like
        Covariance (shape) matrix of the ellipsoid.
    level : float
        Confidence level.
    ax : matplotlib.axes.Axes
        Axes into which to draw the projection.
    plane_idxs : (2,) array_like of int
        Indexes of the dimensions/coordinate axes on which
        to project the ellipsoid.
    samples : int, optional
        Number of uniform random samples to plot.
    **kwargs
        Additional keyword arguments passed to
        `plot_confidence_ellipsoid`.
    """
    assert len(plane_idxs) == 2
    mean = np.array(mean)

    newmean, newcovmat, newconflvl = evaluate_projection_params(
        mean, cov_mat, level, plane_idxs
    )
    sampled_pts = sample_ellipsoid(
        mean=mean,
        cov_mat=cov_mat,
        level=level,
        rng=np.random.default_rng(123456),
        samples=samples,
    )
    ax.scatter(
        sampled_pts[:, plane_idxs[0]],
        sampled_pts[:, plane_idxs[1]],
        color="red",
    )

    return plot_confidence_ellipsoid(
        mean=newmean,
        cov_mat=newcovmat,
        level=newconflvl,
        ax=ax,
        samples=0,
        **kwargs,
    )
