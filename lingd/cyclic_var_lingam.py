import numpy as np

from lingam import VARLiNGAM
from lingam import VARBootstrapResult
from sklearn.utils import check_array
from sklearn.utils import check_scalar
from sklearn.utils import check_random_state
from sklearn.utils import resample

from .lingd import LiNGD


class CyclicVARLiNGAM(VARLiNGAM):
    """ Implementation of VAR-LiNGAM algorithm for cyclic structured data """
    
    def __init__(self, lags=1, criterion="bic", ar_coefs=None, k=5, random_state=None):
        """Construct a CyclicVARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
            Shape must be (``lags``, n_features, n_features).
        k : int, option (default=5)
            Number of candidate causal graphs to estimate.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        super().__init__(
            lags=lags,
            criterion=criterion,
            ar_coefs=ar_coefs,
            random_state=random_state
        )
        
        self._k = k
        self._random_state = random_state
    
    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        returns
        -------
        self : object
            Returns the instance itself.
        """
        self._adjacency_matrices = None
        self._adjacency_matrices_list = []

        # check args
        X = check_array(X)

        k = check_scalar(self._k, "k", int, min_val=1)

        random_state = check_random_state(self._random_state)

        # XXX: using updated ar_coefs like VARLiNGAM causes bugs
        #self._ar_coefs = None
        M_taus = self._ar_coefs

        if M_taus is None:
            M_taus, lags, residuals = self._estimate_var_coefs(X)
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)

        model = LiNGD(k=k)
        model.fit(residuals)

        B_taus_list = []
        for adj in model.adjacency_matrices_:
            B_taus = self._calc_b(X, adj, M_taus)
            B_taus_list.append(B_taus)
            
        if np.all(~model.is_stables_):
            best_index = np.argmin(model.costs_).ravel()[0]
        else:
            best_index = np.argmin(model.costs_[model.is_stables_]).ravel()[0]
        B_taus_best = B_taus_list[best_index]
         
        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals
        self._adjacency_matrices = B_taus_best
        
        self._adjacency_matrices_list = np.array(B_taus_list)
        self._is_stables = model.is_stables_
        self._costs = model.costs_

        return self

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : TimeseriesBootstrapResult
            Returns the result of bootstrapping.
        """
        X = check_array(X)

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # store initial settings
        ar_coefs = self._ar_coefs
        lags = self._lags

        criterion = self._criterion
        self._criterion = None

        self.fit(X)

        fitted_ar_coefs = self._ar_coefs

        # no total effects are calculated because there is no causal order
        total_effects = np.zeros(
            [n_sampling, n_features, n_features * (1 + self._lags)]
        )

        adjacency_matrices = []
        for i in range(n_sampling):
            sampled_residuals = resample(self._residuals, n_samples=n_samples)

            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < lags:
                    resampled_X[j, :] = sampled_residuals[j]
                    continue

                ar = np.zeros((1, n_features))
                for t, M in enumerate(fitted_ar_coefs):
                    ar += np.dot(M, resampled_X[j - t - 1, :].T).T

                resampled_X[j, :] = ar + sampled_residuals[j]

            # restore initial settings
            self._ar_coefs = ar_coefs
            self._lags = lags

            self.fit(resampled_X)
            am = np.concatenate([*self._adjacency_matrices], axis=1)
            adjacency_matrices.append(am)

        self._criterion = criterion

        return VARBootstrapResult(adjacency_matrices, total_effects)
    
    @property
    def adjacency_matrices_list_(self):
        return self._adjacency_matrices_list

    @property
    def is_stables_(self):
        return self._is_stables

    @property
    def costs_(self):
        return self._costs

    @property
    def causal_order_(self):
        raise NotImplementedError
