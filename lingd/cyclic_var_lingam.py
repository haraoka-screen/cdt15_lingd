import numpy as np

from lingam import VARLiNGAM
from sklearn.utils import check_array
from sklearn.utils import check_scalar
from sklearn.utils import check_random_state

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

        # XXX: using updated ar_coefs causes bugs
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
