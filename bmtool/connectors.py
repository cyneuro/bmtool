from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize_scalar
from functools import partial
import time
import types

rng = np.random.default_rng()

##############################################################################
############################## CONNECT CELLS #################################

# Utility Functions
def decision(prob, size=None):
    """
    Make single random decision based on input probability.
    prob: scalar input
    Return bool array if size specified, otherwise scalar
    """
    return rng.random(size) < prob


def decisions(prob):
    """
    Make multiple random decisions based on input probabilities.
    prob: iterable
    Return bool array of the same shape
    """
    prob = np.asarray(prob)
    return rng.random(prob.shape) < prob


def euclid_dist(p1, p2):
    """
    Euclidean distance between two points
    p1, p2: Coordinates in numpy array
    """
    dvec = np.asarray(p1) - np.asarray(p2)
    return (dvec @ dvec) ** .5


def spherical_dist(node1, node2):
    """Spherical distance between two input nodes"""
    return euclid_dist(node1['positions'], node2['positions']).item()


def cylindrical_dist_z(node1, node2):
    """Cylindircal distance between two input nodes (ignoring z-axis)"""
    return euclid_dist(node1['positions'][:2], node2['positions'][:2]).item()


# Probability Classes
class ProbabilityFunction(ABC):
    """Abstract base class for connection probability function"""

    @abstractmethod
    def probability(self, *arg, **kwargs):
        """Allow numpy array input and return probability in numpy array"""
        return NotImplemented

    @abstractmethod
    def __call__(self, *arg, **kwargs):
        """Return probability within [0, 1] for single input"""
        return NotImplemented

    @abstractmethod
    def decisions(self, *arg, **kwargs):
        """Return bool array of decisions according probability"""
        return NotImplemented


class DistantDependentProbability(ProbabilityFunction):
    """Base class for distance dependent probability"""

    def __init__(self, min_dist=0., max_dist=np.inf):
        assert(min_dist >= 0 and min_dist < max_dist)
        self.min_dist, self.max_dist = min_dist, max_dist

    def __call__(self, dist, *arg, **kwargs):
        """Return probability for single distance input"""
        if dist >= self.min_dist and dist <= self.max_dist:
            return self.probability(dist)
        else:
            return 0.

    def decisions(self, dist):
        """Return bool array of decisions given distance array"""
        dist = np.asarray(dist)
        dec = np.zeros(dist.shape, dtype=bool)
        mask = (dist >= self.min_dist) & (dist <= self.max_dist)
        dist = dist[mask]
        prob = np.empty(dist.shape)
        prob[:] = self.probability(dist)
        dec[mask] = decisions(prob)
        return dec


class UniformInRange(DistantDependentProbability):
    """Constant probability within a distance range"""

    def __init__(self, p=0., min_dist=0., max_dist=np.inf):
        super().__init__(min_dist=min_dist, max_dist=max_dist)
        self.p = np.array(p)
        assert(self.p.size == 1)
        assert(p >= 0. and p <= 1.)

    def probability(self, dist):
        return self.p


NORM_COEF = (2 * np.pi) ** (-.5)  # coefficient of standard normal PDF

def gaussian(x, mean=0., stdev=1., pmax=NORM_COEF):
    """Gaussian function. Default is the PDF of standard normal distribution"""
    x = (x - mean) / stdev
    return pmax * np.exp(- x * x / 2)


class GaussianDropoff(DistantDependentProbability):
    """
    Object for calculating connection probability following a Gaussian function
    of the distance between cells, using spherical or cylindrical distance.

    Parameters:
        mean, stdev: Parameters for the Gaussian function.
        min_dist, max_dist: Distance range for any possible connection,
            the support of the Gaussian function.
        pmax: The maximum value of the Gaussian function at its mean parameter.
        ptotal: Overall probability within distance range. If specified, ignore
            input pmax, and calculate pmax. See calc_pmax_from_ptotal() method.
        ptotal_dist_range: Distance range for calculating pmax when ptotal is
        specified. If not specified, set to range (min_dist, max_dist).
        dist_type: 'spherical' or 'cylindrical' for distance metric.
            Used when ptotal is specified.

    Returns:
        A callable object. When called with a single distance input,
        returns the probability value.

    TODO: Accept convergence and cell density information for calculating pmax.
    """

    def __init__(self, mean=0., stdev=1., min_dist=0., max_dist=np.inf,
                 pmax=1, ptotal=None, ptotal_dist_range=None,
                 dist_type='spherical'):
        super().__init__(min_dist=min_dist, max_dist=max_dist)
        self.mean, self.stdev = mean, stdev
        self.ptotal = ptotal
        self.ptotal_dist_range = (min_dist, max_dist) \
            if ptotal_dist_range is None else ptotal_dist_range
        self.dist_type = dist_type if dist_type in \
            ['cylindrical'] else 'spherical'
        self.pmax = pmax if ptotal is None else self.calc_pmax_from_ptotal()
        self.set_probability_func()

    def calc_pmax_from_ptotal(self):
        """
        Calculate the pmax value such that the expected overall connection
        probability to all possible targets within the distance range [r1, r2]=
        `ptotal_dist_range` equals ptotal, assuming homogeneous cell density.
        That is, integral_r1^r2 {g(r)p(r)dr} = ptotal, where g is the Gaussian
        function with pmax, p(r) is the cell density per unit distance at r
        normalized by total cell number within the distance range.
        For cylindrical distance, p(r) = 2 * r / (r2^2 - r1^2)
        For spherical distance, p(r) = 3 * r^2 / (r2^3 - r1^3)
        The solution has a closed form except that te error function erf is in
        the expression, but only when resulting pmax <= 1.

        Caveat: When the calculated pmax > 1, the actual overall probability
        will be lower than expected and all cells within certain distance will
        be always connected. This usually happens when the distance range is
        set too wide. Because a large population will be included for
        evaluating ptotal, and there will be a significant drop in the Gaussian
        function as distance gets further. So, a large pmax will be required to
        achieve the desired ptotal.
        """
        mu, sig = self.mean, self.stdev
        r1, r2 = self.ptotal_dist_range[:2]
        x1, x2 = (r1 - mu) / sig, (r2 - mu) / sig  # normalized distance
        if self.dist_type == 'cylindrical':
            dr = r2 ** 2 - r1 ** 2
            def F(x):
                f1 = sig * mu / NORM_COEF * erf(x / 2**.5)
                f2 = -2 * sig * sig * gaussian(x, pmax=1.)
                return f1 + f2
        else:
            dr = r2 ** 3 - r1 ** 3
            def F(x):
                f1 = 1.5 * sig * (sig**2 + mu**2) / NORM_COEF * erf(x / 2**.5)
                f2 = -3 * sig * sig * (2 * mu + sig * x) * gaussian(x, pmax=1.)
                return f1 + f2
        return self.ptotal * dr / (F(x2) - F(x1))

    def probability(self):
        pass  # to be set up in set_probability_func()

    def set_probability_func(self):
        """Set up function for calculating probability"""
        keys = ['mean', 'stdev', 'pmax']
        kwargs = {key: getattr(self, key) for key in keys}
        probability = partial(gaussian, **kwargs)

        # Verify maximum probability
        # (is not self.pmax if self.mean outside distance range)
        bounds = (self.min_dist, min(self.max_dist, 1e9))
        pmax = self.pmax if self.mean >= bounds[0] and self.mean <= bounds[1] \
            else probability(np.asarray(bounds)).max()
        if pmax > 1:
            d = minimize_scalar(lambda x: (probability(x) - 1)**2,
                                method='bounded', bounds=bounds).x
            warn = ("\nWarning: Maximum probability=%.3f is greater than 1. "
                    "Probability crosses 1 at distance %.3g.\n") % (pmax, d)
            if self.ptotal is not None:
                warn += " ptotal may not be reached."
            print(warn)
            self.probability = lambda dist: np.fmin(probability(dist), 1.)
        else:
            self.probability = probability


class NormalizedReciprocalRate(ProbabilityFunction):
    """Reciprocal connection probability given normalized reciprocal rate.
    Normalized reciprocal rate is defined as the ratio between the reciprocal
    connection probability and the connection probability for a randomly
    connected network where the two unidirectional connections between any pair
    of neurons are independent. NRR = pr / (p0 * p1)
    
    Parameters:
        NRR: a constant or distance dependent function for normalized reciprocal
            rate. When being a function, it should be accept vectorized input.
    Returns:
        A callable object that returns the probability value.
    """

    def __init__(self, NRR=1.):
        self.NRR = NRR if callable(NRR) else lambda *x: NRR

    def probability(self, dist, p0, p1):
        """Allow numpy array input and return probability in numpy array"""
        return p0 * p1 * self.NRR(dist)

    def __call__(self, dist, p0, p1, *arg, **kwargs):
        """Return probability for single distance input"""
        return self.probability(dist, p0, p1)

    def decisions(self, dist, p0, p1, cond=None):
        """Return bool array of decisions
        dist: distance (scalar or array). Will be ignored if NRR is constant.
        p0, p1: forward and backward probability (scalar or array)
        cond: A tuple (direction, array of outcomes) representing the condition.
            Conditional probability will be returned if specified. The condition
            event is determined by connection direction (0 for forward, or 1 for
            backward) and outcomes (bool array of whether connection exists).
        """
        dist, p0, p1 = map(np.asarray, (dist, p0, p1))
        pr = np.empty(dist.shape)
        pr[:] = self.probability(dist, p0, p1)
        pr = np.clip(pr, a_min=np.fmax(p0 + p1 - 1., 0.), a_max=np.fmin(p0, p1))
        if cond is not None:
            mask = np.asarray(cond[1])
            pr[mask] /= p1 if cond[0] else p0
            pr[~mask] = 0.
        return decisions(pr)


# Connector Classes
class AbstractConnector(ABC):
    """Abstract base class for connectors"""
    @abstractmethod
    def setup_nodes(self, source=None, target=None):
        """After network nodes are added to the BMTK network. Pass in the
        Nodepool objects of source and target nodes using this method.
        Must run this before building connections."""
        return NotImplemented

    @abstractmethod
    def edge_params(self, **kwargs):
        """Create the arguments for BMTK add_edges() method including the
        `connection_rule` method."""
        return NotImplemented

    @staticmethod
    def is_same_pop(source, target, quick=True):
        """Whether two NodePool objects direct to the same population"""
        if quick:
            # Quick check (compare filter conditions)
            same = (source.network_name == target.network_name and
                    source._NodePool__properties ==
                    target._NodePool__properties)
        else:
            # Strict check (compare all nodes)
            same = (source.network_name == target.network_name and
                    len(source) == len(target) and
                    all([s.node_id == t.node_id
                         for s, t in zip(source, target)]))
        return same

    @staticmethod
    def constant_function(val):
        """Convert a constant to a constant function"""
        def constant(*arg):
            return val
        return constant


# Helper class
class Timer(object):
    def __init__(self, unit='sec'):
        if unit == 'ms':
            self.scale = 1e3
        elif unit == 'us':
            self.scale = 1e6
        elif unit == 'min':
            self.scale = 1 / 60
        else:
            self.scale = 1
            unit = 'sec'
        self.unit = unit
        self.start()

    def start(self):
        self._start = time.perf_counter()

    def end(self):
        return (time.perf_counter() - self._start) * self.scale

    def report(self, msg='Run time'):
        print((msg + ": %.3f " + self.unit) % self.end())


def pr_2_rho(p0, p1, pr):
    """Calculate correlation coefficient rho given reciprocal probability pr"""
    for p in (p0, p1):
        assert(p > 0 and p < 1)
    assert(pr >= 0 and pr <= p0 and pr <= p1 and pr >= p0 + p1 - 1)
    return (pr - p0 * p1) / (p0 * (1 - p0) * p1 * (1 - p1)) ** .5


def rho_2_pr(p0, p1, rho):
    """Calculate reciprocal probability pr given correlation coefficient rho"""
    for p in (p0, p1):
        assert(p > 0 and p < 1)
    pr = p0 * p1 + rho * (p0 * (1 - p0) * p1 * (1 - p1)) ** .5
    if not (pr >= 0 and pr <= p0 and pr <= p1 and pr >= p0 + p1 - 1):
        pr0, pr = pr, np.max((0., p0 + p1 - 1, np.min((p0, p1, pr))))
        rho0, rho = rho, (pr - p0 * p1) / (p0 * (1 - p0) * p1 * (1 - p1)) ** .5
        print('rho changed from %.3f to %.3f; pr changed from %.3f to %.3f'
              % (rho0, rho, pr0, pr))
    return pr


class ReciprocalConnector(AbstractConnector):
    """
    Object for buiilding connections in bmtk network model with reciprocal
    probability within a single population (or between two populations).

    Algorithm:
        Create random connection for every pair of cells independently,
        following a bivariate Bernoulli distribution. Each variable is 0 or 1,
        whether a connection exists in a forward or backward direction. There
        are four possible outcomes for each pair, no connection, unidirectional
        connection in two ways, and reciprocal connection. The probability of
        each outcome forms a contingency table.
            b a c k w a r d
        f   ---------------
        o  |   |  0  |  1  |  The total forward connection probability is
        r  |---|-----|-----|  p0 = p10 + p11
        w  | 0 | p00 | p01 |  The total backward connection probability is
        a  |---|-----|-----|  p1 = p01 + p11
        r  | 1 | p10 | p11 |  The reciprocal connection probability is
        d   ---------------   pr = p11
        The distribution can be characterized by three parameters, p0, p1, pr.
        pr = p0 * p1 when two directions are independent. The correlation
        coefficient rho between the two has a relation with pr as follow.
        rho = (pr-p0*p1) / (p0*(1-p0)*p1*(1-p1))^(1/2)
        Generating random outcome consists of two steps. First draw random
        outcome for forward connection with probability p0. Then draw backward
        outcome following a conditional probability given the forward outcome,
        represented by p0, p1, and either pr or rho.

    Use with BMTK:
        1. Create this object with parameters.

            connector = ReciprocalConnector(**parameters)

        2. After network nodes are added to the BMTK network. Pass in the
        Nodepool objects of source and target nodes using setup_nodes() method.

            source = net.nodes(**source_filter)
            target = net.nodes(**target_filter)
            connector.setup_nodes(source, target)

        3. Use edge_params() method to get the arguments for BMTK add_edges()
        method including the `connection_rule` method.

            net.add_edges(**connector.edge_params(),
                          **other_source_to_target_edge_properties)

        If the source and target are two different populations, do this again
        for the backward connections (from target to source population).

            net.add_edges(**connector.edge_params(),
                          **other_target_to_source_edge_properties)

        4. When executing net.build(), BMTK uses built-in `one_to_all` iterator
        that calls the make_forward_connection() method to build connections
        from source to target. If the two are different populations,
        `all_to_one` iterator that calls the make_backward_connection() method
        is then used to build connections from target to source.
        During the initial iteration when make_forward_connection() is called,
        the algorithm is run to generate a connection matrix for both forward
        and backward connections. In the iterations afterward, it's only
        assigning the generated connections in BMTK.

    Parameters:
        p0, p1: Probability of forward and backward connection. It can be a
            constant or a deterministic function whose value must be within
            range [0, 1], otherwise incorrect value may occur in the algorithm.
            When p0, p1 are constant, the connection is homogenous.
        symmetric_p1: Whether p0 and p1 are identical. When the probabilities
            are equal for forward and backward connections, set this to True,
            Argument p1 will be ignored. This is forced to be True when the
            population is recurrent, i.e., the source and target are the same.
            This is forced to be False if symmetric_p1_arg is False.
        p0_arg, p1_arg: Input argument(s) for p0 and p1 function, e.g.,
            p0(p0_arg). It can be a constant or a deterministic function whose
            input arguments are two node objects in BMTK, e.g.,
            p0_arg(src_node,trg_node), p1_arg(trg_node,src_node). The latter
            has reversed order since it's for backward connection. They are
            usually distance between two nodes which is used for distance
            dependent connection probability, where the order does not matter.
            When p0 and p1 does not need inputs arguments, set p0_arg and
            p1_arg to None as so by default. Functions p0 and p1 need to accept
            one unused positional argument as placeholder, e.g., p0(*args), so
            it does not raise an error when p0(None) is called.
        symmetric_p1_arg: Whether p0_arg and p1_arg are identical. If this is
            set to True, argument p1_arg will be ignored. This is forced to be
            True when the population is recurrent.
        pr, pr_arg: Probability of reciprocal connection and its first input
            argument when it is a function, similar to p0, p0_arg, p1, p1_arg.
            It can be a function when it has an explicit relation with some node
            properties such as distance. A function pr requires two additional
            positional arguments p0 and p1 even if they are not used, i.e.,
            pr(pr_arg, p0, p1), just in case pr is dependent on p0 and p1, e.g.,
            when normalized reciprocal rate NRR = pr/(p0*p1) is given.
            When pr_arg is a string, the same value as p1_arg will be used for
            pr_arg if the string contains '1', e.g., '1', 'p1'. Otherwise, e.g.,
            '', '0', 'p0', p0_arg will be used for pr_arg. Specifying this can
            avoid recomputing pr_arg when it's given by p0_arg or p1_arg.
        estimate_rho: Whether estimate rho that result in an overall pr. This
            is forced to be False if pr is a function or if rho is specified.
            To estimate rho, all the pairs with possible connections, meaning
            p0 and p1 are both non-zero for these pairs, are used to estimate
            a value of rho that will result in an expected number of reciprocal
            connections with the given pr. Note that pr is not over all pairs
            of source and target cells but only those has a chance to connect,
            e.g., for only pair of cells within some distance range. The
            estimation is done before generating random connections. The values
            of p0, p0_arg, p1, p1_arg can be cached during estimation of rho
            and retrieved when generating random connections for performance.
        dist_range_forward: If specified, when estimating rho, consider only
            cell pairs whose distance (p0_arg) is within the specified range.
        dist_range_backward: Similar to dist_range_forward but consider
            backward distance range (p1_arg) instead. If both are specified,
            consider only cell pairs whose both distances are within range. If
            neither is specified, infer valid pairs by non-zero connection
            probability.
        rho: The correlation coefficient rho. When specified, do not estimate
            it but instead use the given value throughout, pr will not be used.
            In cases where both p0 and p1 are simple functions, i.e., are
            constant on their support, e.g., function UniformInRange(), the
            estimation of rho will be equal to pr_2_rho(p0, p1, pr) where p0,
            p1 are non-zero. Estimation is not necessary. Directly set rho.
        n_syn0, n_syn1: Number of synapses in the forward and backward
            connection if connected. It can be a constant or a (deterministic
            or random) function whose input arguments are two node objects in
            BMTK like p0_arg, p1_arg. n_syn1 is force to be the same as n_syn0
            when the population is recurrent. Warning: The number must not be
            greater than 255 since it will be converted to uint8 when written
            into the connection matrix to reduce memory consumption.
        autapses: Whether to allow connecting a cell to itself. Default: False.
            This is ignored when the population is not recurrent.
        cache_data: Whether to cache the values of p0, p0_arg, p1, p1_arg
            during estimation of rho. This improves performance when
            estimate_rho is True while not creating a significant overhead in
            the opposite case. However, it requires large memory allocation
            as the population size grows. Set it to False if there is a memory
            issue.
        verbose: Whether show verbose information in console.

    Returns:
        An object that works with BMTK to build edges in a network.

    Important attributes:
        vars: Dictionary that stores part of the original input parameters.
        source, target: NodePool objects for the source and target populations.
        recurrent: Whether the source and target populations are the same.
        callable_set: Set of arguments that are functions but not constants.
        cache: ConnectorCache object for caching data.
        conn_mat: Connection matrix
        stage: Indicator of stage. 0 for forward and 1 for backward connection.
        conn_prop: List of two dictionaries that stores properties of connected
            pairs, for forward and backward connections respectively. In each
            dictionary, each key is the source node id and the value is a
            dictionary, where each key is the target node id that the source
            node connects to, and the value is the value of p0_arg or p1_arg.
            Example: [{sid0: {tid0: p0_arg0, tid1: p0_arg1, ...},
                       sid1: {...}, sid2: {...}, ... },
                      {tid2: {sid3: p1_arg0, sid4: p1_arg1, ...},
                       tid3: {...}, tid4: {...}, ... }]
            This is useful when properties of edges such as distance is used to
            determine other edge properties such as delay. So the distance does
            not need to be calculated repeatedly. The connector can be passed
            as an argument for the functions that generates additional edge
            properties, so that they can access the information here.
    """

    def __init__(self, p0=1., p1=1., symmetric_p1=False,
                 p0_arg=None, p1_arg=None, symmetric_p1_arg=False,
                 pr=0., pr_arg=None, estimate_rho=True, rho=None,
                 dist_range_forward=None, dist_range_backward=None,
                 n_syn0=1, n_syn1=1, autapses=False,
                 cache_data=True, verbose=True):
        args = locals()
        var_set = ('p0', 'p0_arg', 'p1', 'p1_arg',
                   'pr', 'pr_arg', 'n_syn0', 'n_syn1')
        self.vars = {key: args[key] for key in var_set}

        self.symmetric_p1 = symmetric_p1 and symmetric_p1_arg
        self.symmetric_p1_arg = symmetric_p1_arg

        self.estimate_rho = estimate_rho and not callable(pr) and rho is None
        self.dist_range_forward = dist_range_forward
        self.dist_range_backward = dist_range_backward
        self.rho = rho

        self.autapses = autapses
        self.cache = self.ConnectorCache(cache_data and self.estimate_rho)
        self.verbose = verbose

        self.conn_prop = [{}, {}]
        self.stage = 0
        self.iter_count = 0

    # *** Two methods executed during bmtk edge creation net.add_edges() ***
    def setup_nodes(self, source=None, target=None):
        """Must run this before building connections"""
        if self.stage:
            # check whether the correct populations
            if (source is None or target is None or
                    not self.is_same_pop(source, self.target) or
                    not self.is_same_pop(target, self.source)):
                raise ValueError("Source or target population not consistent.")
            # Skip adding nodes for the backward stage.
            return

        # Update node pools
        self.source = source
        self.target = target
        if self.source is None or len(self.source) == 0:
            raise ValueError("Source nodes do not exists")
        if self.target is None or len(self.target) == 0:
            raise ValueError("Target nodes do not exists")

        # Setup nodes
        self.recurrent = self.is_same_pop(self.source, self.target, quick=True)
        self.source_ids = [s.node_id for s in self.source]
        self.n_source = len(self.source_ids)
        self.source_list = list(self.source)
        if self.recurrent:
            self.target_ids = self.source_ids
            self.n_target = self.n_source
            self.target_list = self.source_list
        else:
            self.target_ids = [t.node_id for t in self.target]
            self.n_target = len(self.target_ids)
            self.target_list = list(self.target)

        # Setup for recurrent connection
        if self.recurrent:
            self.symmetric_p1_arg = True
            self.symmetric_p1 = True
            self.vars['n_syn1'] = self.vars['n_syn0']
        if self.symmetric_p1_arg:
            self.vars['p1_arg'] = self.vars['p0_arg']
        if self.symmetric_p1:
            self.vars['p1'] = self.vars['p0']

    def edge_params(self):
        """Create the arguments for BMTK add_edges() method"""
        if self.stage == 0:
            params = {'source': self.source, 'target': self.target,
                      'iterator': 'one_to_all',
                      'connection_rule': self.make_forward_connection}
        else:
            params = {'source': self.target, 'target': self.source,
                      'iterator': 'all_to_one',
                      'connection_rule': self.make_backward_connection}
        self.stage += 1
        return params

    # *** Methods executed during bmtk network.build() ***
    # *** Helper functions ***
    class ConnectorCache(object):
        def __init__(self, enable=True):
            self.enable = enable
            self._output = {}
            self.cache_dict = {}
            self.set_next_it()
            self.write_mode()

        def cache_output(self, func, func_name, cache=True):
            if self.enable and cache:
                self.cache_dict[func_name] = func
                self._output[func_name] = []
                output = self._output[func_name]

                def writer(*args):
                    val = func(*args)
                    output.append(val)
                    return val
                setattr(self, func_name, writer)
            else:
                setattr(self, func_name, func)

        def write_mode(self):
            for val in self._output.values():
                val.clear()
            self.mode = 'write'
            self.iter_count = 0

        def fetch_output(self, func_name, fetch=True):
            output = self._output[func_name]

            if fetch:
                def reader(*args):
                    return output[self.iter_count]
                setattr(self, func_name, reader)
            else:
                setattr(self, func_name, self.cache_dict[func_name])

        def read_mode(self):
            if self.enable and len(self.cache_dict):
                # check whether outputs were written correctly
                output_len = [len(val) for val in self._output.values()]
                # whether any stored and have the same length
                valid = [n for n in output_len if n]
                flag = len(valid) > 0 and all(n == valid[0] for n in valid[1:])
                if flag:
                    for func_name, out_len in zip(self._output, output_len):
                        fetch = out_len > 0
                        if not fetch:
                            print("\nWarning: Cache did not work properly for "
                                  + func_name + '\n')
                        self.fetch_output(func_name, fetch)
                    self.iter_count = 0
                else:
                    # if output not correct, disable and use original function
                    print("\nWarning: Cache did not work properly.\n")
                    for func_name in self.cache_dict:
                        self.fetch_output(func_name, False)
                    self.enable = False
            self.mode = 'read'

        def set_next_it(self):
            if self.enable:
                def next_it():
                    self.iter_count += 1
            else:
                def next_it():
                    pass
            self.next_it = next_it

    def node_2_idx_input(self, var_func, reverse=False):
        """Convert a function that accept nodes as input
        to accept indices as input"""
        if reverse:
            def idx_2_var(j, i):
                return var_func(self.target_list[j], self.source_list[i])
        else:
            def idx_2_var(i, j):
                return var_func(self.source_list[i], self.target_list[j])
        return idx_2_var

    def iterate_pairs(self):
        """Generate indices of source and target for each case"""
        if self.recurrent:
            if self.autapses:
                for i in range(self.n_source):
                    for j in range(i, self.n_target):
                        yield i, j
            else:
                for i in range(self.n_source - 1):
                    for j in range(i + 1, self.n_target):
                        yield i, j
        else:
            for i in range(self.n_source):
                for j in range(self.n_target):
                    yield i, j

    def calc_pair(self, i, j):
        """Calculate intermediate data that can be cached"""
        cache = self.cache
        # cache = self  # test performance for not using cache
        p0_arg = cache.p0_arg(i, j)
        p1_arg = p0_arg if self.symmetric_p1_arg else cache.p1_arg(j, i)
        p0 = cache.p0(p0_arg)
        p1 = p0 if self.symmetric_p1 else cache.p1(p1_arg)
        return p0_arg, p1_arg, p0, p1

    def setup_conditional_backward_probability(self):
        """Create a function that calculates the conditional probability of
        backward connection given the forward connection outcome 'cond'"""
        # For all cases, assume p0, p1, pr are all within [0, 1] already.
        self.wrong_pr = False
        if self.rho is None:
            # Determine by pr for each pair
            if self.verbose:
                def cond_backward(cond, p0, p1, pr):
                    if p0 > 0:
                        pr_bound = (p0 + p1 - 1, min(p0, p1))
                        # check whether pr within bounds
                        if pr < pr_bound[0] or pr > pr_bound[1]:
                            self.wrong_pr = True
                            pr = min(max(pr, pr_bound[0]), pr_bound[1])
                        return pr / p0 if cond else (p1 - pr) / (1 - p0)
                    else:
                        return p1
            else:
                def cond_backward(cond, p0, p1, pr):
                    if p0 > 0:
                        pr_bound = (p0 + p1 - 1, min(p0, p1))
                        pr = min(max(pr, pr_bound[0]), pr_bound[1])
                        return pr / p0 if cond else (p1 - pr) / (1 - p0)
                    else:
                        return p1
        elif self.rho == 0:
            # Independent case
            def cond_backward(cond, p0, p1, pr):
                return p1
        else:
            # Dependent with fixed correlation coefficient rho
            def cond_backward(cond, p0, p1, pr):
                # Standard deviation of r.v. for p1
                sd = ((1 - p1) * p1) ** .5
                # Z-score of random variable for p0
                zs = ((1 - p0) / p0) ** .5 if cond else - (p0 / (1 - p0)) ** .5
                return p1 + self.rho * sd * zs
        self.cond_backward = cond_backward

    def add_conn_prop(self, src, trg, prop, stage=0):
        """Store p0_arg and p1_arg for a connected pair"""
        sid = self.source_ids[src]
        tid = self.target_ids[trg]
        conn_dict = self.conn_prop[stage]
        if stage:
            sid, tid = tid, sid  # during backward, from target to source
        trg_dict = conn_dict.setdefault(sid, {})
        trg_dict[tid] = prop

    def get_conn_prop(self, sid, tid):
        """Get stored value given node ids in a connection"""
        return self.conn_prop[self.stage][sid][tid]

    # *** A sequence of major methods executed during build ***
    def setup_variables(self):
        # If pr_arg is string, use the same value as p0_arg or p1_arg
        if isinstance(self.vars['pr_arg'], str):
            pr_arg_func = 'p1_arg' if '1' in self.vars['pr_arg'] else 'p0_arg'
            self.vars['pr_arg'] = self.vars[pr_arg_func]
        else:
            pr_arg_func = None

        callable_set = set()
        # Make constant variables constant functions
        for name, var in self.vars.items():
            if callable(var):
                callable_set.add(name)  # record callable variables
                setattr(self, name, var)
            else:
                setattr(self, name, self.constant_function(var))
        self.callable_set = callable_set

        # Make callable variables except a few, accept index input instead
        for name in callable_set - {'p0', 'p1', 'pr'}:
            var = self.vars[name]
            setattr(self, name, self.node_2_idx_input(var, '1' in name))

        # Set up function for pr_arg if use value from p0_arg or p1_arg
        if pr_arg_func is None:
            self._pr_arg = self.pr_arg  # use specified pr_arg
        else:
            self._pr_arg_val = 0.  # storing current value from p_arg
            p_arg = getattr(self, pr_arg_func)
            def p_arg_4_pr(*args, **kwargs):
                val = p_arg(*args, **kwargs)
                self._pr_arg_val = val
                return val
            setattr(self, pr_arg_func, p_arg_4_pr)
            def pr_arg(self, *arg):
                return self._pr_arg_val
            self._pr_arg = types.MethodType(pr_arg, self)

    def cache_variables(self):
        # Select cacheable attrilbutes
        cache_set = {'p0', 'p0_arg', 'p1', 'p1_arg'}
        if self.symmetric_p1:
            cache_set.remove('p1')
        if self.symmetric_p1_arg:
            cache_set.remove('p1_arg')
        # Output of callable variables will be cached
        # Constant functions will be called from cache but output not cached
        for name in cache_set:
            var = getattr(self, name)
            self.cache.cache_output(var, name, name in self.callable_set)
        if self.verbose and len(self.cache.cache_dict):
            print('Output of %s will be cached.'
                  % ', '.join(self.cache.cache_dict))

    def setup_dist_range_checker(self):
        # Checker that determines whether to consider a pair for rho estimation
        if self.dist_range_forward is None and self.dist_range_backward is None:
            def checker(var):
                p0, p1 = var[2:]
                return p0 > 0 and p1 > 0
        else:
            def in_range(p_arg, dist_range):
                return p_arg >= dist_range[0] and p_arg <= dist_range[1]
            r0, r1 = self.dist_range_forward, self.dist_range_backward
            if r1 is None:
                def checker(var):
                    return in_range(var[0], r0)
            elif r0 is None:
                def checker(var):
                    return in_range(var[1], r1)
            else:
                def checker(var):
                    return in_range(var[0], r0) and in_range(var[1], r1)
        return checker

    def initialize(self):
        self.setup_variables()
        self.cache_variables()
        # Intialize connection matrix and get nubmer of pairs
        self.end_stage = 0 if self.recurrent else 1
        shape = (self.end_stage + 1, self.n_source, self.n_target)
        self.conn_mat = np.zeros(shape, dtype=np.uint8)  # 1 byte per entry

    def initial_all_to_all(self):
        """The major part of the algorithm run at beginning of BMTK iterator"""
        if self.verbose:
            src_str, trg_str = self.get_nodes_info()
            print("\nStart building connection between: \n  "
                  + src_str + "\n  " + trg_str)
        self.initialize()
        cache = self.cache  # write mode

        # Estimate pr
        if self.verbose:
            self.timer = Timer()
        if self.estimate_rho:
            dist_range_checker = self.setup_dist_range_checker()
            p0p1_sum = 0.
            norm_fac_sum = 0.
            n = 0
            # Make sure each cacheable function runs excatly once per iteration
            for i, j in self.iterate_pairs():
                var = self.calc_pair(i, j)
                valid = dist_range_checker(var)
                if valid:
                    n += 1
                    p0, p1 = var[2:]
                    p0p1_sum += p0 * p1
                    norm_fac_sum += (p0 * (1 - p0) * p1 * (1 - p1)) ** .5
            if norm_fac_sum > 0:
                rho = (self.pr() * n - p0p1_sum) / norm_fac_sum
                if abs(rho) > 1:
                    print("\nWarning: Estimated value of rho=%.3f "
                          "outside the range [-1, 1]." % rho)
                    rho = np.clip(rho, -1, 1).item()
                    print("Force rho to be %.0f.\n" % rho)
                elif self.verbose:
                    print("Estimated value of rho=%.3f" % rho)
                self.rho = rho
            else:
                self.rho = 0

            if self.verbose:
                self.timer.report('Time for estimating rho')

        # Setup function for calculating conditional backward probability
        self.setup_conditional_backward_probability()

        # Make random connections
        cache.read_mode()
        possible_count = 0 if self.recurrent else np.zeros(3)
        for i, j in self.iterate_pairs():
            p0_arg, p1_arg, p0, p1 = self.calc_pair(i, j)
            # Check whether at all possible and count
            forward = p0 > 0
            backward = p1 > 0
            if self.recurrent:
                possible_count += forward
            else:
                possible_count += [forward, backward, forward and backward]

            # Make random decision
            if forward:
                forward = decision(p0)
            if backward:
                pr = self.pr(self._pr_arg(i, j), p0, p1)
                backward = decision(self.cond_backward(forward, p0, p1, pr))

            # Make connection
            if forward:
                n_forward = self.n_syn0(i, j)
                self.add_conn_prop(i, j, p0_arg, 0)
                self.conn_mat[0, i, j] = n_forward
            if backward:
                n_backward = self.n_syn1(j, i)
                if self.recurrent:
                    if i != j:
                        self.conn_mat[0, j, i] = n_backward
                        self.add_conn_prop(j, i, p1_arg, 0)
                else:
                    self.conn_mat[1, i, j] = n_backward
                    self.add_conn_prop(i, j, p1_arg, 1)
            self.cache.next_it()
        self.cache.write_mode()  # clear memory
        self.possible_count = possible_count

        if self.verbose:
            self.timer.report('Total time for creating connection matrix')
            if self.wrong_pr:
                print("Warning: Value of 'pr' outside the bounds occurred.\n")
            self.connection_number_info()

    def make_connection(self):
        """ Assign number of synapses per iteration.
        Use iterator one_to_all for forward and all_to_one for backward.
        """
        nsyns = self.conn_mat[self.stage, self.iter_count, :]
        self.iter_count += 1

        # Detect end of iteration
        if self.iter_count == self.n_source:
            self.iter_count = 0
            if self.stage == self.end_stage:
                if self.verbose:
                    self.timer.report('Done! \nTime for building connections')
                self.free_memory()
        return nsyns

    def make_forward_connection(self, source, targets, *args, **kwargs):
        """Function to be called by BMTK iterator for forward connection"""
        # Initialize in the first iteration
        if self.iter_count == 0:
            self.stage = 0
            self.initial_all_to_all()
            if self.verbose:
                print("Assigning forward connections.")
                self.timer.start()
        return self.make_connection()

    def make_backward_connection(self, targets, source, *args, **kwargs):
        """Function to be called by BMTK iterator for backward connection"""
        if self.iter_count == 0:
            self.stage = 1
            if self.verbose:
                print("Assigning backward connections.")
        return self.make_connection()

    def free_memory(self):
        """Free up memory after connections are built"""
        # Do not clear self.conn_prop if it will be used by conn.add_properties
        variables = ('conn_mat', 'source_list', 'target_list',
                     'source_ids', 'target_ids')
        for var in variables:
            setattr(self, var, None)

    # *** Helper functions for verbose ***
    def get_nodes_info(self):
        """Get strings with source and target population information"""
        source_str = self.source.network_name + ': ' + self.source.filter_str
        target_str = self.target.network_name + ': ' + self.target.filter_str
        return source_str, target_str

    def connection_number(self):
        """
        Return the number of the following:
        n_conn: connected pairs [forward, (backward,) reciprocal]
        n_poss: possible connections (prob>0) [forward, (backward, reciprocal)]
        n_pair: pairs of cells
        proportion: of connections in possible and total pairs
        """
        conn_mat = self.conn_mat.astype(bool)
        n_conn = np.count_nonzero(conn_mat, axis=(1, 2))
        n_poss = np.array(self.possible_count)
        n_pair = conn_mat.size / 2
        if self.recurrent:
            n_recp = np.count_nonzero(conn_mat[0] & conn_mat[0].T)
            if self.autapses:
                n_recp -= np.count_nonzero(np.diag(conn_mat[0]))
            n_recp //= 2
            n_conn -= n_recp
            n_poss = n_poss[None]
            n_pair += (1 if self.autapses else -1) * self.n_source / 2
        else:
            n_recp = np.count_nonzero(conn_mat[0] & conn_mat[1])
        n_conn = np.append(n_conn, n_recp)
        n_pair = int(n_pair)
        fraction = np.array([n_conn / n_poss, n_conn / n_pair])
        fraction[np.isnan(fraction)] = 0.
        return n_conn, n_poss, n_pair, fraction

    def connection_number_info(self):
        """Print connection numbers after connections built"""
        def arr2str(a, f):
            return ', '.join([f] * a.size) % tuple(a.tolist())
        n_conn, n_poss, n_pair, fraction = self.connection_number()
        conn_type = "(all, reciprocal)" if self.recurrent \
                    else "(forward, backward, reciprocal)"
        print("Numbers of " + conn_type + " connections:")
        print("Number of connected pairs: (%s)" % arr2str(n_conn, '%d'))
        print("Number of possible connections: (%s)" % arr2str(n_poss, '%d'))
        print("Fraction of connected pairs in possible ones: (%s)"
              % arr2str(100 * fraction[0], '%.2f%%'))
        print("Number of total pairs: %d" % n_pair)
        print("Fraction of connected pairs in all pairs: (%s)\n"
              % arr2str(100 * fraction[1], '%.2f%%'))


class UnidirectionConnector(AbstractConnector):
    """
    Object for buiilding unidirectional connections in bmtk network model with
    given probability within a single population (or between two populations).

    Parameters:
        p, p_arg: Probability of forward connection and its input argument when
            it is a function, similar to p0, p0_arg in ReciprocalConnector. It
            can be a constant or a deterministic function whose value must be
            within range [0, 1]. When p is constant, the connection is
            homogenous.
        n_syn: Number of synapses in the forward connection if connected. It
            can be a constant or a (deterministic or random) function whose
            input arguments are two node objects in BMTK like p_arg.
        verbose: Whether show verbose information in console.

    Returns:
        An object that works with BMTK to build edges in a network.

    Important attributes:
        vars: Dictionary that stores part of the original input parameters.
        source, target: NodePool objects for the source and target populations.
        conn_prop: A dictionaries that stores properties of connected pairs.
        Each key is the source node id and the value is a dictionary, where
        each key is the target node id that the source node connects to, and
        the value is the value of p_arg.
            Example: {sid0: {tid0: p_arg0, tid1: p_arg1, ...},
                      sid1: {...}, sid2: {...}, ... }
            This is useful in similar manner as in ReciprocalConnector.
    """

    def __init__(self, p=1., p_arg=None, n_syn=1, verbose=True):
        args = locals()
        var_set = ('p', 'p_arg', 'n_syn')
        self.vars = {key: args[key] for key in var_set}

        self.verbose = verbose
        self.conn_prop = {}
        self.iter_count = 0

    # *** Two methods executed during bmtk edge creation net.add_edges() ***
    def setup_nodes(self, source=None, target=None):
        """Must run this before building connections"""
        # Update node pools
        self.source = source
        self.target = target
        if self.source is None or len(self.source) == 0:
            raise ValueError("Source nodes do not exists")
        if self.target is None or len(self.target) == 0:
            raise ValueError("Target nodes do not exists")
        self.n_pair = len(self.source) * len(self.target)

    def edge_params(self):
        """Create the arguments for BMTK add_edges() method"""
        params = {'source': self.source, 'target': self.target,
                  'iterator': 'one_to_one',
                  'connection_rule': self.make_connection}
        return params

    # *** Methods executed during bmtk network.build() ***
    # *** Helper functions ***
    def add_conn_prop(self, sid, tid, prop):
        """Store p0_arg and p1_arg for a connected pair"""
        trg_dict = self.conn_prop.setdefault(sid, {})
        trg_dict[tid] = prop

    def get_conn_prop(self, sid, tid):
        """Get stored value given node ids in a connection"""
        return self.conn_prop[sid][tid]

    def setup_variables(self):
        """Make constant variables constant functions"""
        for name, var in self.vars.items():
            if callable(var):
                setattr(self, name, var)
            else:
                setattr(self, name, self.constant_function(var))

    def initialize(self):
        self.setup_variables()
        self.n_conn = 0
        self.n_poss = 0
        if self.verbose:
            self.timer = Timer()

    def make_connection(self, source, target, *args, **kwargs):
        """Assign number of synapses per iteration using one_to_one iterator"""
        # Initialize in the first iteration
        if self.iter_count == 0:
            self.initialize()
            if self.verbose:
                src_str, trg_str = self.get_nodes_info()
                print("\nStart building connection \n  from "
                      + src_str + "\n  to " + trg_str)

        # Make random connections
        p_arg = self.p_arg(source, target)
        p = self.p(p_arg)
        possible = p > 0
        self.n_poss += possible
        if possible and decision(p):
            nsyns = self.n_syn(source, target)
            self.add_conn_prop(source.node_id, target.node_id, p_arg)
            self.n_conn += 1
        else:
            nsyns = 0

        self.iter_count += 1

        # Detect end of iteration
        if self.iter_count == self.n_pair:
            if self.verbose:
                self.connection_number_info()
                self.timer.report('Done! \nTime for building connections')
        return nsyns

    # *** Helper functions for verbose ***
    def get_nodes_info(self):
        """Get strings with source and target population information"""
        source_str = self.source.network_name + ': ' + self.source.filter_str
        target_str = self.target.network_name + ': ' + self.target.filter_str
        return source_str, target_str

    def connection_number_info(self):
        """Print connection numbers after connections built"""
        print("Number of connected pairs: %d" % self.n_conn)
        print("Number of possible connections: %d" % self.n_poss)
        print("Fraction of connected pairs in possible ones: %.2f%%"
              % (100. * self.n_conn / self.n_poss) if self.n_poss else 0.)
        print("Number of total pairs: %d" % self.n_pair)
        print("Fraction of connected pairs in all pairs: %.2f%%\n"
              % (100. * self.n_conn / self.n_pair))


class GapJunction(UnidirectionConnector):
    """
    Object for buiilding gap junction connections in bmtk network model with
    given probabilities within a single population which is uncorrelated with
    the recurrent chemical synapses in this population.

    Parameters:
        p, p_arg: Probability of forward connection and its input argument when
            it is a function, similar to p0, p0_arg in ReciprocalConnector. It
            can be a constant or a deterministic function whose value must be
            within range [0, 1]. When p is constant, the connection is
            homogenous.
        verbose: Whether show verbose information in console.

    Returns:
        An object that works with BMTK to build edges in a network.

    Important attributes:
        Similar to `UnidirectionConnector`.
    """

    def __init__(self, p=1., p_arg=None, verbose=True):
        super().__init__(p=p, p_arg=p_arg, verbose=verbose)

    def setup_nodes(self, source=None, target=None):
        super().setup_nodes(source=source, target=target)
        if len(self.source) != len(self.target):
            raise ValueError("Source and target must be the same for "
                             "gap junction.")
        self.n_source = len(self.source)

    def make_connection(self, source, target, *args, **kwargs):
        """Assign gap junction per iteration using one_to_one iterator"""
        # Initialize in the first iteration
        if self.iter_count == 0:
            self.initialize()
            if self.verbose:
                src_str, _ = self.get_nodes_info()
                print("\nStart building gap junction \n  in " + src_str)

        # Consider each pair only once
        nsyns = 0
        i, j = divmod(self.iter_count, self.n_source)
        if i < j:
            p_arg = self.p_arg(source, target)
            p = self.p(p_arg)
            possible = p > 0
            self.n_poss += possible
            if possible and decision(p):
                nsyns = 1
                sid, tid = source.node_id, target.node_id
                self.add_conn_prop(sid, tid, p_arg)
                self.add_conn_prop(tid, sid, p_arg)
                self.n_conn += 1

        self.iter_count += 1

        # Detect end of iteration
        if self.iter_count == self.n_pair:
            if self.verbose:
                self.connection_number_info()
                self.timer.report('Done! \nTime for building connections')
        return nsyns

    def connection_number_info(self):
        n_pair = self.n_pair
        self.n_pair = (n_pair - len(self.source)) // 2
        super().connection_number_info()
        self.n_pair = n_pair


class CorrelatedGapJunction(GapJunction):
    """
    Object for buiilding gap junction connections in bmtk network model with
    given probabilities within a single population which could be correlated
    with the recurrent chemical synapses in this population.

    Parameters:
        p_non, p_uni, p_rec: Probabilities of gap junction connection for each
            pair of cells given the following three conditions of chemical
            synaptic connections between them, no connection, unidirectional,
            and reciprocal, respectively. It can be a constant or a
            deterministic function whose value must be within range [0, 1].
        p_arg: Input argument for p_non, p_uni, or p_rec, when any of them is a
            function, similar to p0_arg, p1_arg in ReciprocalConnector.
        connector: Connector object used to generate the chemical synapses of
            within this population, which contains the connection information
            in its attribute `conn_prop`. So this connector should have
            generated the chemical synapses before generating the gap junction.
        verbose: Whether show verbose information in console.

    Returns:
        An object that works with BMTK to build edges in a network.

    Important attributes:
        Similar to `UnidirectionConnector`.
    """

    def __init__(self, p_non=1., p_uni=1., p_rec=1., p_arg=None,
                 connector=None, verbose=True):
        super().__init__(p=p_non, p_arg=p_arg, verbose=verbose)
        self.vars['p_non'] = self.vars.pop('p')
        self.vars['p_uni'] = p_uni
        self.vars['p_rec'] = p_rec
        self.connector = connector
        conn_prop = connector.conn_prop
        if isinstance(conn_prop, list):
            conn_prop = conn_prop[0]
        self.ref_conn_prop = conn_prop

    def conn_exist(self, sid, tid):
        trg_dict = self.ref_conn_prop.get(sid)
        if trg_dict is not None and tid in trg_dict:
            return True, trg_dict[tid]
        else:
            return False, None

    def connection_type(self, sid, tid):
        conn0, prop0 = self.conn_exist(sid, tid)
        conn1, prop1 = self.conn_exist(tid, sid)
        return conn0 + conn1, prop0 if conn0 else prop1

    def initialize(self):
        self.has_p_arg = self.vars['p_arg'] is not None
        if not self.has_p_arg:
            var = self.connector.vars
            self.vars['p_arg'] = var.get('p_arg', var.get('p0_arg', None))
        super().initialize()
        self.ps = [self.p_non, self.p_uni, self.p_rec]

    def make_connection(self, source, target, *args, **kwargs):
        """Assign gap junction per iteration using one_to_one iterator"""
        # Initialize in the first iteration
        if self.iter_count == 0:
            self.initialize()
            if self.verbose:
                src_str, _ = self.get_nodes_info()
                print("\nStart building gap junction \n  in " + src_str)

        # Consider each pair only once
        nsyns = 0
        i, j = divmod(self.iter_count, self.n_source)
        if i < j:
            sid, tid = source.node_id, target.node_id
            conn_type, p_arg = self.connection_type(sid, tid)
            if self.has_p_arg or not conn_type:
                p_arg = self.p_arg(source, target)
            p = self.ps[conn_type](p_arg)
            possible = p > 0
            self.n_poss += possible
            if possible and decision(p):
                nsyns = 1
                self.add_conn_prop(sid, tid, p_arg)
                self.add_conn_prop(tid, sid, p_arg)
                self.n_conn += 1

        self.iter_count += 1

        # Detect end of iteration
        if self.iter_count == self.n_pair:
            if self.verbose:
                self.connection_number_info()
                self.timer.report('Done! \nTime for building connections')
        return nsyns


class OneToOneSequentialConnector(AbstractConnector):
    """Object for buiilding one to one correspondence connections in bmtk
    network model with between two populations. One of the population can
    consist of multiple sub-populations. These sub-populations need to be added
    sequentially using setup_nodes() and edge_params() methods followed by BMTK
    add_edges() method. For example, to connect 30 nodes in population A to 30
    nodes in populations B1, B2, B3, each with 10 nodes, set up as follows.
        connector = OneToOneSequentialConnector(**parameters)
        connector.setup_nodes(source=A, target=B1)
        net.add_edges(**connector.edge_params())
        connector.setup_nodes(target=B2)
        net.add_edges(**connector.edge_params())
        connector.setup_nodes(target=B3)
        net.add_edges(**connector.edge_params())
    After BMTK executes net.build(), the first 10 nodes in A will connect one-
    to-one to the 10 nodes in B1, then the 11 to 20-th nodes to those in B2,
    finally the 21 to 30-th nodes to those in B3.
    This connector is useful for creating input drives to a population. Each
    node in it receives an independent drive from a unique source node.

    Parameters:
        n_syn: Number of synapses in each connection. It accepts only constant
            for now.
        partition_source: Whether the source population consists of multiple
            sub-populations. By default, the source has one population, and the
            target can have multiple sub-populations. If set to true, the
            source can have multiple sub-populations and the target has only
            one population.
        verbose: Whether show verbose information in console.

    Returns:
        An object that works with BMTK to build edges in a network.

    Important attributes:
        source: NodePool object for the single population.
        targets: List of NodePool objects for the multiple sub-populations.
    """

    def __init__(self, n_syn=1, partition_source=False, verbose=True):
        self.n_syn = int(n_syn)
        self.partition_source = partition_source
        self.verbose = verbose

        self.targets = []
        self.n_source = 0
        self.idx_range = [0]
        self.target_count = 0
        self.iter_count = 0

    # *** Two methods executed during bmtk edge creation net.add_edges() ***
    def setup_nodes(self, source=None, target=None):
        """Must run this before building connections"""
        # Update node pools
        if self.partition_source:
            source, target = target, source
        if self.target_count == 0:
            if source is None or len(source) == 0:
                raise ValueError(("Target" if self.partition_source else
                                  "Source") + " nodes do not exists")
            self.source = source
            self.n_source = len(source)
        if target is None or len(target) == 0:
            raise ValueError(("Source" if self.partition_source else
                              "Target") + " nodes do not exists")

        self.targets.append(target)
        self.idx_range.append(self.idx_range[-1] + len(target))
        self.target_count += 1

        if self.idx_range[-1] > self.n_source:
            if self.partition_source:
                raise ValueError(
                    "Total target populations exceed the source population."
                    if self.partition_source else
                    "Total source populations exceed the target population."
                    )

        if self.verbose and self.idx_range[-1] == self.n_source:
            print("All " + ("source" if self.partition_source else "target")
                  + " population partitions are filled.")

    def edge_params(self, target_pop_idx=-1):
        """Create the arguments for BMTK add_edges() method"""
        if self.partition_source:
            params = {'source': self.targets[target_pop_idx],
                      'target': self.source,
                      'iterator': 'one_to_all'}
        else:
            params = {'source': self.source,
                      'target': self.targets[target_pop_idx],
                      'iterator': 'all_to_one'}
        params['connection_rule'] = self.make_connection
        return params

    # *** Methods executed during bmtk network.build() ***
    def make_connection(self, source, targets, *args, **kwargs):
        """Assign one connection per iteration using all_to_one iterator"""
        # Initialize in the first iteration
        if self.verbose:
            if self.iter_count == 0:
                # Very beginning
                self.target_count = 0
                src_str, trg_str = self.get_nodes_info()
                print("\nStart building connection " +
                      ("to " if self.partition_source else "from ") + src_str)
                self.timer = Timer()

            if self.iter_count == self.idx_range[self.target_count]:
                # Beginning of each target population
                src_str, trg_str = self.get_nodes_info(self.target_count)
                print(("  %d. " % self.target_count) +
                      ("from " if self.partition_source else "to ") + trg_str)
                self.target_count += 1
                self.timer_part = Timer()

        # Make connection
        nsyns = np.zeros(self.n_source, dtype=int)
        nsyns[self.iter_count] = self.n_syn
        self.iter_count += 1

        # Detect end of iteration
        if self.verbose:
            if self.iter_count == self.idx_range[self.target_count]:
                # End of each target population
                self.timer_part.report('    Time for this partition')
            if self.iter_count == self.n_source:
                # Very end
                self.timer.report('Done! \nTime for building connections')
        return nsyns

    # *** Helper functions for verbose ***
    def get_nodes_info(self, target_pop_idx=-1):
        """Get strings with source and target population information"""
        target = self.targets[target_pop_idx]
        source_str = self.source.network_name + ': ' + self.source.filter_str
        target_str = target.network_name + ': ' + target.filter_str
        return source_str, target_str


##############################################################################
######################### ADDTIONAL EDGE PROPERTIES ##########################

SYN_MIN_DELAY = 0.8  # ms
SYN_VELOCITY = 1000.  # um/ms
FLUC_STDEV = 0.2  # ms
DELAY_LOWBOUND = 0.2  # ms must be greater than h.dt
DELAY_UPBOUND = 2.0  # ms

def syn_dist_delay_feng(source, target, min_delay=SYN_MIN_DELAY,
                        velocity=SYN_VELOCITY, fluc_stdev=FLUC_STDEV,
                        delay_bound=(DELAY_LOWBOUND, DELAY_UPBOUND),
                        connector=None):
    """Synpase delay linearly dependent on distance.
    min_delay: minimum delay (ms)
    velocity: synapse conduction velocity (micron/ms)
    fluc_stdev: standard deviation of random Gaussian fluctuation (ms)
    delay_bound: (lower, upper) bounds of delay (ms)
    connector: connector object from which to read distance
    """
    if connector is None:
        dist = euclid_dist(target['positions'], source['positions'])
    else:
        dist = connector.get_conn_prop(source.node_id, target.node_id)
    del_fluc = fluc_stdev * rng.normal()
    delay = dist / velocity + min_delay + del_fluc
    delay = min(max(delay, delay_bound[0]), delay_bound[1])
    return delay


def syn_section_PN(source, target, p=0.9,
                   sec_id=(1, 2), sec_x=(0.4, 0.6), **kwargs):
    """Synapse location follows a Bernoulli distribution, with probability p
    to obtain the former in sec_id and sec_x"""
    syn_loc = int(not decision(p))
    return sec_id[syn_loc], sec_x[syn_loc]


def syn_dist_delay_feng_section_PN(source, target, p=0.9,
                                   sec_id=(1, 2), sec_x=(0.4, 0.6), **kwargs):
    """Assign both synapse delay and location"""
    delay = syn_dist_delay_feng(source, target, **kwargs)
    s_id, s_x = syn_section_PN(source, target, p=p, sec_id=sec_id, sec_x=sec_x)
    return delay, s_id, s_x


def syn_uniform_delay_section(source, target, low=DELAY_LOWBOUND,
                              high=DELAY_UPBOUND, **kwargs):
    return rng.uniform(low, high)
