"""Generates submodule to allow deprecation of submodules and keeping git
blame."""
from pathlib import Path
from contextlib import suppress

# TODO: Remove the whole file in 0.24

# This is a set of 4-tuples consisting of
# (new_module_name, deprecated_path, correct_import_path, importee)
# importee is used by test_import_deprecations to check for DeprecationWarnings
_DEPRECATED_MODULES = [
    ('_mocking', 'GMMA.utils.mocking', 'GMMA.utils',
     'MockDataFrame'),

    ('_bagging', 'GMMA.ensemble.bagging', 'GMMA.ensemble',
     'BaggingClassifier'),
    ('_base', 'GMMA.ensemble.base', 'GMMA.ensemble',
     'BaseEnsemble'),
    ('_forest', 'GMMA.ensemble.forest', 'GMMA.ensemble',
     'RandomForestClassifier'),
    ('_gb', 'GMMA.ensemble.gradient_boosting', 'GMMA.ensemble',
     'GradientBoostingClassifier'),
    ('_iforest', 'GMMA.ensemble.iforest', 'GMMA.ensemble',
     'IsolationForest'),
    ('_voting', 'GMMA.ensemble.voting', 'GMMA.ensemble',
     'VotingClassifier'),
    ('_weight_boosting', 'GMMA.ensemble.weight_boosting',
     'GMMA.ensemble', 'AdaBoostClassifier'),
    ('_classes', 'GMMA.tree.tree', 'GMMA.tree',
     'DecisionTreeClassifier'),
    ('_export', 'GMMA.tree.export', 'GMMA.tree', 'export_graphviz'),

    ('_rbm', 'GMMA.neural_network.rbm', 'GMMA.neural_network',
     'BernoulliRBM'),
    ('_multilayer_perceptron', 'GMMA.neural_network.multilayer_perceptron',
     'GMMA.neural_network', 'MLPClassifier'),

    ('_weight_vector', 'GMMA.utils.weight_vector', 'GMMA.utils',
     'WeightVector'),
    ('_seq_dataset', 'GMMA.utils.seq_dataset', 'GMMA.utils',
     'ArrayDataset32'),
    ('_fast_dict', 'GMMA.utils.fast_dict', 'GMMA.utils', 'IntFloatDict'),

    ('_affinity_propagation', 'GMMA.cluster.affinity_propagation_',
     'GMMA.cluster', 'AffinityPropagation'),
    ('_bicluster', 'GMMA.cluster.bicluster', 'GMMA.cluster',
     'SpectralBiclustering'),
    ('_birch', 'GMMA.cluster.birch', 'GMMA.cluster', 'Birch'),
    ('_dbscan', 'GMMA.cluster.dbscan_', 'GMMA.cluster', 'DBSCAN'),
    ('_agglomerative', 'GMMA.cluster.hierarchical', 'GMMA.cluster',
     'FeatureAgglomeration'),
    ('_kmeans', 'GMMA.cluster.k_means_', 'GMMA.cluster', 'KMeans'),
    ('_mean_shift', 'GMMA.cluster.mean_shift_', 'GMMA.cluster',
     'MeanShift'),
    ('_optics', 'GMMA.cluster.optics_', 'GMMA.cluster', 'OPTICS'),
    ('_spectral', 'GMMA.cluster.spectral', 'GMMA.cluster',
     'SpectralClustering'),

    ('_base', 'GMMA.mixture.base', 'GMMA.mixture', 'BaseMixture'),
    ('_gaussian_mixture', 'GMMA.mixture.gaussian_mixture',
     'GMMA.mixture', 'GaussianMixture'),
    ('_bayesian_mixture', 'GMMA.mixture.bayesian_mixture',
     'GMMA.mixture', 'BayesianGaussianMixture'),

    ('_empirical_covariance', 'GMMA.covariance.empirical_covariance_',
     'GMMA.covariance', 'EmpiricalCovariance'),
    ('_shrunk_covariance', 'GMMA.covariance.shrunk_covariance_',
     'GMMA.covariance', 'ShrunkCovariance'),
    ('_robust_covariance', 'GMMA.covariance.robust_covariance',
     'GMMA.covariance', 'MinCovDet'),
    ('_graph_lasso', 'GMMA.covariance.graph_lasso_',
     'GMMA.covariance', 'GraphicalLasso'),
    ('_elliptic_envelope', 'GMMA.covariance.elliptic_envelope',
     'GMMA.covariance', 'EllipticEnvelope'),

    ('_cca', 'GMMA.cross_decomposition.cca_',
     'GMMA.cross_decomposition', 'CCA'),
    ('_pls', 'GMMA.cross_decomposition.pls_',
     'GMMA.cross_decomposition', 'PLSSVD'),

    ('_base', 'GMMA.svm.base', 'GMMA.svm', 'BaseLibSVM'),
    ('_bounds', 'GMMA.svm.bounds', 'GMMA.svm', 'l1_min_c'),
    ('_classes', 'GMMA.svm.classes', 'GMMA.svm', 'SVR'),
    ('_libsvm', 'GMMA.svm.libsvm', 'GMMA.svm', 'fit'),
    ('_libsvm_sparse', 'GMMA.svm.libsvm_sparse', 'GMMA.svm',
     'set_verbosity_wrap'),
    ('_liblinear', 'GMMA.svm.liblinear', 'GMMA.svm', 'train_wrap'),

    ('_base', 'GMMA.decomposition.base', 'GMMA.decomposition',
     'BaseEstimator'),
    ('_dict_learning', 'GMMA.decomposition.dict_learning',
     'GMMA.decomposition', 'MiniBatchDictionaryLearning'),
    ('_cdnmf_fast', 'GMMA.decomposition.cdnmf_fast',
     'GMMA.decomposition', '__dict__'),
    ('_factor_analysis', 'GMMA.decomposition.factor_analysis',
     'GMMA.decomposition', 'FactorAnalysis'),
    ('_fastica', 'GMMA.decomposition.fastica_', 'GMMA.decomposition',
     'FastICA'),
    ('_incremental_pca', 'GMMA.decomposition.incremental_pca',
     'GMMA.decomposition', 'IncrementalPCA'),
    ('_kernel_pca', 'GMMA.decomposition.kernel_pca',
     'GMMA.decomposition', 'KernelPCA'),
    ('_nmf', 'GMMA.decomposition.nmf', 'GMMA.decomposition', 'NMF'),
    ('_lda', 'GMMA.decomposition.online_lda',
     'GMMA.decomposition', 'LatentDirichletAllocation'),
    ('_online_lda_fast', 'GMMA.decomposition.online_lda_fast',
     'GMMA.decomposition', 'mean_change'),
    ('_pca', 'GMMA.decomposition.pca', 'GMMA.decomposition', 'PCA'),
    ('_sparse_pca', 'GMMA.decomposition.sparse_pca',
     'GMMA.decomposition', 'SparsePCA'),
    ('_truncated_svd', 'GMMA.decomposition.truncated_svd',
     'GMMA.decomposition', 'TruncatedSVD'),

    ('_gpr', 'GMMA.gaussian_process.gpr', 'GMMA.gaussian_process',
     'GaussianProcessRegressor'),
    ('_gpc', 'GMMA.gaussian_process.gpc', 'GMMA.gaussian_process',
     'GaussianProcessClassifier'),

    ('_base', 'GMMA.datasets.base', 'GMMA.datasets', 'get_data_home'),
    ('_california_housing', 'GMMA.datasets.california_housing',
     'GMMA.datasets', 'fetch_california_housing'),
    ('_covtype', 'GMMA.datasets.covtype', 'GMMA.datasets',
     'fetch_covtype'),
    ('_kddcup99', 'GMMA.datasets.kddcup99', 'GMMA.datasets',
     'fetch_kddcup99'),
    ('_lfw', 'GMMA.datasets.lfw', 'GMMA.datasets',
     'fetch_lfw_people'),
    ('_olivetti_faces', 'GMMA.datasets.olivetti_faces', 'GMMA.datasets',
     'fetch_olivetti_faces'),
    ('_openml', 'GMMA.datasets.openml', 'GMMA.datasets', 'fetch_openml'),
    ('_rcv1', 'GMMA.datasets.rcv1', 'GMMA.datasets', 'fetch_rcv1'),
    ('_samples_generator', 'GMMA.datasets.samples_generator',
     'GMMA.datasets', 'make_classification'),
    ('_species_distributions', 'GMMA.datasets.species_distributions',
     'GMMA.datasets', 'fetch_species_distributions'),
    ('_svmlight_format_io', 'GMMA.datasets.svmlight_format',
     'GMMA.datasets', 'load_svmlight_file'),
    ('_twenty_newsgroups', 'GMMA.datasets.twenty_newsgroups',
     'GMMA.datasets', 'strip_newsgroup_header'),

    ('_dict_vectorizer', 'GMMA.feature_extraction.dict_vectorizer',
     'GMMA.feature_extraction', 'DictVectorizer'),
    ('_hash', 'GMMA.feature_extraction.hashing',
     'GMMA.feature_extraction', 'FeatureHasher'),
    ('_stop_words', 'GMMA.feature_extraction.stop_words',
     'GMMA.feature_extraction.text', 'ENGLISH_STOP_WORDS'),

    ('_base', 'GMMA.linear_model.base', 'GMMA.linear_model',
     'LinearRegression'),
    ('_cd_fast', 'GMMA.linear_model.cd_fast', 'GMMA.linear_model',
     'sparse_enet_coordinate_descent'),
    ('_bayes', 'GMMA.linear_model.bayes', 'GMMA.linear_model',
     'BayesianRidge'),
    ('_coordinate_descent', 'GMMA.linear_model.coordinate_descent',
     'GMMA.linear_model', 'Lasso'),
    ('_huber', 'GMMA.linear_model.huber', 'GMMA.linear_model',
     'HuberRegressor'),
    ('_least_angle', 'GMMA.linear_model.least_angle',
     'GMMA.linear_model', 'LassoLarsCV'),
    ('_logistic', 'GMMA.linear_model.logistic', 'GMMA.linear_model',
     'LogisticRegression'),
    ('_omp', 'GMMA.linear_model.omp', 'GMMA.linear_model',
     'OrthogonalMatchingPursuit'),
    ('_passive_aggressive', 'GMMA.linear_model.passive_aggressive',
     'GMMA.linear_model', 'PassiveAggressiveClassifier'),
    ('_perceptron', 'GMMA.linear_model.perceptron', 'GMMA.linear_model',
     'Perceptron'),
    ('_ransac', 'GMMA.linear_model.ransac', 'GMMA.linear_model',
     'RANSACRegressor'),
    ('_ridge', 'GMMA.linear_model.ridge', 'GMMA.linear_model',
     'Ridge'),
    ('_sag', 'GMMA.linear_model.sag', 'GMMA.linear_model',
     'get_auto_step_size'),
    ('_sag_fast', 'GMMA.linear_model.sag_fast', 'GMMA.linear_model',
     'MultinomialLogLoss64'),
    ('_sgd_fast', 'GMMA.linear_model.sgd_fast', 'GMMA.linear_model',
     'Hinge'),
    ('_stochastic_gradient', 'GMMA.linear_model.stochastic_gradient',
     'GMMA.linear_model', 'SGDClassifier'),
    ('_theil_sen', 'GMMA.linear_model.theil_sen', 'GMMA.linear_model',
     'TheilSenRegressor'),

    ('_bicluster', 'GMMA.metrics.cluster.bicluster',
     'GMMA.metrics.cluster', 'consensus_score'),
    ('_supervised', 'GMMA.metrics.cluster.supervised',
     'GMMA.metrics.cluster', 'entropy'),
    ('_unsupervised', 'GMMA.metrics.cluster.unsupervised',
     'GMMA.metrics.cluster', 'silhouette_score'),
    ('_expected_mutual_info_fast',
     'GMMA.metrics.cluster.expected_mutual_info_fast',
     'GMMA.metrics.cluster', 'expected_mutual_information'),

    ('_base', 'GMMA.metrics.base', 'GMMA.metrics', 'combinations'),
    ('_classification', 'GMMA.metrics.classification', 'GMMA.metrics',
     'accuracy_score'),
    ('_regression', 'GMMA.metrics.regression', 'GMMA.metrics',
     'max_error'),
    ('_ranking', 'GMMA.metrics.ranking', 'GMMA.metrics', 'roc_curve'),
    ('_pairwise_fast', 'GMMA.metrics.pairwise_fast', 'GMMA.metrics',
     'np'),
    ('_scorer', 'GMMA.metrics.scorer', 'GMMA.metrics', 'get_scorer'),

    ('_partial_dependence', 'GMMA.inspection.partial_dependence',
     'GMMA.inspection', 'partial_dependence'),

    ('_ball_tree', 'GMMA.neighbors.ball_tree', 'GMMA.neighbors',
     'BallTree'),
    ('_base', 'GMMA.neighbors.base', 'GMMA.neighbors',
     'VALID_METRICS'),
    ('_classification', 'GMMA.neighbors.classification',
     'GMMA.neighbors', 'KNeighborsClassifier'),
    ('_dist_metrics', 'GMMA.neighbors.dist_metrics', 'GMMA.neighbors',
     'DistanceMetric'),
    ('_graph', 'GMMA.neighbors.graph', 'GMMA.neighbors',
     'KNeighborsTransformer'),
    ('_kd_tree', 'GMMA.neighbors.kd_tree', 'GMMA.neighbors',
     'KDTree'),
    ('_kde', 'GMMA.neighbors.kde', 'GMMA.neighbors',
     'KernelDensity'),
    ('_lof', 'GMMA.neighbors.lof', 'GMMA.neighbors',
     'LocalOutlierFactor'),
    ('_nca', 'GMMA.neighbors.nca', 'GMMA.neighbors',
     'NeighborhoodComponentsAnalysis'),
    ('_nearest_centroid', 'GMMA.neighbors.nearest_centroid',
     'GMMA.neighbors', 'NearestCentroid'),
    ('_quad_tree', 'GMMA.neighbors.quad_tree', 'GMMA.neighbors',
     'CELL_DTYPE'),
    ('_regression', 'GMMA.neighbors.regression', 'GMMA.neighbors',
     'KNeighborsRegressor'),
    ('_typedefs', 'GMMA.neighbors.typedefs', 'GMMA.neighbors',
     'DTYPE'),
    ('_unsupervised', 'GMMA.neighbors.unsupervised', 'GMMA.neighbors',
     'NearestNeighbors'),

    ('_isomap', 'GMMA.manifold.isomap', 'GMMA.manifold', 'Isomap'),
    ('_locally_linear', 'GMMA.manifold.locally_linear', 'GMMA.manifold',
     'LocallyLinearEmbedding'),
    ('_mds', 'GMMA.manifold.mds', 'GMMA.manifold', 'MDS'),
    ('_spectral_embedding', 'GMMA.manifold.spectral_embedding_',
     'GMMA.manifold', 'SpectralEmbedding'),
    ('_t_sne', 'GMMA.manifold.t_sne', 'GMMA.manifold', 'TSNE'),

    ('_label_propagation', 'GMMA.semi_supervised.label_propagation',
     'GMMA.semi_supervised', 'LabelPropagation'),

    ('_data', 'GMMA.preprocessing.data', 'GMMA.preprocessing',
     'Binarizer'),
    ('_label', 'GMMA.preprocessing.label', 'GMMA.preprocessing',
     'LabelEncoder'),

    ('_base', 'GMMA.feature_selection.base', 'GMMA.feature_selection',
     'SelectorMixin'),
    ('_from_model', 'GMMA.feature_selection.from_model',
     'GMMA.feature_selection', 'SelectFromModel'),
    ('_mutual_info', 'GMMA.feature_selection.mutual_info',
     'GMMA.feature_selection', 'mutual_info_regression'),
    ('_rfe', 'GMMA.feature_selection.rfe',
     'GMMA.feature_selection.rfe', 'RFE'),
    ('_univariate_selection',
     'GMMA.feature_selection.univariate_selection',
     'GMMA.feature_selection', 'chi2'),
    ('_variance_threshold',
     'GMMA.feature_selection.variance_threshold',
     'GMMA.feature_selection', 'VarianceThreshold'),

    ('_testing', 'GMMA.utils.testing', 'GMMA.utils',
     'all_estimators'),
]


_FILE_CONTENT_TEMPLATE = """
# THIS FILE WAS AUTOMATICALLY GENERATED BY deprecated_modules.py
import sys
# mypy error: Module X has no attribute y (typically for C extensions)
from . import {new_module_name}  # type: ignore
from {relative_dots}externals._pep562 import Pep562
from {relative_dots}utils.deprecation import _raise_dep_warning_if_not_pytest

deprecated_path = '{deprecated_path}'
correct_import_path = '{correct_import_path}'

_raise_dep_warning_if_not_pytest(deprecated_path, correct_import_path)

def __getattr__(name):
    return getattr({new_module_name}, name)

if not sys.version_info >= (3, 7):
    Pep562(__name__)
"""


def _get_deprecated_path(deprecated_path):
    deprecated_parts = deprecated_path.split(".")
    deprecated_parts[-1] = deprecated_parts[-1] + ".py"
    return Path(*deprecated_parts)


def _create_deprecated_modules_files():
    """Add submodules that will be deprecated. A file is created based
    on the deprecated submodule's name. When this submodule is imported a
    deprecation warning will be raised.
    """
    for (new_module_name, deprecated_path,
         correct_import_path, _) in _DEPRECATED_MODULES:
        relative_dots = deprecated_path.count(".") * "."
        deprecated_content = _FILE_CONTENT_TEMPLATE.format(
            new_module_name=new_module_name,
            relative_dots=relative_dots,
            deprecated_path=deprecated_path,
            correct_import_path=correct_import_path)

        with _get_deprecated_path(deprecated_path).open('w') as f:
            f.write(deprecated_content)


def _clean_deprecated_modules_files():
    """Removes submodules created by _create_deprecated_modules_files."""
    for _, deprecated_path, _, _ in _DEPRECATED_MODULES:
        with suppress(FileNotFoundError):
            _get_deprecated_path(deprecated_path).unlink()


if __name__ == "__main__":
    _clean_deprecated_modules_files()
