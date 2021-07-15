"""
Connect FERC1 Steam table to EIA's master unit list via record linkage.

FERC plant records are reported... kind of messily. In the same table there are
records that are reported as whole plants, generators, collections of prime
movers. So we have this heterogeneously reported collection of parts of plants
in FERC1.

EIA on the other hand is reported in a much cleaner way. The are generators
with ids and plants with ids reported in *seperate* tables. What a joy. In
`make_plant_parts_eia`, we've generated the "master unit list". The master unit
list (often referred to as `plant_parts_df` in this module) generated records
for various levels or granularies of plant parts.

For each of the FERC1 steam records we want to figure out if which master unit
list record is the corresponding record. We do this with a record linkage/
scikitlearn machine learning model. The recordlinkage package helps us create
feature vectors (via `make_features`) for each candidate match between FERC
and EIA. Feature vectors are a number between 0 and 1 that indicates the
closeness for each value we want to compare.

We use the feature vectors of our known-to-be-connected training data to cross
validate and tune parameters to choose hyperparameters for scikitlearn models
(via `test_model_parameters`). We choose the "best" model based on the cross
validated results. This best scikitlearn model is then used to generate matches
with the full dataset (`fit_predict_lrc`). The model can return multiple
options for each FERC1 record, so we rank them and choose the best/winning
match (`calc_wins`). We then ensure those connections cointain our training
data (`override_winners_with_training_df`). These "best" results are the
connections we keep as the matches between FERC1 steam records and the EIA
master unit list.
"""

import logging
import statistics
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import recordlinkage as rl
import scipy
from recordlinkage.compare import Exact, Numeric, String  # , Date
from sklearn.model_selection import KFold  # , cross_val_score

import pudl
from pudl_rmi import make_plant_parts_eia

logger = logging.getLogger(__name__)


def main(file_path_training, file_path_mul, pudl_out):
    """
    Coordinate the connection between FERC1 steam and EIA master unit list.

    Note: idk if this will end up as a script or what, but I wanted a place to
    coordinate the connection. May be temporary.
    """
    inputs = InputManager(file_path_training, file_path_mul, pudl_out)
    features_all = (Features(feature_type='all', inputs=inputs)
                    .get_features(clobber=False))
    features_train = (Features(feature_type='training', inputs=inputs)
                      .get_features(clobber=False))
    tuner = ModelTuner(features_train, inputs.get_train_index(), n_splits=10)

    matcher = MatchManager(best=tuner.get_best_fit_model(), inputs=inputs)
    matches_best = matcher.get_best_matches(features_train, features_all)
    connects_ferc1_eia = prettyify_best_matches(
        matches_best,
        plant_parts_true_df=inputs.plant_parts_true_df,
        steam_df=inputs.steam_df
    )
    return connects_ferc1_eia


class InputManager:
    """Class prepare inputs for linking FERC1 and EIA."""

    def __init__(self, file_path_training, file_path_mul, pudl_out):
        """
        Initialize inputs manager that gets inputs for linking FERC and EIA.

        Args:
            file_path_training (path-like): path to the CSV of training data.
                The training data needs to have at least two columns:
                record_id_eia record_id_ferc1.
            file_path_mul (pathlib.Path): path to EIA's the master unit list.
            pudl_out (object): instance of `pudl.output.pudltabl.PudlTabl()`.
        """
        self.file_path_mul = file_path_mul
        self.file_path_training = file_path_training
        self.pudl_out = pudl_out

        # generate empty versions of the inputs.. this let's this class check
        # whether or not the compiled inputs exist before compilnig
        self.plant_parts_df = None
        self.plant_parts_true_df = None
        self.steam_df = None
        self.all_plants_ferc1_df = None
        self.train_df = None
        self.train_index = None
        self.plant_parts_train_df = None
        self.steam_train_df = None

    def get_plant_parts_full(self, clobber=False):
        """Get the full master unit list."""
        if clobber or self.plant_parts_df is None:
            self.plant_parts_df = (
                make_plant_parts_eia.get_master_unit_list_eia(
                    self.file_path_mul)
                .assign(plant_id_report_year_util_id=lambda x:
                        x.plant_id_report_year + "_" +
                        x.utility_id_pudl.map(str))
            )
        return self.plant_parts_df

    def get_plant_parts_true(self, clobber=False):
        """Get the master unit list."""
        # We want only the records of the master unit list that are "true
        # granularies" and those which are not duplicates based on their
        # ownership  so the model doesn't get confused as to which option to
        # pick if there are many records with duplicate data
        if clobber or self.plant_parts_true_df is None:
            plant_parts_df = self.get_plant_parts_full()
            self.plant_parts_true_df = (
                plant_parts_df[(plant_parts_df['true_gran'])
                               & (~plant_parts_df['ownership_dupe'])
                               ]
            )
        return self.plant_parts_true_df

    def prep_train_connections(self, clobber=False):
        """
        Get and prepare the training connections.

        We have stored training data, which consists of records with ids
        columns for both FERC and EIA. Those id columns serve as a connection
        between ferc1 steam and the eia master unit list. These connections
        indicate that a ferc1 steam record is reported at the same granularity
        as the connected master unit list record. These records to train a
        machine learning model.

        Returns:
            pandas.DataFrame: training connections. A dataframe with has a
            MultiIndex with record_id_eia and record_id_ferc1.
        """
        if clobber or self.train_df is None:
            mul_cols = ['true_gran', 'appro_part_label',
                        'appro_record_id_eia', 'plant_part', 'ownership_dupe']
            self.train_df = (
                # we want to ensure that the records are associated with a
                # "true granularity" - which is a way we filter out whether or
                # not each record in the master unit list is actually a
                # new/unique collection of plant parts
                # once the true_gran is dealt with, we also need to convert the
                # records which are ownership dupes to reflect their "total"
                # ownership counterparts
                pd.read_csv(self.file_path_training,)
                .merge(
                    self.get_plant_parts_full().reset_index()
                    [['record_id_eia'] + mul_cols],
                    how='left', on=['record_id_eia'],
                )
                .assign(plant_part=lambda x: x['appro_part_label'],
                        record_id_eia=lambda x: x['appro_record_id_eia'])
                .pipe(make_plant_parts_eia.reassign_id_ownership_dupes)
                .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
                .replace(to_replace="nan", value={'record_id_eia': pd.NA, })
                # recordlinkage and sklearn wants MultiIndexs to do the stuff
                .set_index(['record_id_ferc1', 'record_id_eia', ])
                .drop(columns=mul_cols)
            )
        return self.train_df

    def get_train_index(self):
        """Get the index for the training data."""
        self.train_index = self.prep_train_connections().index
        return self.train_index

    def get_all_ferc1(self, clobber=False):
        """
        Prepare FERC1 steam data for record linkage with EIA master unit list.

        This method grabs two tables from `pudl_out` (`plants_steam_ferc1`
        and `fuel_by_plant_ferc1`) and ensures that the columns the same as
        their EIA counterparts, because the output of this method will be used
        to link FERC and EIA.

        Returns:
            pandas.DataFrame: a cleaned table of FERC1 steam plant records with
            fuel cost data.

        """
        if clobber or self.all_plants_ferc1_df is None:

            # fpb_cols_to_use = [
            #     'report_year', 'utility_id_ferc1', 'plant_name_ferc1',
            #     'utility_id_pudl', 'fuel_cost', 'fuel_mmbtu',
            #     'primary_fuel_by_mmbtu']

            logger.info("Preparing the FERC1 tables.")
            self.all_plants_ferc1_df = (
                self.pudl_out.all_plants_ferc1()
                .assign(
                    plant_id_report_year=lambda x: (
                        x.plant_id_pudl.map(str) + "_" + x.report_year.map(str)),
                    plant_id_report_year_util_id=lambda x: (
                        x.plant_id_report_year + "_" + x.utility_id_pudl.map(str)))
                .set_index('record_id_ferc1'))

            # self.steam_df = (
            #     pd.merge(
            #         self.pudl_out.plants_steam_ferc1(),
            #         self.pudl_out.fbp_ferc1()[fpb_cols_to_use],
            #         on=['report_year',
            #             'utility_id_ferc1',
            #             'utility_id_pudl',
            #             'plant_name_ferc1',
            #             ],
            #         how='left')
            #     .pipe(pudl.helpers.convert_cols_dtypes,
            #           'ferc1', 'ferc1 plant records')
            #     # we want the column names to conform to EIA's column names
            #     .rename(columns={
            #         'fuel_cost': 'total_fuel_cost',
            #         'fuel_mmbtu': 'total_mmbtu',
            #         'opex_fuel_per_mwh': 'fuel_cost_per_mwh',
            #         'primary_fuel_by_mmbtu': 'fuel_type_code_pudl',
            #         'record_id': 'record_id_ferc1', })
            #     .set_index('record_id_ferc1')
            #     .assign(
            #         fuel_cost_per_mmbtu=lambda x: (
            #             x.total_fuel_cost / x.total_mmbtu),
            #         heat_rate_mmbtu_mwh=lambda x: (
            #             x.total_mmbtu / x.net_generation_mwh),
            #         plant_id_report_year=lambda x: x.plant_id_pudl.map(
            #             str) + "_" + x.report_year.map(str),
            #         plant_id_report_year_util_id=lambda x:
            #             x.plant_id_report_year + "_" + \
            #         x.utility_id_pudl.map(str)
            #     ))
            # if 0.9 > (len(self.steam_df) /
            #           len(self.steam_df.drop_duplicates(
            #               subset=['report_year',
            #                       'utility_id_pudl',
            #                       'plant_id_ferc1'])) < 1.1):
            #     raise AssertionError(
            #         'Merge issue w/ pudl_out.plants_steam_ferc1 and fbp_ferc1')

        return self.all_plants_ferc1_df  # self.steam_df

    def get_train_records(self, dataset_df, dataset_id_col):
        """
        Generate a set of known connections from a dataset using training data.

        This method grabs only the records from the the datasets (EIA or FERC)
        that we have in our training data.

        Args:
            dataset_df (pandas.DataFrame): either FERC1 steam table (result of
                `get_all_ferc1()`) or EIA master unit list (result of
                `get_master_unit_list_eia()`).
            dataset_id_col (string): either `record_id_eia` for
                plant_parts_true_df or `record_id_ferc1` for steam_df.

        """
        known_df = (
            pd.merge(dataset_df,
                     self.prep_train_connections().reset_index()[
                         [dataset_id_col]],
                     left_index=True,
                     right_on=[dataset_id_col]
                     )
            .drop_duplicates(subset=[dataset_id_col])
            .set_index(dataset_id_col)
            .astype({'total_fuel_cost': float,
                     'total_mmbtu': float})
        )
        return known_df

    # Note: Is there a way to avoid these little shell methods? I need a
    # standard way to access
    def get_train_eia(self, clobber=False):
        """Get the known training data from EIA."""
        if clobber or self.plant_parts_train_df is None:
            self.plant_parts_train_df = self.get_train_records(
                self.get_plant_parts_true(),
                dataset_id_col='record_id_eia')
        return self.plant_parts_train_df

    def get_train_ferc1(self, clobber=False):
        """Get the known training data from FERC1."""
        if clobber or self.steam_train_df is None:
            self.steam_train_df = self.get_train_records(
                self.get_all_ferc1(),
                dataset_id_col='record_id_ferc1')
        return self.steam_train_df

    def execute(self, clobber=False):
        """Compile all the inputs."""
        # grab the main two data tables we are trying to connect
        self.plant_parts_true_df = self.get_plant_parts_true(clobber=clobber)
        self.steam_df = self.get_all_ferc1(clobber=clobber)

        # we want both the df version and just the index; skl uses just the
        # index and we use the df in merges and such
        self.train_df = self.prep_train_connections(clobber=clobber)
        self.train_index = self.get_train_index()

        # generate the list of the records in the EIA and FERC records that
        # exist in the training data
        self.plant_parts_train_df = self.get_train_eia(clobber=clobber)
        self.steam_train_df = self.get_train_ferc1(clobber=clobber)
        return


class Features:
    """Generate featrue vectors for connecting FERC and EIA."""

    def __init__(self, feature_type, inputs):
        """
        Initialize feature generator.

        Args:
            feature_type (string): either 'training' or 'all'. Type of features
                to compile.
            inputs (instance of class): instance of ``Inputs``

        """
        self.inputs = inputs
        self.features_df = None

        if feature_type not in ['all', 'training']:
            raise ValueError(
                f"feature_type {feature_type} not allowable. Must be either "
                "'all' or 'training'")
        self.feature_type = feature_type
        # the input_dict is going to help in standardizing how we generate
        # features. Based on the feature_type (keys), the latter methods will
        # know which dataframes to use as inputs for ``make_features()``
        self.input_dict = {
            'all': {
                'ferc1_df': self.inputs.get_all_ferc1,
                'eia_df': self.inputs.get_plant_parts_true,
            },
            'training': {
                'ferc1_df': self.inputs.get_train_ferc1,
                'eia_df': self.inputs.get_train_eia,
            },
        }

    def make_features(self, ferc1_df, eia_df, block_col=None):
        """
        Generate comparison features based on defined features.

        The recordlinkage package helps us create feature vectors.
        For each column that we have in both datasets, this method generates
        a column of feature vecotrs, which contain values between 0 and 1 that
        are measures of the similarity between each datapoint the two datasets
        (1 meaning the two datapoints were exactly the same and 0 meaning they
        were not similar at all).

        For more details see recordlinkage's documentaion:
        https://recordlinkage.readthedocs.io/en/latest/ref-compare.html

        Args:
            ferc1_df (pandas.DataFrame): Either training or all records from
                steam table (`steam_train_df` or `steam_df`).
            eia_df (pandas.DataFrame): Either training or all records from the
                EIA master unit list (`plant_parts_train_df` or
                `plant_parts_true_df`).
            block_col (string):  If you want to restrict possible matches
                between ferc_df and eia_df based on a particular column,
                block_col is the column name of blocking column. Default is
                None. If None, this method will generate features between all
                possible matches.

        Returns:
            pandas.DataFrame: a dataframe of feature vectors between FERC and
            EIA.

        """
        compare_cl = rl.Compare(features=[
            String('plant_name_ferc1', 'plant_name_new',
                   label='plant_name', method='jarowinkler'),
            Numeric('net_generation_mwh', 'net_generation_mwh',
                    label='net_generation_mwh', method='exp', scale=1000),
            Numeric('capacity_mw', 'capacity_mw',
                    label='capacity_mw', method='exp', scale=10),
            Numeric('total_fuel_cost', 'total_fuel_cost',
                    label='total_fuel_cost', method='exp', offset=2500,
                    scale=10000, missing_value=0.5),
            Numeric('total_mmbtu', 'total_mmbtu', label='total_mmbtu',
                    method='exp', offset=1, scale=100, missing_value=0.5),
            Numeric('capacity_factor', 'capacity_factor',
                    label='capacity_factor'),
            Numeric('fuel_cost_per_mmbtu', 'fuel_cost_per_mmbtu',
                    label='fuel_cost_per_mmbtu'),
            Numeric('heat_rate_mmbtu_mwh', 'heat_rate_mmbtu_mwh',
                    label='heat_rate_mmbtu_mwh'),
            Exact('fuel_type_code_pudl', 'fuel_type_code_pudl',
                  label='fuel_type_code_pudl'),
            Exact('installation_year', 'installation_year',
                  label='installation_year'),
            # Exact('utility_id_pudl', 'utility_id_pudl',
            #      label='utility_id_pudl'),
        ])

        # generate the index of all candidate features
        indexer = rl.Index()
        indexer.block(block_col)
        feature_index = indexer.index(ferc1_df, eia_df)

        features = compare_cl.compute(feature_index, ferc1_df, eia_df)
        return features

    def get_features(self, clobber=False):
        """Get the feature vectors for the training matches."""
        # generate feature matrixes for known/training data
        if clobber or self.features_df is None:
            self.features_df = self.make_features(
                ferc1_df=self.input_dict[self.feature_type]['ferc1_df'](),
                eia_df=self.input_dict[self.feature_type]['eia_df'](),
                block_col='plant_id_report_year_util_id')
            logger.info(
                f"Generated {len(self.features_df)} {self.feature_type} "
                "candidate features.")
        return self.features_df


class ModelTuner:
    """A class for tuning scikitlearn model."""

    def __init__(self, features_train, train_index, n_splits=10):
        """
        Initialize model tuner; test hyperparameters with cross validation.

        Initializing this object runs `test_model_parameters()` which runs
        through many options for model hyperparameters and collects scores
        from those model runs.

        Args:
            file_path_training (path-like): path to the CSV of training data.
                The training data needs to have at least two columns:
                record_id_eia record_id_ferc1.
            file_path_mul (pathlib.Path): path to EIA's the master unit list.
            pudl_out (object): instance of `pudl.output.pudltabl.PudlTabl()`.

        """
        self.features_train = features_train
        self.train_index = train_index
        self.n_splits = n_splits
        self.results_options = None
        self.best = None

    @staticmethod
    def kfold_cross_val(n_splits, features_known, known_index, lrc):
        """
        K-fold cross validation for model.

        Args:
            n_splits (int): the number of splits for the cross validation.
                If 5, the known data will be spilt 5 times into testing and
                training sets for example.
            features_known (pandas.DataFrame): a dataframe of comparison
                features. This should be created via `make_features`. This
                will contain all possible combinations of matches between
                your records.
            known_index (pandas.MultiIndex): an index with the known
                matches. The index must be a mutltiindex with record ids
                from both sets of records.

        """
        kf = KFold(n_splits=n_splits)
        fscore = []
        precision = []
        accuracy = []
        result_lrc_complied = pd.DataFrame()
        for train_index, test_index in kf.split(features_known):
            x_train = features_known.iloc[train_index]
            x_test = features_known.iloc[test_index]
            y_train = x_train.index & known_index
            y_test = x_test.index & known_index
            # Train the classifier
            lrc.fit(x_train, y_train)
            # predict matches for the test
            result_lrc = lrc.predict(x_test)
            # generate and compile the scores and outcomes of the
            # prediction
            fscore.append(rl.fscore(y_test, links_pred=result_lrc))
            precision.append(rl.precision(y_test, links_pred=result_lrc))
            accuracy.append(rl.accuracy(
                y_test, links_pred=result_lrc, total=result_lrc))
            result_lrc_complied = result_lrc_complied.append(
                pd.DataFrame(index=result_lrc))
        return result_lrc_complied, fscore, precision, accuracy

    def fit_predict_option(self, solver, c, cw, p, l1, n_splits,
                           multi_class, results_options):
        """
        Test and cross validate with a set of model parameters.

        In this method, we instantiate a model object with a given set of
        hyperparameters (which are selected within `test_model_parameters`)
        and then run k-fold cross vaidation with that model and our training
        data.

        Returns:
            pandas.DataFrame
        """
        logger.debug(f'train: {solver}: c-{c}, cw-{cw}, p-{p}, l1-{l1}')
        lrc = rl.LogisticRegressionClassifier(solver=solver,
                                              C=c,
                                              class_weight=cw,
                                              penalty=p,
                                              l1_ratio=l1,
                                              random_state=0,
                                              multi_class=multi_class,
                                              )
        results, fscore, precision, accuracy = self.kfold_cross_val(
            lrc=lrc,
            n_splits=n_splits,
            features_known=self.features_train,
            known_index=self.train_index)

        # we're going to want to choose the best model so we need to save the
        # results of this model run...
        results_options = results_options.append(pd.DataFrame(
            data={
                # result scores
                'precision': [statistics.mean(precision)],
                'f_score': [statistics.mean(fscore)],
                'accuracy': [statistics.mean(accuracy)],
                # info about results
                'coef': [lrc.coefficients],
                'interc': [lrc.intercept],
                'predictions': [len(results)],
                # info about which model hyperparameters we choose
                'solver': [solver],
                'c': [c],
                'cw': [cw],
                'penalty': [p],
                'l1': [l1],
                'multi_class': [multi_class],
            },
        ))
        return results_options

    @staticmethod
    def get_hyperparameters_options():
        """
        Generate a dictionary with sets of options for model hyperparameters.

        Note: The looping over all of the hyperparameters options here feels..
        messy. I investigated scikitlearn's documentaion for a cleaner way to
        do this. I came up empty handed, but I'm still sure I just missed it.

        Returns:
            dictionary: dictionary with autogenerated integers (keys) for each
            dictionary of model
        """
        # we are going to loop through the options for logistic regression
        # hyperparameters
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        cs = [1, 10, 100, 1000]
        cws = ['balanced', None]
        ps = {'newton-cg': ['l2', 'none'],
              'lbfgs': ['l2', 'none'],
              'liblinear': ['l1', 'l2'],
              'sag': ['l2', 'none'],
              'saga': ['l1', 'l2', 'elasticnet', 'none'],
              }
        hyper_options = []
        # we set l1_ratios and multi_classes inside this loop land bc
        for solver in solvers:
            for c in cs:
                for cw in cws:
                    for p in ps[solver]:
                        if p == 'elasticnet':
                            l1_ratios = [.1, .3, .5, .7, .9]
                        else:
                            l1_ratios = [None]
                        for l1 in l1_ratios:
                            # liblinear solver doesnt allow multinomial
                            # multi_class
                            if solver == 'liblinear':
                                multi_classes = ['auto', 'ovr']
                            else:
                                multi_classes = [
                                    'auto', 'ovr', 'multinomial']
                            for multi_class in multi_classes:
                                hyper_options.append({
                                    'solver': solver,
                                    'c': c,
                                    'cw': cw,
                                    'penalty': p,
                                    'l1': l1,
                                    'multi_class': multi_class,
                                })
        return hyper_options

    def test_model_parameters(self, clobber=False):
        """
        Test and corss validate model parameters.

        The method runs `fit_predict_option()` on many options for model
        hyperparameters and saves info about the results for each model run so
        we can later determine which set of hyperparameters works best on
        predicting our training data.

        Args:
            n_splits (int): the number of times we want to split the training
                data during the k-fold cross validation.
        Returns:
            pandas.DataFrame: dataframe in which each record correspondings to
            one model run and contains info like scores of the run (how well
            it predicted our training data).

        """
        if clobber or self.results_options is None:
            logger.info(
                "We are about to test hyper parameters of the model while "
                "doing k-fold cross validation. This takes a few minutes....")
            # it is testing an array of model hyper parameters and
            # cross-vaildating with the training data. It returns a df with
            # losts of result scores to be used to find the best resutls
            hyper_options = self.get_hyperparameters_options()
            # make an empty df to save the results into
            self.results_options = pd.DataFrame()
            for hyper in hyper_options:
                self.results_options = self.fit_predict_option(
                    solver=hyper['solver'], c=hyper['c'], cw=hyper['cw'],
                    p=hyper['penalty'], l1=hyper['l1'],
                    multi_class=hyper['multi_class'],
                    n_splits=self.n_splits,
                    results_options=self.results_options)
        return self.results_options

    def get_best_fit_model(self, clobber=False):
        """Get the best fitting model hyperparameters."""
        if clobber or self.best is None:
            # grab the highest scoring model...the f_score is most encompassing
            # score so we'll lead with that f_score
            self.best = (self.test_model_parameters().sort_values(
                ['f_score', 'precision', 'accuracy'], ascending=False).head(1))
            logger.info("Scores from the best model hyperparameters:")
            logger.info(f"  F-Score:   {self.best.loc[0,'f_score']:.02}")
            logger.info(f"  Precision: {self.best.loc[0,'precision']:.02}")
            logger.info(f"  Accuracy:  {self.best.loc[0,'accuracy']:.02}")
        return self.best


class MatchManager:
    """Manages the results of ModelTuner and chooses best matches."""

    def __init__(self, best, inputs):
        """
        Initialize match manager.

        Args:
            best (pandas.DataFrame): one row table with details about the
                hyperparameters of best model option. Result of
                ``ModelTuner.get_best_fit_model()``.
            inputs (instance of ``InputManager``): instance of ``InputManager``
        """
        self.best = best
        self.train_df = inputs.prep_train_connections()
        # get the # of ferc options within the available eia years.
        self.ferc1_options_len = len(
            inputs.get_all_ferc1()[
                inputs.get_all_ferc1().report_year.isin(
                    inputs.get_plant_parts_true().report_date.dt.year.unique())
            ]
        )

    def _apply_weights(self, features, coefs):
        """
        Apply coefficient weights to each feature.

        Args:
            features (pandas.DataFrame): a dataframe containing features of
                candidate or model matches. The order of the columns
                matters! They must be in the same order as they were fed
                into the model that produced the coefficients.
            coefs (array): array of integers with the same length as the
                columns in features.

        """
        if len(coefs) != len(features.columns):
            raise AssertionError(
                """The number of coeficients (the weight of the importance of the
            columns) should be the same as the number of the columns in the
            candiate matches coefficients.""")
        for coef_n in np.array(range(len(coefs))):
            features[features.columns[coef_n]] = \
                features[features.columns[coef_n]].multiply(coefs[coef_n])
        return features

    def weight_features(self, features):
        """
        Weight features of candidate (or model) matches with coefficients.

        Args:
            features (pandas.DataFrame): a dataframe containing features of
                candidate or model matches. The order of the columns
                matters! They must be in the same order as they were fed
                into the model that produced the coefficients.
            coefs (array): array of integers with the same length as the
                columns in features.

        """
        df = deepcopy(features)
        return (df.
                pipe(self._apply_weights, self.get_coefs()).
                assign(score=lambda x: x.sum(axis=1)).
                pipe(pudl.helpers.organize_cols, ['score']).
                sort_values(['score'], ascending=False).
                sort_index(level='record_id_ferc1'))

    def calc_match_stats(self, df):
        """
        Calculate stats needed to judge candidate matches.

        rank: diffs: iqr:

        Args:
            df (pandas.DataFrame): Dataframe of comparison features with
                MultiIndex containing the ferc and eia record ids.

        Returns
            pandas.DataFrame: the input df with the stats.

        """
        df = self.weight_features(df).reset_index()
        gb = df.groupby('record_id_ferc1')[['record_id_ferc1', 'score']]
        df = (
            df.sort_values(['record_id_ferc1', 'score'])
            # rank the scores
            .assign(rank=gb.rank(ascending=0, method='average'),
                    # calculate differences between scores
                    diffs=lambda x: x['score'].diff())
            # count grouped records
            .merge(pudl.helpers.count_records(df, ['record_id_ferc1'],
                                              'count'),
                   how='left',)
            # calculate the iqr for each record_id_ferc1 group
            .merge((gb.agg(scipy.stats.iqr)
                    # .droplevel(0, axis=1)
                    .rename(columns={'score': 'iqr'})),
                   left_on=['record_id_ferc1'],
                   right_index=True))

        # assign the first diff of each ferc_id as a nan
        df['diffs'][df.record_id_ferc1 !=
                    df.record_id_ferc1.shift(1)] = np.nan

        df = df.set_index(['record_id_ferc1', 'record_id_eia'])
        return df

    def calc_murk(self, df, iqr_perc_diff):
        """Calculate the murky model matches."""
        distinction = (df['iqr_all'] * iqr_perc_diff)
        matches_murk = (df[(df['rank'] == 1) &
                           (df['diffs'] < distinction)])
        return matches_murk

    def calc_best_matches(self, df, iqr_perc_diff):
        """
        Find the highest scoring matches and report on match coverage.

        With the matches resulting from a model run, generate "best" matches by
        finding the highest ranking EIA match for each FERC record. If it is
        either the only match or it is different enough from the #2 ranked
        match, we consider it a winner. Also log stats about the coverage of
        the best matches.

        The matches are all of the results from the model prediction. The
        best matches are all of the matches that are distinct enough from it’s
        next closest match. The `murky_df` are the matches that are not
        “distinct enough” from the closes match. Distinct enough means that
        the best match isn’t one iqr away from the second best match.

        Args:
            df (pandas.DataFrame): dataframe with all of the model generate
                matches. This df needs to have been run through
                `calc_match_stats()`.
            iqr_perc_diff (float):

        Returns
            pandas.DataFrame : winning matches. Matches that had the
            highest rank in their record_id_ferc1, by a wide enough margin.

        """
        unique_f = df.reset_index().drop_duplicates(
            subset=['record_id_ferc1'])
        distinction = (df['iqr_all'] * iqr_perc_diff)
        # for the best matches, grab the top ranked model match if there is a
        # big enough difference between it and the next highest ranked match
        # diffs is a measure of the difference between each record and the next
        # highest ranked model match
        # the other option here is if a model match is the highest rank and
        # there there is no other model matches
        best_match = (df[((df['rank'] == 1) & (df['diffs'] > distinction))
                         | ((df['rank'] == 1) & (df['diffs'].isnull()))])
        # we want to know how many of the
        self.murk_df = self.calc_murk(df, iqr_perc_diff)
        self.ties_df = df[df['rank'] == 1.5]

        logger.info(
            f"""Winning match stats:
        matches vs ferc:      {len(unique_f)/self.ferc1_options_len:.02%}
        best match v ferc:    {len(best_match)/self.ferc1_options_len:.02%}
        best match vs matches:{len(best_match)/len(unique_f):.02%}
        murk vs matches:      {len(self.murk_df)/len(unique_f):.02%}
        ties vs matches:      {len(self.ties_df)/2/len(unique_f):.02%}"""
        )
        return best_match

    def override_best_match_with_training_df(self, matches_best_df, train_df):
        """
        Override winning matches with training data matches.

        We want to ensure that all of the matches that we put in the
        training data for the record linkage model actually end up in the
        resutls from the record linkage model.

        Args:
            matches_best_df (pandas.DataFrame): best matches generated via
                `calc_best_matches()`. Matches that had the highest rank in
                their record_id_ferc1, by a wide enough margin.
            train_df (pandas.DataFrame): training data/known matches
                between ferc and the master unit list. Result of
                `prep_train_connections()`.

        Returns:
            pandas.DataFrame: overridden winning matches. Matches that show
            up in the training data `train_df` or if there was no
            corresponding training data, matches that had the highest rank
            in their record_id_ferc1, by a wide enough margin.
        """
        # we want to override the eia when the training id is
        # different than the "winning" match from the recrod linkage
        matches_best_df = (
            pd.merge(
                matches_best_df.reset_index(),
                train_df.reset_index().dropna(),
                on=['record_id_ferc1'],
                how='outer',
                suffixes=('_rl', '_trn'))
            .assign(
                record_id_eia=lambda x: np.where(
                    x.record_id_eia_trn.notnull(),
                    x.record_id_eia_trn,
                    x.record_id_eia_rl)
            )
        )
        # check how many records were overridden
        overridden = matches_best_df.loc[
            (matches_best_df.record_id_eia_trn.notnull())
            & (matches_best_df.record_id_eia_rl.notnull())
            & (matches_best_df.record_id_eia_trn !=
               matches_best_df.record_id_eia_rl)
        ]
        logger.info(
            f"Overridden records:       {len(overridden)/len(train_df):.01%}\n"
            "New best match v ferc:    "
            f"{len(matches_best_df)/self.ferc1_options_len:.02%}"
        )
        # we don't need these cols anymore...
        matches_best_df = matches_best_df.drop(
            columns=['record_id_eia_trn', 'record_id_eia_rl'])
        return matches_best_df

    @staticmethod
    def fit_predict_lrc(best,
                        features_known,
                        features_all,
                        train_df_ids):
        """Generate, fit and predict model. Wahoo."""
        # prep the model with the hyperparameters
        lrc = rl.LogisticRegressionClassifier(
            solver=best['solver'].values[0],
            C=best['c'].values[0],
            class_weight=best['cw'].values[0],
            penalty=best['penalty'].values[0],
            l1_ratio=best['l1'].values[0],
            random_state=0,
            multi_class=best['multi_class'].values[0])
        # fit the model with all of the
        lrc.fit(features_known, train_df_ids.index)
        # this step is getting preditions on all of the possible matches based
        # on the last run model above
        predict_all = lrc.predict(features_all)
        predict_all_df = pd.merge(pd.DataFrame(index=predict_all),
                                  features_all,
                                  left_index=True,
                                  right_index=True,
                                  how='left')
        return predict_all_df

    def get_coefs(self):
        """
        Get the best models coeficients.

        The coeficients are the measure of relative importance that the model
        determined that each feature vector.
        """
        self.coefs = self.best.loc[0, "coef"]
        return self.coefs

    def get_best_matches(self, features_train, features_all):
        """
        Run logistic regression model and choose highest scoring matches.

        From `TuneModel.test_model_parameters()`, we get a dataframe in which
        each record correspondings to one model run and contains info like
        scores of the run (how well it predicted our training data). This
        method takes the result from `TuneModel.test_model_parameters()` and
        choose the model hyperparameters that scored the highest.

        The logistic regression model this method employs returns all matches
        from the candiate matches in `features_all`. But we only want one match
        for each FERC1 Steam record. So this method uses the coeficients from
        the model (which are a measure of the importance of each of the
        features/columns in `features_all`) so weight the feature vecotrs. With
        the sum of the weighted feature vectors for each model match, this
        method selects the hightest scoring match via `calc_best_matches()`.

        Args:
            features_train (pandas.DataFrame): feature vectors between training
                data from FERC steam and EIA master unit list. Result of
                ``Features.make_features()``.
            features_all (pandas.DataFrame): feature vectors between all data
                from FERC steam and EIA master unit list. Result of
                ``Features.make_features()``.

        Returns:
            pandas.DataFrame: the best matches between ferc1 steam records and
            the EIA master unit list. Each ferc1 steam record has a maximum of
            one best match. The dataframe has a MultiIndex with `record_id_eia`
            and `record_id_ferc1`.
        """
        # actually run a model using the "best" model!!
        logger.info(
            "Fit and predict a model w/ the highest scoring hyperparameters.")
        # this returns all matches that the model deems good enough from the
        # candidate matches in the `features_all`
        matches_model = self.fit_predict_lrc(
            self.best, features_train, features_all, self.train_df)
        # weight the features of model matches with the coeficients
        # we need a metric of how different each match is
        # merge in the IRQ of the full options
        self.matches_model = pd.merge(
            self.calc_match_stats(matches_model),
            self.calc_match_stats(features_all)[['iqr']],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=("", "_all"))
        logger.info("Get the top scoring match for each FERC1 steam record.")
        matches_best_df = (
            self.calc_best_matches(self.matches_model, .02)
            .pipe(self.override_best_match_with_training_df, self.train_df)
        )
        return matches_best_df


def prettyify_best_matches(matches_best, plant_parts_true_df, steam_df,
                           debug=False):
    """
    Make the EIA-FERC best matches usable.

    Use the ID columns from the best matches to merge together both EIA
    master unit list data and FERC steam plant data. This removes the
    comparison vectors (the floats between 0 and 1 that compare the two
    columns from each dataset).
    """
    # if utility_id_pudl is not in the `MUL_COLS`,  we need to in include it
    mul_cols_to_grab = make_plant_parts_eia.MUL_COLS + [
        'plant_id_pudl', 'total_fuel_cost', 'net_generation_mwh', 'capacity_mw'
    ]
    connects_ferc1_eia = (
        # first merge in the EIA Master Unit List
        pd.merge(
            matches_best.reset_index()
            [['record_id_ferc1', 'record_id_eia']],
            # we only want the identifying columns from the MUL
            plant_parts_true_df.reset_index()[mul_cols_to_grab],
            how='left',
            on=['record_id_eia'],
            validate='m:1'  # multiple FERC records can have the same EIA match
        )
        .astype({"report_year": pd.Int64Dtype()})
        # then merge in the FERC data
        # we want the backbone of this table to be the steam records
        # so we have all possible steam records, even the unmapped ones
        .merge(
            steam_df,
            how='outer',
            on=['record_id_ferc1', 'report_year'],
            suffixes=('_eia', '_ferc1'),
            validate='1:1',
            indicator=True
        )
        .assign(
            opex_nonfuel=lambda x: (x.opex_production_total - x.opex_fuel),
            report_date=lambda x: pd.to_datetime(x.report_year, format="%Y")
        )
    )

    no_ferc = connects_ferc1_eia[
        (connects_ferc1_eia._merge == 'left_only')
        & (connects_ferc1_eia.record_id_eia.notnull())
        & ~(connects_ferc1_eia.record_id_ferc1.str.contains('_hydro_'))
        & ~(connects_ferc1_eia.record_id_ferc1.str.contains('_gnrt_plant_'))
    ]
    if not no_ferc.empty:
        message = (
            "Help. \nI'm trapped in this computer and I can't get out.\n"
            ".... jk there shouldn't be any matches between FERC and EIA\n"
            "that have EIA matches but aren't in the Steam table, but we\n"
            f"got {len(no_ferc)}. Check the training data and "
            "prettyify_best_matches()"
        )
        if debug:
            warnings.warn(message)
            return no_ferc
        else:
            raise AssertionError(message)
    _log_match_coverage(connects_ferc1_eia)
    return connects_ferc1_eia


def _log_match_coverage(connects_ferc1_eia):
    eia_years = pudl.constants.working_partitions['eia860']['years']
    # get the matches from just the EIA working years
    m_eia_years = connects_ferc1_eia[
        (connects_ferc1_eia.report_date.dt.year.isin(eia_years))
        & (connects_ferc1_eia.record_id_eia.notnull())]
    # get all records from just the EIA working years
    r_eia_years = connects_ferc1_eia[
        connects_ferc1_eia.report_date.dt.year.isin(eia_years)]

    fuel_type_coverage = (
        len(m_eia_years[m_eia_years.energy_source_code_1.notnull()])
        / len(m_eia_years))
    tech_type_coverage = (
        len(m_eia_years[m_eia_years.technology_description.notnull()])
        / len(m_eia_years))

    def _get_subtable(table_name):
        return r_eia_years[r_eia_years.record_id_ferc1.str.contains(f"{table_name}")]

    def _get_match_pct(df):
        return round((len(df[df['record_id_eia'].notna()]) / len(df) * 100), 1)

    logger.info(
        "Coverage for matches during EIA working years:\n"
        f"    Fuel type: {fuel_type_coverage:.01%}\n"
        f"    Tech type: {tech_type_coverage:.01%}\n\n"
        "Coverage for all steam table records during EIA working years:\n"
        f"    EIA matches: {_get_match_pct(_get_subtable('steam'))}\n\n"
        f"Coverage for all small gen table records during EIA working years:\n"
        f"    EIA matches: {_get_match_pct(_get_subtable('gnrt_plant'))}\n\n"
        f"Coverage for all hydro table records during EIA working years:\n"
        f"    EIA matches: {_get_match_pct(_get_subtable('hydro'))}\n\n"
        f"Coverage for all pumped storage table records during EIA working years:\n"
        f"    EIA matches: {_get_match_pct(_get_subtable('pumped'))}"
    )
