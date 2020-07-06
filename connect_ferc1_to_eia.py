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

import numpy as np
import pandas as pd
import recordlinkage as rl
import scipy
from recordlinkage.compare import Exact, Numeric, String  # , Date
from sklearn.model_selection import KFold  # , cross_val_score

import make_plant_parts_eia
import pudl

logger = logging.getLogger(__name__)


class ConnectFERC1EIA(object):
    """Connector."""

    def __init__(self, file_path_training, file_path_mul, pudl_out):
        """Initialize connector, get and store our inputs."""
        self.file_path_training = file_path_training
        self.pudl_out = pudl_out

        plant_parts_df_full = make_plant_parts_eia.get_master_unit_list_eia(
            self.file_path_mul)
        # we want
        self.plant_parts_df = plant_parts_df_full[
            plant_parts_df_full['true_gran']]
        self.train_df = self.prep_train_connections(
            file_path_training, plant_parts_df_full)
        self.steam_df = self.prep_ferc_data(pudl_out)

    def prep_train_connections(self):
        """
        Get and prepare the training connections.

        We have stored connections between ferc1 steam and the eia master unit
        list. These records should be compiled of connections between these
        data sets that are known to be correct because we will use these
        records to train a machine learning model.

        Args:
            training_file_path (path-like): path to the CSV of training data.
                The training data needs to have at least two columns:
                record_id_eia and record_id_ferc1.
            plant_parts_df (pandas.DataFrame): master unit list. generated from
                `CompilePlantParts.generate_master_unit_list()` or
                `get_master_unit_list_eia()`.

        Returns:
            pandas.DataFrame: training connections. A dataframe with has a
            MultiIndex with record_id_eia and record_id_ferc1.
        """
        mul_cols = ['true_gran', 'appro_part_label',
                    'appro_record_id_eia', 'plant_part']
        self.train_df = (
            # we want to ensure that the records are associated with a
            # "true granularity" - which is a way we filter out whether or not
            # each record in the master unit list is actually a new/unique
            # collection of plant parts
            pd.read_csv(self.file_path_training,)
            .merge(
                self.plant_parts_df.reset_index(
                )[['record_id_eia'] + mul_cols],
                how='left', on=['record_id_eia'],
            )
            .assign(plant_part=lambda x: x['appro_part_label'],
                    record_id_eia=lambda x: x['appro_record_id_eia'])

            # come light cleaning
            .pipe(pudl.helpers.cleanstrings_snake, ['record_id_eia'])
            .replace(to_replace="nan", value={'record_id_eia': pd.NA, })
            # recordlinkage and sklearn wants MultiIndexs to do the stuff
            .set_index(['record_id_ferc1', 'record_id_eia', ])
            .drop(columns=mul_cols)
        )
        return self.train_df

    def prep_ferc_data(self):
        """TODO: clean and condense."""
        cols_to_use = ['report_year',
                       'utility_id_ferc1',
                       'plant_name_ferc1',
                       'utility_id_pudl',
                       'plant_id_pudl',
                       'plant_id_ferc1',
                       'capacity_factor',
                       'capacity_mw',
                       'net_generation_mwh',
                       'opex_fuel',
                       'opex_fuel_per_mwh',
                       'fuel_cost',
                       'fuel_mmbtu',
                       'construction_year',
                       'installation_year',
                       'primary_fuel_by_mmbtu',
                       'plant_type',
                       'record_id',
                       # we don't need this for the record linkage, but we do
                       # want it when we connect depreciation to ferc.
                       'opex_production_total',
                       ]
        fpb_cols_to_use = ['report_year',
                           'utility_id_ferc1',
                           'plant_name_ferc1',
                           'utility_id_pudl',
                           'fuel_cost',
                           'fuel_mmbtu',
                           'primary_fuel_by_mmbtu']

        self.steam_df = (
            self.pudl_out.plants_steam_ferc1().
            merge(
                self.pudl_out.fbp_ferc1()[fpb_cols_to_use],
                on=['report_year',
                    'utility_id_ferc1',
                    'utility_id_pudl',
                    'plant_name_ferc1'
                    ],
                how='left')[cols_to_use]
            .pipe(pudl.helpers.convert_cols_dtypes,
                  'ferc1', 'ferc1 plant records').
            # dropna().
            rename(columns={
                'fuel_cost': 'total_fuel_cost',
                'fuel_mmbtu': 'total_mmbtu',
                'opex_fuel_per_mwh': 'fuel_cost_per_mwh',
                'primary_fuel_by_mmbtu': 'fuel_type_code_pudl',
                'record_id': 'record_id_ferc1', }).
            set_index('record_id_ferc1').
            assign(
                fuel_cost_per_mmbtu=lambda x: (
                    x.total_fuel_cost / x.total_mmbtu),
                heat_rate_mmbtu_mwh=lambda x: (
                    x.total_mmbtu / x.net_generation_mwh),
                plant_id_report_year=lambda x: x.plant_id_pudl.map(
                    str) + "_" + x.report_year.map(str)
            ))
        if 0.9 > (len(self.steam_df) /
                  len(self.steam_df.drop_duplicates(
                      subset=['report_year',
                              'utility_id_pudl',
                              'plant_id_ferc1'])) < 1.1):
            raise AssertionError(
                'Merge issue with pudl_outs plants_steam_ferc1 and fbp_ferc1')

        return self.steam_df

    def make_candidate_links(self, dfa, dfb, block_col=None):
        """Generate canidate links for comparison features."""
        indexer = rl.Index()
        indexer.block(block_col)
        return indexer.index(dfa, dfb)

    def make_features(self, dfa, dfb, block_col=None):
        """Generate comparison features based on defined features."""
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
            Exact('utility_id_pudl', 'utility_id_pudl',
                  label='utility_id_pudl'),
        ])

        features = compare_cl.compute(
            self.make_candidate_links(dfa, dfb, block_col), dfa, dfb)
        return features

    def get_known_connections(self, dataset_df, dataset_id_col):
        """
        Generate a set of known connections from a dataset using training data.

        In our training

        Args:
            dataset_df (pandas.DataFrame)
            dataset_id_col (string): either `record_id_eia` for plant_parts_df
                or `record_id_ferc1` for steam_df.

        """
        known_df = (pd.merge(dataset_df,
                             self.train_df.reset_index()[[dataset_id_col]],
                             left_index=True,
                             right_on=[dataset_id_col]
                             )
                    .drop_duplicates(subset=[dataset_id_col])
                    .set_index(dataset_id_col)
                    .astype({'total_fuel_cost': float,
                             'total_mmbtu': float}))
        return known_df


class MakeConnections(ConnectFERC1EIA):
    """d."""

    def kfold_cross_val(self, n_splits, features_known, known_index, lrc):
        """
        K-fold cross validation for model .

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
                           multi_class, features_known, training_index,
                           results_options):
        """
        Test and cross validate with a set of model parameters.

        In this method, we instantiate a model object with a given set of
        hyperparameters (which are selected within `test_model_parameters`)
        and then run k-fold cross vaidation with that model and our training
        data.
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
            features_known=features_known,
            known_index=training_index)

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

    def test_model_parameters(self,
                              features_known, training_index, n_splits):
        """
        Test and corss validate model parameters.

        The method runs `fit_predict_option` on many options for model
        hyperparameters and saves info about the result options so we can later
        determine which set of hyperparameters works best on predicting our
        training data.

        Args:
            features_known (pandas.DataFrame)
            training_index (pandas.MultiIndex)
            n_splits (int)
        Returns:
            pandas.DataFrame: dataframe in which each record correspondings to
            one model run and contains info like scores of the run (how well
            it predicted our training data), info about the results
            .

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
        # we set l1_ratios and multi_classes inside this loop land bc
        results_options = pd.DataFrame()
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
                                results_options = self.fit_predict_option(
                                    solver=solver, c=c, cw=cw, p=p, l1=l1,
                                    n_splits=n_splits,
                                    multi_class=multi_class,
                                    features_known=features_known,
                                    training_index=training_index,
                                    results_options=results_options)
        return results_options

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
        assert len(coefs) == len(features.columns)
        for coef_n in np.array(range(len(coefs))):
            features[features.columns[coef_n]
                     ] = features[features.columns[coef_n]].multiply(coefs[coef_n])
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
                pipe(self._apply_weights, self.coefs).
                assign(score=lambda x: x.sum(axis=1)).
                pipe(pudl.helpers.organize_cols, ['score']).
                sort_values(['score'], ascending=False).
                sort_index(level='record_id_ferc1'))

    def calc_match_stats(self, df):
        """
        Calculate stats needed to judge candidate matches.

        Args:
            df (pandas.DataFrame): Dataframe of comparison features with
                MultiIndex containing the ferc and eia record ids.

        Returns
            pandas.DataFrame: the input df with the stats.

        """
        df = self.weight_features(df)
        df = df.reset_index()
        gb = df.groupby('record_id_ferc1')[['record_id_ferc1', 'score']]
        df = (
            df.sort_values(['record_id_ferc1', 'score'])
            # rank the scores
            .assign(rank=gb.rank(ascending=0, method='average'))
            # calculate differences between scores
            .assign(diffs=lambda x: x['score'].diff())
            # count grouped records
            .merge(pudl.helpers.count_records(df, ['record_id_ferc1'],
                                              'count'),
                   how='left',)
            # calculate the iqr for each
            .merge((gb.agg({'score': scipy.stats.iqr})
                    .droplevel(0, axis=1)
                    .rename(columns={'score': 'iqr'})),
                   left_on=['record_id_ferc1'],
                   right_index=True))

        # assign the first diff of each ferc_id as a nan
        df['diffs'][df.record_id_ferc1 !=
                    df.record_id_ferc1.shift(1)] = np.nan

        df = df.set_index(['record_id_ferc1', 'record_id_eia'])
        return df

    def calc_murk(self, df, iqr_perc_diff):
        """Calculate the murky wins."""
        distinction = (df['iqr_all'] * iqr_perc_diff)
        murky_wins = (df[(df['rank'] == 1) &
                         (df['diffs'] < distinction)])
        return murky_wins

    def calc_wins(self, df, ferc1_options, iqr_perc_diff):
        """
        Find the winners and report on winning ratios.

        With the matches resulting from a model run, generate "winning"
        matches by finding the highest ranking EIA match for each FERC
        record. If it is either the only match or it is different enough
        from the #2 ranked match, we consider it a winner. Also log win
        stats.

        The matches are all of the results from the model prediction. the
        wins are all of the matches that are distinct enough from it’s
        closest match. The murky_wins are the matches that are not
        “distinct enough” from its closes match. Distinct enough means that
        the top match isn’t one iqr away from the second top match.

        Args:
            df (pandas.DataFrame): dataframe with all of the model generate
                matches. This df needs to have been run through
                `calc_match_stats`.
            ferc1_options (pandas.DataFrame): dataframe with all of the
                possible `record_id_ferc1`s.

        Returns
            pandas.DataFrame : winning matches. Matches that had the
            highest rank in their record_id_ferc1, by a wide enough margin.

        """
        unique_ferc = df.reset_index().drop_duplicates(
            subset=['record_id_ferc1'])
        ties = df[df['rank'] == 1.5]
        distinction = (df['iqr_all'] * iqr_perc_diff)
        # for the winners, grab the top ranked,
        winners = (df[((df['rank'] == 1) & (df['diffs'] > distinction)) |
                      ((df['rank'] == 1) & (df['diffs'].isnull()))])

        murky_wins = self.calc_murk(df, iqr_perc_diff)

        logger.info('Winning match stats:')
        logger.info(
            f'  matches vs total ferc:  {len(unique_ferc)/len(ferc1_options):.02}')
        logger.info(
            f'  wins vs total ferc:     {len(winners)/len(ferc1_options):.02}')
        logger.info(
            f'  wins vs matches:        {len(winners)/len(unique_ferc):.02}')
        logger.info(
            f'  murk vs matches:        {len(murky_wins)/len(unique_ferc):.02}')
        logger.info(
            f'  ties vs matches:        {len(ties)/2/len(unique_ferc):.02}')
        return winners

    def override_winners_with_training_df(self, winners, train_df):
        """
        Override winning matches with training data matches.

        We want to ensure that all of the matches that we put in the
        training data for the record linkage model actually end up in the
        resutls from the record linkage model.

        Args:
            winners (pandas.DataFrame): winning matches generated via
                `calc_wins`. Matches that had the highest rank in their
                record_id_ferc1, by a wide enough margin.
            train_df (pandas.DataFrame): training data/known matches
                between ferc and the master unit list. Result of
                `prep_train_connections`.

        Returns:
            pandas.DataFrame: overridden winning matches. Matches that show
            up in the training data `train_df` or if there was no
            corresponding training data, matches that had the highest rank
            in their record_id_ferc1, by a wide enough margin.
        """
        # we want to override the eia when the training id is
        # different than the "winning" match from the recrod linkage
        winners = (
            pd.merge(
                winners.reset_index(),
                train_df.reset_index(),
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
        overridden = winners.loc[
            (winners.record_id_eia_trn.notnull())
            & (winners.record_id_eia_rl.notnull())
            & (winners.record_id_eia_trn != winners.record_id_eia_rl)
        ]
        logger.info(f"Overridden records: {len(overridden)}")
        # we don't need these cols anymore...
        winners = winners.drop(
            columns=['record_id_eia_trn', 'record_id_eia_rl'])
        return winners

    def fit_predict_lrc(self, best, features_known, features_all, train_df_ids):
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

        # this step is getting preditions on all of the possible matches based on
        # the last run model above
        predict_all = lrc.predict(features_all)
        return (pd.DataFrame(index=predict_all).
                merge(features_all, left_index=True, right_index=True, how='left'))

    def make_connections(self):
        """Make."""
        # generate the list of the records in the EIA and FERC records that
        # exist in the training data
        self.eia_known = self.get_known_connections(
            self.plant_parts_df,
            self.train_df,
            dataset_id_col='record_id_eia')
        self.ferc_known = self.get_known_connections(
            self.steam_df,
            self.train_df,
            dataset_id_col='record_id_ferc1')

        # generate feature matrixes for known/training data
        self.features_train = self.make_features(
            dfa=self.ferc_known,
            dfb=self.eia_known,
            block_col='plant_id_report_year')
        self.features_all = self.make_features(
            dfa=self.steam_df,
            dfb=self.plant_parts_df,
            block_col='plant_id_report_year')

        logger.info(
            """We are about to test hyper parameters of the model while doing
    k-fold cross validation. This takes a few minutes....""")
        # it is testing an array of model hyper parameters and cross-vaildating
        # with the training data. It returns a df with losts of result scores
        # to be used to find the best resutls
        results_options = self.test_model_parameters(
            features_known=self.features_train,
            training_index=self.train_df.index,
            n_splits=10)

        # grab the highest scoring
        self.best = (results_options.sort_values(
            ['f_score', 'precision', 'accuracy'], ascending=False).head(1))
        logger.info("Scores from the best model hyperparameters:")
        logger.info(f"  F-Score:   {self.best.loc[0,'f_score']:.02}")
        logger.info(f"  Precision: {self.best.loc[0,'precision']:.02}")
        logger.info(f"  Accuracy:  {self.best.loc[0,'accuracy']:.02}")

        # actually run a model using the "best" model!!
        prediction_df = self.fit_predict_lrc(
            self.best, self.features_train, self.features_all, self.train_df)

        self.coefs = self.best.loc[0, 'coef']

        # generate results with weighted features
        # we want a metric of how merge in the IRQ of the full options
        self.results = pd.merge(
            self.calc_match_stats(prediction_df),
            self.calc_match_stats(self.features_all)[['iqr']],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=("", "_all"))

        winners = (self.calc_wins(self.results, self.steam_df, .1)
                   .pipe(self.override_winners_with_training_df, self.train_df)
                   )

        return winners
