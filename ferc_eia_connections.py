"""Beginning of compilation of FERC/EIA granular connections."""

import logging
import statistics
from copy import deepcopy

import numpy as np
import pandas as pd
import recordlinkage as rl
import scipy
from recordlinkage.compare import Exact, Numeric, String  # , Date
from sklearn.model_selection import KFold  # , cross_val_score

import pudl

logger = logging.getLogger(__name__)


class TrainXlxsCompiler():
    """Grab the training data excel file."""

    def __init__(self, file_path):
        """Initialize a compiler of training data."""
        self.train_df = None
        self.file_path = file_path

    def _grab_test_xlxs(self):
        """TODO: Add file path."""
        if self.train_df is not None:
            pass
        else:
            logger.info('grabbing xlxs file.')
            self.train_df = pd.read_excel(
                self.file_path,
                skiprows=1,
                dtype={'EIA Plant Code': pd.Int64Dtype(),
                       'Generator': pd.Int64Dtype(),
                       'EIA Utility Code': pd.Int64Dtype(),
                       'report_year': pd.Int64Dtype(),
                       'report_prd': pd.Int64Dtype(),
                       'respondent_id': pd.Int64Dtype(),
                       'spplmnt_num': pd.Int64Dtype(),
                       })
        return self.train_df


def prep_train_connections(compiler_mul, compiler_train):
    """TODO: Clean and condense."""
    # grab the excel file
    test_df = compiler_train._grab_test_xlxs()
    # some things to use for cleaning
    cols_to_rename = {
        'EIA Plant Code': 'plant_id_eia',
        'FERC Line Type': 'plant_part',
        'EIA Utility Code': 'utility_id_eia',
        'Unit Code': 'unit_id_pudl',
        'EIA Technology': 'technology_description',
        'Generator': 'generator_id',
        'EIA Prime Mover': 'prime_mover_code',
        'EIA Energy Source Code': 'energy_source_code_1',
        # use the RMI labels, not our eia_ownership relabel
        'Owned or Total': 'ownership', }
    string_cols = ['FERC Line Type', 'EIA Technology',
                   'EIA Prime Mover', 'EIA Energy Source Code',
                   'Owned or Total']
    plant_part_rename = {'plant_part': {
        'plant': 'plant',
        'generator': 'plant_gen',
        'unit': 'plant_unit',
        'technology': 'plant_technology',
        'plant_prime_fuel': 'plant_prime_fuel',
        'plant_prime': 'plant_prime_mover'}, }

    for col in string_cols:
        if col in test_df.columns:
            test_df.loc[:, col] = (
                test_df[col].astype(str).
                str.strip().
                str.lower().
                str.replace(r'\s+', '_')
            )

    test_df = (test_df.assign(report_date='2018-01-01').
               rename(columns=cols_to_rename).
               astype({'report_date': 'datetime64[ns]',
                       'utility_id_eia': pd.Int64Dtype()}).
               replace(plant_part_rename))

    train_df_ids = compiler_mul.assign_record_id_eia(test_df)
    train_df_ids['plant_part_og'] = train_df_ids['plant_part']
    train_df_ids = (
        train_df_ids.set_index(['record_id_eia']).
        merge(compiler_mul.plant_parts_df[['true_gran', 'appro_part_label',
                                           'appro_record_id_eia']],
              how='left', right_index=True, left_index=True).
        assign(plant_part=lambda x: x['appro_part_label'],
               record_id_eia=lambda x: x['appro_record_id_eia']).
        reset_index(drop=True))
    # train_df_ids = compiler_mul.assign_record_id_eia(train_df_ids)

    train_df_ids['record_id_ferc'] = (
        train_df_ids.Source + '_' +
        train_df_ids.report_year.astype(str) + '_' +
        train_df_ids.report_prd.astype(str) + '_' +
        train_df_ids.respondent_id.astype(str) + '_' +
        train_df_ids.spplmnt_num.astype(str)
    )
    if "row_number" in train_df_ids.columns:
        train_df_ids["record_id_ferc"] = train_df_ids["record_id_ferc"] + \
            "_" + train_df_ids.row_number.astype('Int64').astype(str)

    return train_df_ids.set_index(['record_id_ferc', 'record_id_eia'])


def prep_ferc_data(pudl_out):
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
                   'record_id']
    fpb_cols_to_use = ['report_year',
                       'utility_id_ferc1',
                       'plant_name_ferc1',
                       'utility_id_pudl',
                       'fuel_cost',
                       'fuel_mmbtu',
                       'primary_fuel_by_mmbtu']

    steam = (
        pudl_out.plants_steam_ferc1().
        merge(
            pudl_out.fbp_ferc1()[fpb_cols_to_use],
            on=['report_year',
                'utility_id_ferc1',
                'utility_id_pudl',
                'plant_name_ferc1'
                ],
            how='left')[cols_to_use].
        pipe(pudl.helpers.convert_cols_dtypes,
             'ferc1', 'ferc1 plant records').
        dropna().
        rename(columns={
            'fuel_cost': 'total_fuel_cost',
            'fuel_mmbtu': 'total_mmbtu',
            'opex_fuel_per_mwh': 'fuel_cost_per_mwh',
            'primary_fuel_by_mmbtu': 'fuel_type_code_pudl',
            'record_id': 'record_id_ferc', }).
        set_index('record_id_ferc').
        assign(
            fuel_cost_per_mmbtu=lambda x: x.total_fuel_cost / x.total_mmbtu,
            heat_rate_mmbtu_mwh=lambda x: x.total_mmbtu / x.net_generation_mwh,
            plant_id_report_year=lambda x: x.plant_id_pudl.map(
                str) + "_" + x.report_year.map(str)
        ))
    if 0.9 > (len(steam) /
              len(steam.drop_duplicates(subset=['report_year',
                                                'utility_id_pudl',
                                                'plant_id_ferc1'])) < 1.1):
        raise AssertionError(
            'Merge issue with pudl_outs plants_steam_ferc1 and fbp_ferc1')

    return steam


###########################
# Generate feature vectors
###########################

def make_candidate_links(dfa, dfb, block_col=None):
    """Generate canidate links for comparison features."""
    indexer = rl.Index()
    indexer.block(block_col)
    return indexer.index(dfa, dfb)


def make_features(dfa, dfb, block_col=None):
    """Generate comparison features based on defined features."""
    compare_cl = rl.Compare(features=[
        String('plant_name_ferc1', 'plant_name_new',
               label='plant_name', method='jarowinkler'),
        Numeric('net_generation_mwh', 'net_generation_mwh',
                label='net_generation_mwh', method='exp', scale=1000),
        Numeric('capacity_mw', 'capacity_mw',
                label='capacity_mw', method='exp', scale=10),
        Numeric('total_fuel_cost', 'total_fuel_cost', label='total_fuel_cost',
                method='exp', offset=2500, scale=10000, missing_value=0.5),
        Numeric('total_mmbtu', 'total_mmbtu', label='total_mmbtu',
                method='exp', offset=1, scale=100, missing_value=0.5),

        Numeric('capacity_factor', 'capacity_factor', label='capacity_factor'),
        Numeric('fuel_cost_per_mmbtu', 'fuel_cost_per_mmbtu',
                label='fuel_cost_per_mmbtu'),
        Numeric('heat_rate_mmbtu_mwh', 'heat_rate_mmbtu_mwh',
                label='heat_rate_mmbtu_mwh'),

        Exact('fuel_type_code_pudl', 'fuel_type_code_pudl',
              label='fuel_type_code_pudl'),
        Exact('installation_year', 'installation_year',
              label='installation_year'),
        Exact('utility_id_pudl', 'utility_id_pudl', label='utility_id_pudl'),
        # Exact('plant_id_pudl', 'plant_id_pudl', label='plant_id_pudl'),
    ])

    features = compare_cl.compute(
        make_candidate_links(dfa, dfb, block_col), dfa, dfb)
    return features

###############################
# Testing and Cross Validation
###############################


# K-fold cross validation
def kfold_cross_val(n_splits, features_known, known_index, lrc):
    """
    K-fold cross validation.

    Args:
        n_splits (int): the number of splits for the cross validation. If 5,
            the known data will be spilt 5 times into testing and training sets
            for example.
        features_known (pandas.DataFrame): a dataframe of comparison features.
            This should be created via `make_features`. This will contain all
            possible combinations of matches between your records.
        known_index (pandas.MultiIndex): an index with the known matches. The
            index must be a mutltiindex with record ids from both sets of
            records.

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
        # generate and compile the scores and outcomes of the prediction
        fscore.append(rl.fscore(y_test, links_pred=result_lrc))
        precision.append(rl.precision(y_test, links_pred=result_lrc))
        accuracy.append(rl.accuracy(
            y_test, links_pred=result_lrc, total=result_lrc))
        result_lrc_complied = result_lrc_complied.append(
            pd.DataFrame(index=result_lrc))
    return result_lrc_complied, fscore, precision, accuracy


def fit_predict_option(solver, c, cw, p, l1, n_splits, multi_class,
                       features_known, training_index, results_options):
    """Test and cross validate with a set of model parameters."""
    logger.debug(f'train: {solver}: c-{c}, cw-{cw}, p-{p}, l1-{l1}')
    lrc = rl.LogisticRegressionClassifier(solver=solver,
                                          C=c,
                                          class_weight=cw,
                                          penalty=p,
                                          l1_ratio=l1,
                                          random_state=0,
                                          multi_class=multi_class,
                                          )
    results, fscore, precision, accuracy = kfold_cross_val(
        lrc=lrc,
        n_splits=n_splits,
        features_known=features_known,
        known_index=training_index)
    results_options = results_options.append(pd.DataFrame(
        data={'solver': [solver],
              'precision': [statistics.mean(precision)],
              'f_score': [statistics.mean(fscore)],
              'accuracy': [statistics.mean(accuracy)],
              'c': [c],
              'cw': [cw],
              'penalty': [p],
              'l1': [l1],
              'multi_class': [multi_class],
              'predictions': [len(results)],
              # 'df': [results],
              'coef': [lrc.coefficients],
              'interc': [lrc.intercept],
              },
    ))
    return results_options, lrc


def test_model_parameters(features_known, training_index, n_splits):
    """
    Test and corss validate model parameters.

    Args:
        features_known (pandas.DataFrame)
        training_index (pandas.MultiIndex)
        n_splits (int)
    Returns:
        pandas.DataFrame: dataframe in which each record contains results of
            each model run.

    """
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    cs = [1, 10, 100, 1000]
    cws = ['balanced', None]
    ps = {'newton-cg': ['l2', 'none'],
          'lbfgs': ['l2', 'none'],
          'liblinear': ['l1', 'l2'],
          'sag': ['l2', 'none'],
          'saga': ['l1', 'l2', 'elasticnet', 'none'],
          }
    multi_classes = ['auto', 'ovr', 'multinomial']

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
                        # liblinear solver doesnt allow multinomial multi_class
                        if solver == 'liblinear':
                            multi_classes = ['auto', 'ovr']
                        else:
                            multi_classes = ['auto', 'ovr', 'multinomial']
                        for multi_class in multi_classes:
                            results_options, lrc = fit_predict_option(
                                solver=solver, c=c, cw=cw, p=p, l1=l1,
                                n_splits=n_splits, multi_class=multi_class,
                                features_known=features_known,
                                training_index=training_index,
                                results_options=results_options)
    return results_options


def _apply_weights(features, coefs):
    """
    Apply coefficient weights to each feature.

    Args:
        features (pandas.DataFrame): a dataframe containing features of
            candidate or model matches. The order of the columns matters! They
            must be in the same order as they were fed into the model that
            produced the coefficients.
    coefs (array): array of integers with the same length as the columns in
        features.

    """
    assert len(coefs) == len(features.columns)
    for coef_n in np.array(range(len(coefs))):
        features[features.columns[coef_n]
                 ] = features[features.columns[coef_n]].multiply(coefs[coef_n])
    return features


def weight_features(features, coefs):
    """
    Weight features of candidate (or model) matches with coefficients.

    Args:
        features (pandas.DataFrame): a dataframe containing features of
            candidate or model matches. The order of the columns matters! They
            must be in the same order as they were fed into the model that
            produced the coefficients.
        coefs (array): array of integers with the same length as the columns in
            features.

    """
    df = deepcopy(features)
    return (df.
            pipe(_apply_weights, coefs).
            assign(score=lambda x: x.sum(axis=1)).
            pipe(pudl.helpers.organize_cols, ['score']).
            sort_values(['score'], ascending=False).
            sort_index(level='record_id_ferc'))


def calc_match_stats(df):
    """
    Calculate stats needed to judge candidate matches.

    Args:
        df (pandas.DataFrame): Dataframe of comparison features with MultiIndex
            containing the ferc and eia record ids.

    Returns
        pandas.DataFrame: the input df with the stats.

    """
    df = df.reset_index()
    gb = df.groupby('record_id_ferc')[['record_id_ferc', 'score']]
    df = (
        df.sort_values(['record_id_ferc', 'score'])
        # rank the scores
        .assign(rank=gb.rank(ascending=0, method='average'))
        # calculate differences between scores
        .assign(diffs=lambda x: x['score'].diff()).
        # count grouped records
        merge(pudl.helpers.count_records(df, ['record_id_ferc'], 'count'),
              how='left',).
        # calculate the iqr for each
        merge((gb.agg({'score': scipy.stats.iqr}).
               droplevel(0, axis=1).
               rename(columns={'score': 'iqr'})), left_on=['record_id_ferc'],
              right_index=True))

    # assign the first diff of each ferc_id as a nan
    df['diffs'][df.record_id_ferc != df.record_id_ferc.shift(1)] = np.nan

    # [['sum','diffs','count','rank']]
    df = df.set_index(['record_id_ferc', 'record_id_eia'])
    return(df)


def calc_murk(df, iqr_perc_diff):
    """Calculate the murky wins."""
    distinction = (df['iqr_all'] * iqr_perc_diff)
    murky_wins = (df[(df['rank'] == 1) &
                     (df['diffs'] < distinction)])
    return murky_wins


def calc_wins(df, ferc1_options, iqr_perc_diff):
    """
    Find the winners and report on winning ratios.

    With the matches resulting from a model run, generate "winning" matches by
    finding the highest ranking EIA match for each FERC record. If it is either
    the only match or it is different enough from the #2 ranked match, we
    consider it a winner. Also log win stats.

    The matches are all of the results from the model prediction. the wins are
    all of the matches that are distinct enough from it’s closest match. The
    murky_wins are the matches that are not “distinct enough” from its closes
    match. Distinct enough means that the top match isn’t one iqr away from the
    second top match.

    Args:
        df (pandas.DataFrame): dataframe with all of the model generate
            matches. This df needs to have been run through `calc_match_stats`.
        ferc1_options (pandas.DataFrame): dataframe with all of the possible.
            `record_id_ferc`s.

    Returns
        pandas.DataFrame : winning matches. Matches that had the highest rank
            in their record_id_ferc, by a wide enough margin.

    """
    unique_ferc = df.reset_index().drop_duplicates(subset=['record_id_ferc'])
    ties = df[df['rank'] == 1.5]
    distinction = (df['iqr_all'] * iqr_perc_diff)
    # for the winners, grab the top ranked,
    winners = (df[((df['rank'] == 1) & (df['diffs'] > distinction)) |
                  ((df['rank'] == 1) & (df['diffs'].isnull()))])

    murky_wins = calc_murk(df, iqr_perc_diff)

    logger.info(
        f'matches vs total ferc:  {len(unique_ferc)/len(ferc1_options):.02}')
    logger.info(
        f'wins vs total ferc:     {len(winners)/len(ferc1_options):.02}')
    logger.info(
        f'wins vs matches:        {len(winners)/len(unique_ferc):.02}')
    logger.info(
        f'murk vs matches:        {len(murky_wins)/len(unique_ferc):.02}')
    logger.info(
        f'ties vs matches:        {len(ties)/2/len(unique_ferc):.02}')
    return winners


def fit_predict_lrc(best, features_known, features_all, train_df_ids):
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
