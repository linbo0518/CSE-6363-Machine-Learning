import numpy as np
import pandas as pd


class NaiveBayes:
    '''categorial naive bayes classifer'''

    def __init__(self):
        self._df = None
        self._n_sample = 0
        self._columns = None
        self._columns_counts = None
        self._targets = None
        self._targets_counts = None
        self._target_col = None

    def fit(self,
            filepath,
            sep=' ',
            target_col="Class",
            ignore_col=None,
            start_idx=0,
            end_idx=-1):
        '''compute the prior probability and likelihood.

        @params:
            filepath (str or buffer): valid string path, could be a URL.
            sep (str): Delimiter to use, default is ' '.
            target_col (str): target column name, default is "Class".
            ignore_col (str): ignore column name, default is None.
        '''
        self._df = pd.read_csv(filepath, sep=sep)
        columns = self._df.columns
        for name in columns:
            if '-' in name:
                self._df = self._df.rename(
                    columns={name: name.replace('-', '_')})
        self._df = self._df[start_idx:end_idx].reset_index(drop=True)
        self._n_sample = len(self._df)
        drop_cols = [target_col]
        if ignore_col:
            drop_cols.append(ignore_col)
        self._columns = self._df.columns.drop(drop_cols)
        self._columns_counts = np.array(self._columns)
        for idx, col in enumerate(self._columns):
            self._columns_counts[idx] = self._df[col].value_counts()
        self._targets = self._df[target_col].unique()
        self._targets_counts = self._df[target_col].value_counts()
        self._target_col = target_col

    def likelihood(self):
        likelihood = set()
        for idx in self._df.index:
            for target in self._targets:
                for col in self._columns:
                    val = self._df.iloc[idx][str(col)]
                    p_xy = self._calc_prob(col, val, target)
                    likelihood.add(f"P({col} = {val} | {target}) = {p_xy}")
        print(self._targets_counts / self._targets_counts.sum())
        likelihood = sorted(likelihood)
        for each in likelihood:
            print(each)

    def predict(self,
                filepath,
                sep=' ',
                start_idx=0,
                end_idx=-1,
                verbose=False):
        '''compute the posterior probabilityusing Bayes theorem.

        @params:
            filepath (str or buffer): valid string path, could be a URL.
            sep (str): Delimiter to use, default is ' '.
            verbose (bool): return probability of each class in a dict data 
                type, default is False
        
        @return:
            list: the predictions of input data
            dict: probability of each class
        '''
        test_df = pd.read_csv(filepath, sep=sep)
        test_df = test_df[start_idx:end_idx].reset_index(drop=True)
        results = list()
        probs = list()
        for idx in test_df.index:
            probs.append(dict())
            max_prob = 0
            max_name = None
            for target in self._targets:
                p_x = 1
                p_xy = 1
                p_y = self._calc_prob(target_value=target)
                for col in self._columns:
                    val = test_df.iloc[idx][col]
                    p_x *= self._calc_prob(col, val)
                    p_xy *= self._calc_prob(col, val, target)
                priori = (p_xy * p_y) / p_x
                if max_prob < priori:
                    max_name = target
                    max_prob = priori
                probs[idx][target] = priori
            results.append(max_name)
        if verbose:
            just_val = len(max(results, key=len)) + 2
            for result, prob in zip(results, probs):
                print(f"Label: {result}")
                print(f"Probs:")
                for pair in prob.items():
                    marker = '<' if pair[0] == result else ''
                    print(f"{pair[0].rjust(just_val)}: {pair[1]:.7f} {marker}")
                for _ in range(just_val + 15):
                    print('-', end='')
                print()
        return results

    def _calc_prob(self, column=None, value=None, target_value=None):
        '''compute the prior probability from given parameters.

        @params:
            column (str): calculate probability in which column, default is None
            value (str): calculate probability in which value, default is None
            target_value (str): calculate conditional probability in which 
                column with which value, default is None

        @return:
            float: probability calculated by given condition
        '''
        if column and value and target_value:
            n = len(
                self._df.query(
                    f"{column} == '{value}' and {self._target_col} == '{target_value}'"
                ))
            return (n + 1) / (self._targets_counts[target_value] +
                              len(self._columns))
        elif column and value and not target_value:
            idx = int(np.where(self._columns == column)[0])
            n = self._columns_counts[idx][value]
            return n / self._n_sample
        elif not column and not value and target_value:
            n = self._targets_counts[target_value]
            return n / self._n_sample
        else:
            raise ValueError("Only can calculate P(x_i), P(y) and P(x_i|y)")
