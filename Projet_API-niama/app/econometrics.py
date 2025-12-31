"""
Econometric Analysis Module
Author: MoSEF Student
Description: Validates sentiment-returns relationship using VAR models and Granger causality.
Adapts methodology from "From Tweets to Returns" paper to cryptocurrency markets.
Master 2 MoSEF Data Science 2024-2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import ccf
import warnings

warnings.filterwarnings('ignore')


class EconometricAnalyzer:
    """
    Econometric analysis for sentiment-returns relationship.
    
    Methodology (adapted from academic paper):
    1. Aggregate daily sentiment scores
    2. Calculate log returns from prices
    3. Test stationarity (ADF test)
    4. Estimate VAR model
    5. Granger causality tests
    6. Cross-correlation analysis
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.results = {}
    
    # ==================== Data Preparation ====================
    
    def prepare_sentiment_data(
        self,
        posts: list[dict],
        sentiments: list[dict]
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by date.
        
        Args:
            posts: List of scraped posts with 'created_utc'
            sentiments: List of sentiment results with 'score'
        
        Returns:
            DataFrame with daily aggregated sentiment
        """
        if len(posts) != len(sentiments):
            raise ValueError("Posts and sentiments must have same length")
        
        # Create DataFrame
        data = []
        for post, sent in zip(posts, sentiments):
            date_str = post.get('created_utc', '')[:10]  # Extract YYYY-MM-DD
            data.append({
                'date': date_str,
                'sentiment': sent.get('score', 0),
                'label': sent.get('label', 'neutral')
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate by day
        daily = df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count'],
            'label': lambda x: (x == 'positive').sum() / len(x)  # % positive
        }).reset_index()
        
        daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'n_posts', 'pct_positive']
        daily = daily.sort_values('date').reset_index(drop=True)
        
        return daily
    
    def prepare_price_data(self, prices: list[dict]) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: List of price dicts with 'date' and 'price'
        
        Returns:
            DataFrame with prices and returns
        """
        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate log returns
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        # Calculate simple returns
        df['simple_return'] = df['price'].pct_change()
        
        # Volatility (rolling 7-day std)
        df['volatility'] = df['log_return'].rolling(7).std()
        
        return df
    
    def merge_data(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge sentiment and price data.
        
        Args:
            sentiment_df: Daily sentiment data
            price_df: Daily price/returns data
        
        Returns:
            Merged DataFrame
        """
        merged = pd.merge(
            price_df,
            sentiment_df,
            on='date',
            how='inner'
        )
        
        # Drop rows with NaN
        merged = merged.dropna()
        
        return merged
    
    # ==================== Stationarity Tests ====================
    
    def adf_test(self, series: pd.Series, name: str = "Series") -> dict:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        
        Args:
            series: Time series to test
            name: Name for reporting
        
        Returns:
            Dict with test results
        """
        result = adfuller(series.dropna(), autolag='AIC')
        
        return {
            'series': name,
            'adf_statistic': round(result[0], 4),
            'p_value': round(result[1], 4),
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': {k: round(v, 4) for k, v in result[4].items()},
            'is_stationary': result[1] < 0.05,
            'interpretation': f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'} at 5% level"
        }
    
    def check_stationarity(self, df: pd.DataFrame) -> dict:
        """
        Check stationarity for all relevant series.
        
        Args:
            df: Merged DataFrame with sentiment and returns
        
        Returns:
            Dict with ADF test results for each series
        """
        results = {}
        
        series_to_test = [
            ('sentiment_mean', 'Sentiment'),
            ('log_return', 'Log Returns'),
            ('price', 'Price Level')
        ]
        
        for col, name in series_to_test:
            if col in df.columns:
                results[col] = self.adf_test(df[col], name)
        
        return results
    
    # ==================== VAR Model ====================
    
    def estimate_var(
        self,
        df: pd.DataFrame,
        maxlags: int = 14,
        ic: str = 'aic'
    ) -> dict:
        """
        Estimate Vector Autoregression model.
        
        Args:
            df: DataFrame with sentiment and returns
            maxlags: Maximum lags to consider
            ic: Information criterion ('aic', 'bic', 'hqic')
        
        Returns:
            Dict with VAR results
        """
        # Prepare data for VAR
        var_data = df[['sentiment_mean', 'log_return']].dropna()
        
        if len(var_data) < maxlags + 10:
            return {'error': f'Insufficient data: {len(var_data)} observations'}
        
        # Fit VAR model
        model = VAR(var_data)
        
        # Select optimal lag order
        lag_order = model.select_order(maxlags=maxlags)
        optimal_lag = lag_order.selected_orders[ic]
        
        # Estimate VAR with optimal lag
        var_result = model.fit(optimal_lag)
        
        # Extract coefficients
        coefficients = {}
        for eq_name in ['sentiment_mean', 'log_return']:
            coefficients[eq_name] = {}
            for i in range(1, optimal_lag + 1):
                for var in ['sentiment_mean', 'log_return']:
                    param_name = f'L{i}.{var}'
                    if param_name in var_result.params[eq_name].index:
                        coefficients[eq_name][param_name] = round(
                            var_result.params[eq_name][param_name], 6
                        )
        
        return {
            'optimal_lag': optimal_lag,
            'lag_order_selection': {
                'aic': lag_order.selected_orders.get('aic'),
                'bic': lag_order.selected_orders.get('bic'),
                'hqic': lag_order.selected_orders.get('hqic')
            },
            'n_obs': var_result.nobs,
            'coefficients': coefficients,
            'aic': round(var_result.aic, 4),
            'bic': round(var_result.bic, 4),
            'model_summary': str(var_result.summary())[:2000]  # Truncate
        }
    
    # ==================== Granger Causality ====================
    
    def granger_causality_test(
        self,
        df: pd.DataFrame,
        maxlag: int = 14
    ) -> dict:
        """
        Granger causality tests between sentiment and returns.
        
        Tests:
        1. Sentiment → Returns (sentiment Granger-causes returns)
        2. Returns → Sentiment (returns Granger-cause sentiment)
        
        Args:
            df: DataFrame with sentiment_mean and log_return
            maxlag: Maximum lag to test
        
        Returns:
            Dict with Granger causality results
        """
        data = df[['sentiment_mean', 'log_return']].dropna()
        
        results = {
            'sentiment_causes_returns': {},
            'returns_cause_sentiment': {}
        }
        
        # Test 1: Sentiment → Returns
        try:
            gc_test_1 = grangercausalitytests(
                data[['log_return', 'sentiment_mean']],  # [y, x] format
                maxlag=maxlag,
                verbose=False
            )
            
            for lag in range(1, maxlag + 1):
                if lag in gc_test_1:
                    f_stat = gc_test_1[lag][0]['ssr_ftest'][0]
                    p_value = gc_test_1[lag][0]['ssr_ftest'][1]
                    results['sentiment_causes_returns'][f'lag_{lag}'] = {
                        'f_statistic': round(f_stat, 4),
                        'p_value': round(p_value, 4),
                        'significant': p_value < 0.05
                    }
        except Exception as e:
            results['sentiment_causes_returns']['error'] = str(e)
        
        # Test 2: Returns → Sentiment
        try:
            gc_test_2 = grangercausalitytests(
                data[['sentiment_mean', 'log_return']],
                maxlag=maxlag,
                verbose=False
            )
            
            for lag in range(1, maxlag + 1):
                if lag in gc_test_2:
                    f_stat = gc_test_2[lag][0]['ssr_ftest'][0]
                    p_value = gc_test_2[lag][0]['ssr_ftest'][1]
                    results['returns_cause_sentiment'][f'lag_{lag}'] = {
                        'f_statistic': round(f_stat, 4),
                        'p_value': round(p_value, 4),
                        'significant': p_value < 0.05
                    }
        except Exception as e:
            results['returns_cause_sentiment']['error'] = str(e)
        
        # Summary
        sig_sent_to_ret = sum(
            1 for v in results['sentiment_causes_returns'].values()
            if isinstance(v, dict) and v.get('significant', False)
        )
        sig_ret_to_sent = sum(
            1 for v in results['returns_cause_sentiment'].values()
            if isinstance(v, dict) and v.get('significant', False)
        )
        
        results['summary'] = {
            'sentiment_granger_causes_returns': sig_sent_to_ret > 0,
            'significant_lags_sent_to_ret': sig_sent_to_ret,
            'returns_granger_cause_sentiment': sig_ret_to_sent > 0,
            'significant_lags_ret_to_sent': sig_ret_to_sent,
            'interpretation': self._interpret_granger(sig_sent_to_ret, sig_ret_to_sent, maxlag)
        }
        
        return results
    
    def _interpret_granger(self, sent_to_ret: int, ret_to_sent: int, maxlag: int) -> str:
        """Generate interpretation of Granger causality results"""
        if sent_to_ret > 0 and ret_to_sent == 0:
            return f"Unidirectional causality: Sentiment Granger-causes returns ({sent_to_ret}/{maxlag} lags significant)"
        elif ret_to_sent > 0 and sent_to_ret == 0:
            return f"Unidirectional causality: Returns Granger-cause sentiment ({ret_to_sent}/{maxlag} lags significant)"
        elif sent_to_ret > 0 and ret_to_sent > 0:
            return f"Bidirectional causality: Both directions significant"
        else:
            return "No significant Granger causality detected"
    
    # ==================== Cross-Correlation ====================
    
    def cross_correlation(
        self,
        df: pd.DataFrame,
        max_lags: int = 14
    ) -> dict:
        """
        Calculate cross-correlation between sentiment and returns.
        
        Positive lag: sentiment leads returns
        Negative lag: returns lead sentiment
        
        Args:
            df: DataFrame with sentiment and returns
            max_lags: Maximum lags to compute
        
        Returns:
            Dict with cross-correlations at each lag
        """
        sentiment = df['sentiment_mean'].dropna().values
        returns = df['log_return'].dropna().values
        
        # Align series
        min_len = min(len(sentiment), len(returns))
        sentiment = sentiment[:min_len]
        returns = returns[:min_len]
        
        # Compute cross-correlation
        correlations = {}
        
        for lag in range(-max_lags, max_lags + 1):
            if lag < 0:
                # Returns lead sentiment
                corr = np.corrcoef(returns[:lag], sentiment[-lag:])[0, 1]
            elif lag > 0:
                # Sentiment leads returns
                corr = np.corrcoef(sentiment[:-lag], returns[lag:])[0, 1]
            else:
                corr = np.corrcoef(sentiment, returns)[0, 1]
            
            correlations[f'lag_{lag}'] = round(corr, 4) if not np.isnan(corr) else 0
        
        # Find optimal lag
        max_corr = max(correlations.values(), key=abs)
        optimal_lag = [k for k, v in correlations.items() if v == max_corr][0]
        
        return {
            'correlations': correlations,
            'optimal_lag': optimal_lag,
            'max_correlation': max_corr,
            'interpretation': f"Strongest correlation at {optimal_lag}: {max_corr:.4f}"
        }
    
    # ==================== Descriptive Statistics ====================
    
    def descriptive_stats(self, df: pd.DataFrame) -> dict:
        """
        Calculate descriptive statistics for the dataset.
        
        Args:
            df: Merged DataFrame
        
        Returns:
            Dict with descriptive statistics
        """
        stats_dict = {}
        
        for col in ['sentiment_mean', 'log_return', 'price']:
            if col in df.columns:
                series = df[col].dropna()
                stats_dict[col] = {
                    'count': len(series),
                    'mean': round(series.mean(), 6),
                    'std': round(series.std(), 6),
                    'min': round(series.min(), 6),
                    'max': round(series.max(), 6),
                    'skewness': round(series.skew(), 4),
                    'kurtosis': round(series.kurtosis(), 4)
                }
        
        # Correlation matrix
        corr_cols = ['sentiment_mean', 'log_return']
        if all(c in df.columns for c in corr_cols):
            corr_matrix = df[corr_cols].corr()
            stats_dict['correlation'] = {
                'sentiment_returns': round(corr_matrix.loc['sentiment_mean', 'log_return'], 4)
            }
        
        return stats_dict
    
    # ==================== Full Analysis Pipeline ====================
    
    def full_analysis(
        self,
        posts: list[dict],
        sentiments: list[dict],
        prices: list[dict],
        maxlag: int = 14
    ) -> dict:
        """
        Run complete econometric analysis pipeline.
        
        Args:
            posts: Scraped posts with timestamps
            sentiments: Sentiment analysis results
            prices: Historical price data
            maxlag: Maximum lag for tests
        
        Returns:
            Complete analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_posts': len(posts),
            'n_prices': len(prices)
        }
        
        try:
            # Prepare data
            sentiment_df = self.prepare_sentiment_data(posts, sentiments)
            price_df = self.prepare_price_data(prices)
            merged_df = self.merge_data(sentiment_df, price_df)
            
            results['data_summary'] = {
                'sentiment_days': len(sentiment_df),
                'price_days': len(price_df),
                'merged_days': len(merged_df),
                'date_range': {
                    'start': str(merged_df['date'].min()),
                    'end': str(merged_df['date'].max())
                }
            }
            
            if len(merged_df) < 15:
                results['error'] = f"Insufficient data after merge: {len(merged_df)} days (minimum 15)"
                return results
            
            # Descriptive statistics
            results['descriptive_stats'] = self.descriptive_stats(merged_df)
            
            # Stationarity tests
            results['stationarity_tests'] = self.check_stationarity(merged_df)
            
            # VAR model
            results['var_model'] = self.estimate_var(merged_df, maxlags=maxlag)
            
            # Granger causality
            results['granger_causality'] = self.granger_causality_test(merged_df, maxlag=maxlag)
            
            # Cross-correlation
            results['cross_correlation'] = self.cross_correlation(merged_df, max_lags=maxlag)
            
            # Overall conclusion
            results['conclusion'] = self._generate_conclusion(results)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _generate_conclusion(self, results: dict) -> dict:
        """Generate overall conclusion from analysis"""
        conclusion = {
            'sentiment_predicts_returns': False,
            'optimal_lag': None,
            'evidence_strength': 'weak'
        }
        
        # Check Granger causality
        gc = results.get('granger_causality', {}).get('summary', {})
        if gc.get('sentiment_granger_causes_returns', False):
            conclusion['sentiment_predicts_returns'] = True
            conclusion['evidence_strength'] = 'moderate' if gc.get('significant_lags_sent_to_ret', 0) > 3 else 'weak'
        
        # Check cross-correlation
        cc = results.get('cross_correlation', {})
        if abs(cc.get('max_correlation', 0)) > 0.3:
            conclusion['optimal_lag'] = cc.get('optimal_lag')
            if conclusion['sentiment_predicts_returns']:
                conclusion['evidence_strength'] = 'strong'
        
        conclusion['summary'] = (
            f"Sentiment {'does' if conclusion['sentiment_predicts_returns'] else 'does not'} "
            f"Granger-cause returns. Evidence strength: {conclusion['evidence_strength']}. "
            f"Optimal lag: {conclusion['optimal_lag'] or 'N/A'}."
        )
        
        return conclusion


# -------------------- Test --------------------
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_days = 60
    
    # Simulated posts
    posts = [
        {'created_utc': f'2024-01-{str(i % 28 + 1).zfill(2)}T12:00:00', 'title': f'Post {i}'}
        for i in range(100)
    ]
    
    # Simulated sentiments
    sentiments = [
        {'score': np.random.uniform(-0.5, 0.5), 'label': 'positive' if np.random.random() > 0.5 else 'negative'}
        for _ in range(100)
    ]
    
    # Simulated prices
    prices = [
        {'date': f'2024-01-{str(i + 1).zfill(2)}', 'price': 40000 + np.random.randn() * 1000}
        for i in range(28)
    ]
    
    analyzer = EconometricAnalyzer()
    results = analyzer.full_analysis(posts, sentiments, prices, maxlag=7)
    
    print("Analysis Results:")
    print(f"Conclusion: {results.get('conclusion', {}).get('summary', 'N/A')}")
