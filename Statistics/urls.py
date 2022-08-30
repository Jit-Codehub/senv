from django.urls import path
from . import views
from .views import *

urlpatterns = [
    # path('odds-calculator/',views.odds,name="odds"),
    path('accuracy-calculator/',views.accuracy,name="accuracy"),
    path('random-number-calculator/',views.random_number,name="random_number"),
    path('geometric-distribution-calculator/',views.geometric_distribution,name="geometric_distribution"),
    path('hypergeometric-distribution-calculator/',views.hypergeometric_distribution,name="hypergeometric_distribution"), 
    path('negative-binomial-distribution-calculator/',views.negative_binomial_distribution,name="negative_binomial_distribution"), 
    path('poisson-distribution-calculator/',views.poisson_distribution,name="poisson_distribution"), 
    path('exponential-distribution-calculator/',views.exponential_distribution,name="exponential_distribution"),
    path('exponential-growth-prediction-calculator/',views.exponential_growth_prediction,name="exponential_growth_prediction"),
    path('binomial-distribution-calculator/',views.binomial_distribution,name="binomial_distribution"),
    path('chi-square-calculator/',views.chi_square,name="chi_square"),
    path('empirical-rule-calculator/',views.empirical_rule,name="empirical_rule"),
    path('population-mean-calculator/',views.population_mean,name="population_mean"),
    path('sample-mean-calculator/',views.sample_mean,name="sample_mean"),
    path('mid-range-calculator/',views.mid_range,name="mid_range"),
    path('coefficient-of-variation-calculator/',views.coefficient_of_variation,name="coefficient_of_variation"),
    path('range-calculator/',views.range_cal,name="range"),
    path('percentile-calculator/',views.percentile,name="percentile"),
    path('iqr-calculator/',views.iqr,name="iqr"),
    path('coefficient-of-determination-calculator/',views.coefficient_of_determination,name="Coefficient_of_Determination"),
    path('confidence-interval-calculator/',views.confidence_interval,name="Confidence_Interval"),
    path('correlation-coefficient-matthews-calculator/',views.correlation_coefficient_matthews,name="Correlation_Coefficient_Matthews"),
    path('point-estimate-calculator/',views.point_estimate,name="Point_Estimate"),
    path('relative-error-calculator/',views.relative_error,name="Relative_Error"),
    path('z-test-calculator/',views.z_test,name="Z-test"),
    path('margin-of-error-calculator/',views.margin_of_error,name="Margin_of_Error"),
    path('shannon-entropy-calculator/',views.shannon_entropy,name="shannon_entropy"),
    path('exponential-regression-calculator/',views.exponential_regression,name="exponential_regression"),
    path('fishers-exact-test-calculator/',views.fishers_exact_test,name="fishers_exact_test"),
    path('critical-value-calculator/',views.critical_value,name="critical_value"),
    path('median-calculator/',views.median, name='median'),
    path('median-of-<str:num1>/',views.median_detail,name='median-detail'),

    path('mode-calculator/',views.mode, name='mode'),
    path('mode-of-<str:num1>/',views.mode_detail,name='mode-detail'),

    path('mean-calculator/',views.mean, name='mean'),
    path('mean-of-<str:num1>/',views.mean_detail,name='mean-detail'),

    path('first-quartile-calculator/',views.first_quartile, name='first-quartile'),
    path('first-quartile-of-<str:num1>/',views.first_quartile_detail,name='first-quartile-detail'),

    path('third-quartile-calculator/',views.third_quartile, name='third-quartile'),
    path('third-quartile-of-<str:num1>/',views.third_quartile_detail,name='third-quartile-detail'),

    path('maximum-calculator/',views.maximum_number, name='maximum-number'),
    path('maximum-of-<str:num1>/',views.maximum_number_detail,name='maximum-number-detail'),

    path('minimum-calculator/',views.minimum_number, name='minimum-number'),
    path('minimum-of-<str:num1>/',views.minimum_number_detail,name='minimum-number-detail'),

    path('five-number-summary-calculator/',views.five_summary, name='five-summary'),
    path('five-number-summary-of-<str:num1>/',views.five_summary_detail,name='five-summary-detail'),

    path('relative-standard-deviation-calculator/',views.relative_standard_dev, name='relative-standard-deviation-calculator'),

    path('anova-calculator/',views.anova_calc, name='anova-calculator'),

    path('coin-toss-probability-calculator/',views.coin_toss_prob, name='coin-toss-probability-calculator'),

    path('covariance-calculator/',covariance_calculator, name='covariance-calculator'),
    path('gamma-function-calculator/',gamma_function_calculator, name='gamma-function-calculator'),
    path('linear-correlation-coefficient-calculator/',linear_correlation_coefficient_calculator, name='linear-correlation-coefficient-calculator'),
    path('mean-deviation-calculator/',mean_deviation_calculator, name='mean-deviation-calculator'),
    path('pearson-correlation-calculator/',pearson_correlation_calculator, name='pearson-correlation-calculator'),
    
    path('average-calculator/',average_calculator, name='average-calculator'),
    path('descriptive-statistics-calculator/',descriptive_statistics_calculator, name='descriptive_statistics'),
    path('final-grade-calculator/',final_grade_calculator, name='final-grade-calculator'),
    path('mean-median-mode-calculator/',mean_median_mode_calculator, name='mean-median-mode-calculator'),
    path('odds-probability-calculator/',odds_probability_calculator, name='odds-probability-calculator'),
    path('standard-deviation-calculator/',standard_deviation_calculator, name='standard-deviation-calculator'),
    path('statistics-formulas/',statistics_formulas, name='statistics-formulas'),
    path('variance-formulas/',variance_formulas, name='variance-formulas'),
    path('vote-percentage-calculator/',vote_percentage_calculator, name='vote-percentage-calculator'),
    path('z-score-calculator/',z_score_calculator, name='z-score-calculator'),
    path("coin-flip-probability/",coin_flip_probability,name="coin_flip_probability"),
    path("average-rate-of-change-calculator/",average_rate_of_change_cal,name="average_rate_of_change_calc"),
    path('linear-regression-calculator/',linear_regression,name="linear_regression"),
    path('bayes-theorem-calculator/',bayes_theorem,name="bayes-theorem"),
    path('birthday-paradox-calculator/', birthday_paradox, name="birthday-paradox"),
    path('chebyshevs-theorem-calculator/', cheb_theorem, name="cheb-theorem"),
    path('permutation-calculator/', perm, name="perm"),
    path('combination-calculator/', comb, name="comb"),
    path('relative-risk-calculator/', relative_risk, name="relative-risk"),
    path('variance-calculator/',variance_formulas,name="variance-formulas1"),
    path("population-variance-calculator/",population_variance,name="population-variance"),
    path("pearson-correlation-coefficient-calculator/",pearson_coeff,name="pearson_coeff"),
    path('class-width-calculator/',class_width ,name='class_width '),
    path('standard-deviation-index-calculator/',standard_deviation_index,name='standard_deviation_index'),
    path('odds-calculator/',odds_calculator,name="odds-calculator"),
    path('normal-distribution-calculator/',normal_distribution,name="normal_distribution"),
    path('probability-calculator/',probability_calculator,name="probability"),
    path('p-value-calculator/',p_value,name='p-value-calculator'),
    path('sample-size-calculator/',sample_size,name="sample-size-calculator"),
    path('dice-probability-calculator/',dicecalculator,name="dicecalculator"),
    path('coin-flip-probability/',coin_flip,name='coin_flip '),
    path('risk-calculator/',risk_calculator,name='risk_calc'),
    path('lottery-calculator/', lottery_calculator, name='lottery'),
    path('dice-roll-calculator/', dice_calculator, name='dice'),
    path('z-test-calculator/',z_test,name='z-test '),
    path('probability-3-events-calculator/',probability_3_events,name='probability-3-events'),
    path('central-limit-theorem-calculator/',central_limit,name="central"),
    path('lognormal-distribution-calculator/',lognormal_distribution_calculator,name="lognormal"),
    path('Rayleigh-distribution-calculator/',Rayleigh_distribution_calculator,name="Rayleigh"),
    path('weibull-distribution-calculator/',weibull_distribution_calculator,name="weibull"),
    path('MannWhitney-U-Test-Calculator/', mann_whitney_u_test, name='calculate'),
    path('Sensitivity-Specificity-Calculator/',sensitivity_Specificity,name='risk_calc'),
    path('Benfords-law-Calculator/',benfords_law,name="benfords_law"),
    path('Smp(x)-Calculator/',smpx,name='smpx'),
    path('Expected-Value-Calculator/',expected_value_calc,name='expected_value_calc'),
    path('Histogram-Calculator/',histogram,name='histogram'),
    path('t-test-Calculator/',t_test,name='t_test'),
    path('operational-ratio-Calculator/',operational_ratio,name='operational_ratio'),
    path('probability-ratio-Calculator/',probability_ratio,name='probability_ratio'),
    path('probability-density-function-calculator/', pdf, name='pdf'),
    path('Exponential-Distribution-Calculator/', exprocalc, name='exprocalc'),
    path('mean-probability-calculator/',prob,name="prob"),
    path('frequency-distribution-calculator/',freq,name="freq"),
    path('conditional-probability-calculator/',cpc,name="cpc"),
    path('empirical-probability-calculator/',empirical_probability,name='empirical-probability'),
    path('card-deck-calculator/',cal_prob,name='card-deck'),
    path('geomatric-probability-calculator/',geomatric_probability,name='geomatric-probability'),
    path('uniform-distribution-calculator/',uniform_distribution,name='uniform-distribution'),
    path('coin-flip-probability-calculator/',coin_flip,name='coin_flip '),
    path('roulette-probability-calculator/',rpc,name='rpc'),
    path('variance-frequency-calculator/',vft,name='vft'),
    path('modal-frequency-table-calculator/',modal_frequency_table,name='modal_frequency_table'),
    path('In-dependent-calculator/',in_dependent,name='in_dependent'),
    path('poker-probability-calculator/',poker_probability,name='poker_probability'),
    path('probability-dependent-calculator/',probability_dependent,name='probability_dependent'),
    path('cumlative-frequency-calculator/',cumlative_frequency,name='cumlative-frequency'),
    path('percentage-frequency-calculator/',percentage_frequency,name='percentage_probability'),
    path('group-frequency-distribution-calculator/',group_frequency_distribution,name='group_frequency_distribution'),
    path('mutually-exclusive-calculator/',Mutually_exclusive,name='Mutually_exclusive'),
    path('non-mutually-exclusive-calculator/',NonMutually_exclusive,name='NonMutually_exclusive'),
    path('probability-independent-calculator/',probability_independent,name='probability_independent'),
    ]