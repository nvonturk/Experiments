/* SETS */
set VALUATIONS;
set SIGNALS;

/* PARAMETERS */
param value{i in VALUATIONS};
param prior_probs{i in VALUATIONS, j in SIGNALS};
param prior_prob_conditionals{i in VALUATIONS, j in SIGNALS};

/* VARIABLES */
var prob_alloc{i in VALUATIONS, j in SIGNALS}, >=0, <=1;
var payments{i in VALUATIONS, j in SIGNALS};

/* OBJECTIVE */
maximize revenue: sum{i in VALUATIONS, j in SIGNALS} prior_probs[i,j]*payments[i,j];

/* CONSTRAINTS */

s.t. ex_interim_ir{i in VALUATIONS}: sum{j in SIGNALS} prior_prob_conditionals[i,j]*(value[i]*prob_alloc[i,j] - payments[i,j]) >= 0;

/*
s.t. ex_post_ir{i in VALUATIONS, j in SIGNALS}: value[i]*prob_alloc[i,j] - payments[i,j] >= 0;
*/

s.t. bne_ic{i1 in VALUATIONS, i2 in VALUATIONS: i1!=i2}: sum{j in SIGNALS} prior_prob_conditionals[i1,j]*((value[i1]*prob_alloc[i1,j] - payments[i1,j]) - (value[i1]*prob_alloc[i2,j] - payments[i2,j])) >= 0;

data;

set VALUATIONS := v1 v2 v3 v4 v5;
set SIGNALS := s1 s2 s3 s4 s5;

param value := v1 1 v2 2 v3 3 v4 4 v5 5;

param prior_probs :
		s1  s2  s3 	s4  s5 :=
v1   0.025   0.0342857142857   0.0178571428571   0.00428571428571   0.000714285714286
v2   0.04   0.0857142857143   0.0771428571429   0.0335714285714   0.00785714285714
v3   0.02   0.0764285714286   0.117857142857   0.0885714285714   0.02
v4   0.005   0.0357142857143   0.0828571428571   0.0942857142857   0.0471428571429
v5   0.000714285714286   0.00642857142857   0.0178571428571   0.035   0.0257142857143;

param prior_prob_conditionals :
		s1  s2  s3 	s4  s5 :=
v1   0.304347826087   0.417391304348   0.217391304348   0.0521739130435   0.00869565217391
v2   0.163742690058   0.350877192982   0.315789473684   0.137426900585   0.0321637426901
v3   0.0619469026549   0.236725663717   0.365044247788   0.274336283186   0.0619469026549
v4   0.0188679245283   0.134770889488   0.312668463612   0.355795148248   0.177897574124
v5   0.00833333333333   0.075   0.208333333333   0.408333333333   0.3;

end;