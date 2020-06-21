 function EVAL = evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures


posidx = (ACTUAL()>0);

p = length(ACTUAL(posidx));
n = length(ACTUAL(~posidx));
N = p+n;

tp = sum(ACTUAL(posidx)==PREDICTED(posidx));
tn = sum(ACTUAL(~posidx)==PREDICTED(~posidx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

EVAL.accuracy = (tp+tn)/N;
EVAL.sensitivity = tp_rate;
EVAL.specificity = tn_rate;
EVAL.precision = tp/(tp+fp);
EVAL.recall = EVAL.sensitivity;
EVAL.f_measure = 2*((EVAL.precision*EVAL.recall)/(EVAL.precision + EVAL.recall));
EVAL.gmean = sqrt(tp_rate*tn_rate);
EVAL.tpr = tp_rate;
EVAL.tnr = tn_rate;
EVAL.fp = fp;
EVAL.fn = fn;
EVAL.tp = tp;
EVAL.tn = tn;

%EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
