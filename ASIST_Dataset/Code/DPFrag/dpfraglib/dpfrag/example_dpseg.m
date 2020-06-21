warning off all
clear all
close all
clc

% imagefiles = dir('13/xml/*.xml');  
% [r c] = size(imagefiles);
% k=1;
% 
% for i=1:r
%    strokes = read_sketch(imagefiles(i,1).name); 
%    s = strcat(num2str(k),'.mat');
%    save(s,'strokes');
%    k=k+1;
% end

params = getparams('C:\Users\KemalTugrul\Desktop\Dropbox\Bin\Study\Research\1003\SoftWorks\Modules\sketchBases\Annotated\14Partial');
[trs tes trf tef] = get_strokes(params.setdir, params.extension, params.exceptions, .8);

params = get_bestparams(trs, params);
prec = dpseg_train(trs, params);
save env_14_Partial



%annot(trf);





% dirData = dir(params.setdir);
% dirIndex = [dirData.isdir];  %# Find the index for directories
% fileList = {dirData(~dirIndex).name}';
% for i=1:length(fileList)
%     load([params.setdir fileList{i}]);
%     trs = [trs strokes];
% end
% params = get_bestparams(trs, params);
% prec = dpseg_train(trs, params);
% save env_new



% load env_new;
% for i=1:length(trs)
%     current = trs(i);
%     current = douglas_peucker(current, params.douglasthresh);
%     current = resample(current, params.resample_interval, params.anglemeasure);     
%     current = fragstroke(current, prec, params);
%     plot_stroke(current);
%     pause;
%     close
% end