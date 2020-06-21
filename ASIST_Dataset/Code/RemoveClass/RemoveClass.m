% RemoveClass.m
%
% Removes class from a dataset after let you analyze it.
%
% Kemal Tugrul Yesilbek
% October 20-Oct-2015
%

%% Initialize
close all;
clear all;
clc;

fprintf('Script Start\n');

%% Options
DatasetsBasePath  = 'C:\Users\KemalTugrul\GoogleDrive\Bin\Study\Research\1003\SoftWorks\SketchDatabases\MatlabPrepared\1003\';
DatasetName       = 'TreeQuestion';
ClassNamesFile    = 'C:\Users\KemalTugrul\GoogleDrive\Bin\Study\Research\1003\SoftWorks\DataPreperation\AnnotDataLabeling\Labels\Labels.mat';

%% Load files
fprintf('Loading Files...\n');

tmp     = load([DatasetsBasePath DatasetName '\' 'Labels.mat']);
Labels  = tmp.labels;

tmp         = load(ClassNamesFile);
ClassNames  = tmp.labels;

%% Show class histogram
NonNegClasses = unique( Labels( find(Labels > 0) ) );
ClassHist(Labels, ClassNames);

%% Get the classes to omit from User
x = inputdlg('Enter Classes to Keep (space seperated):', 'Class Omition', [1 50]);
ClassesToKeep = str2num(x{:});
ClassesToOmit = setdiff(NonNegClasses, ClassesToKeep);

%% Make omited classes garbage (-1)
fprintf('Class Omition...\n');

for OmitCl = 1 : length(ClassesToOmit)
  if(ismember(ClassesToOmit(OmitCl), NonNegClasses))
    ClassMemberIdx = find(Labels == ClassesToOmit(OmitCl));
    Labels(ClassMemberIdx) = -1;
  end
end

%% Show class histogram again
fprintf('\n\n');
ClassHist(Labels, ClassNames);

%% Save to disk
str = input('All Ok, Type OK and hit enter...','s')
if(strcmp('OK', str))
  labels = Labels;
  save('labels.mat', 'labels');
end

fprintf('The new labels has been saved to pwd...\n');

%% Script end
fprintf('Script end...\n');
