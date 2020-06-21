% CompareDatasets.m
%
% Kemal Tugrul Yesilbek
% November-2015
%
% Given two folders which holds xmls, finds the differences in those datasets
%

%% Initialize
close all;
clear all;
clc;
fprintf('Script Start...\n');

%% Options
Extension = '.xml';

%% Get dataset folders form user
BaseFolderPath    = uigetdir(pwd, 'Select Base Dataset');
CompareFolderPath = uigetdir(pwd, 'Select Compare Dataset');

%% Get Files from both folders
fprintf('Reading Files...\n')
BaseFiles = GetFiles(BaseFolderPath, true, Extension);
CompareFiles = GetFiles(CompareFolderPath, true, Extension);

%% Get differences

% Look up from base view
MissingInCompare = {}; Index = 1;
for fb = 1 : length(BaseFiles)
  HasFile = false;

  for fc = 1 : length(CompareFiles)
    if(strcmp(BaseFiles{fb}, CompareFiles{fc}))
      HasFile = true;
      break;
    end
  end

  if(~HasFile)
    MissingInCompare{Index} = BaseFiles{fb};
    Index = Index + 1;
  end
end

% Look up from base view
MissingInBase = {}; Index = 1;
for fc = 1 : length(CompareFiles)
  HasFile = false;

  for fb = 1 : length(BaseFiles)
    if(strcmp(BaseFiles{fb}, CompareFiles{fc}))
      HasFile = true;
      break;
    end
  end

  if(~HasFile)
    MissingInBase{Index} = CompareFiles{fc};
    Index = Index + 1;
  end
end

%% Print decision
if(isempty(MissingInBase) && isempty(MissingInCompare))
  fprintf('No differences in datasets...\n');
else
  fprintf('================\n');
  fprintf('Missing in base:\n\n')

  for i = 1 : length(MissingInBase)
    fprintf('%d: %s\n', i, MissingInBase{i});
  end
  fprintf('================\n\n\n');
  % ----------------------------------------------
  fprintf('================\n');
  fprintf('Missing in compare:\n\n')

  for i = 1 : length(MissingInCompare)
    fprintf('%d: %s\n', i, MissingInCompare{i});
  end
  fprintf('================\n');

end


%% End of Script
fprintf('Script End...\n');
