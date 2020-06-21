% transferFiles.m
%
% Kemal Tugrul Yesilbek
%
% Copies all files recursively to destination folder
%

close all;
clear all;
clc;
workspace,


%% Options
rootFolder = 'C:\Users\KemalTugrul\Desktop\AllData_1003_20150402';
destFolder = 'C:\Users\KemalTugrul\Desktop\Files';

% Get all Files
files = getAllFiles(rootFolder);

for f = 1 : length(files)
	full = files{f};
	lastIdx = findstr(full, '\'); lastIdx = lastIdx(end);
	name = full(lastIdx+1:end)
	
	system(['COPY ' '"' files{f} '"' ' ' destFolder '\' name '"']);
	
	fprintf('\n f: %d / %d\n', f, length(files));
end