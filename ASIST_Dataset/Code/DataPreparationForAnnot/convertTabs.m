% changeTabs.m
%
% Kemal Tugrul Yesilbek
%
% Copies all files recursively to destination folder
%
%
% There are unmatching .sketchdata formattings. The data collected with
% early versions stores every point with x, y, pressure, t, stroke no.
% However, as new versions of data collector does not store pressure, the
% files becomes unmatching. So this issue is handled in this code.

close all;
clear all;
clc;
workspace,


%% Options
rootFolder = 'C:\Users\KemalTugrul\Desktop\Files_sketchdata';
destFolder = 'C:\Users\KemalTugrul\Desktop\Files_sametype';

% Get all Files
files = getAllFiles(rootFolder);

% Open files
parfor i = 1 : length(files)
	
	filePath = files{i};
	
	% Check if the file is collected using early, or later version of data
	% collector
	fid = fopen(filePath);

	outstr = '';
	
	tline = fgets(fid);
	while ischar(tline)
		
		
		if(tline(1) == '#')
			outstr = [outstr tline];
			
		else
			
			parts = regexp(tline, '\t', 'split');
			
			if(length(parts) == 1) % This is seperated by "space" and has pressure data
				parts = regexp(tline, ' ', 'split');
				
				
				% I just need x, y, stroke and t, in this order, which are indexed with:
				% 1, 2, 5, 4
				combedOut = [parts{1}, '*', parts{2}, '*', parts{5}(1:end-1), '*', parts{4}, '.000'];
				combedOut = regexprep(combedOut, '*', '\t');
				
				outstr = [outstr combedOut  '*'];
				outstr = regexprep(outstr, '*', '\n');
				
			else % This is seperated by "space" and does not have pressure data
				outstr = [outstr tline];
			end
			
			
		end
		
		tline = fgets(fid);
	end
	
	% Save to file
	
	full = files{i};
	lastIdx = findstr(full, '\'); lastIdx = lastIdx(end);
	name = full(lastIdx+1:end);
	newFilePath = [destFolder '\' name];
	
	fileID = fopen(newFilePath,'w');
	fprintf(fileID,'%s',outstr);
	fclose(fileID);
	
	fprintf('File: %d / %d\n', i, length(files) );

	fclose(fid);
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
% 	str = fileread(filePath);
% 	
% 	
% 	updatedString = regexprep(str, '\t', ' ');
% 	 
% 	full = files{i};
% 	lastIdx = findstr(full, '\'); lastIdx = lastIdx(end);
% 	name = full(lastIdx+1:end);
% 	newFilePath = [destFolder '\' name];
% 	
% 	fileID = fopen(newFilePath,'w');
% 	fprintf(fileID,'%s',updatedString);
% 	fclose(fileID);
% 	
% 	fprintf('File: %d / %d\n', i, length(files) );
	
end











