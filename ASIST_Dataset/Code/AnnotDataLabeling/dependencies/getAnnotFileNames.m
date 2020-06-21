% getAnnotFileNames.m
%
% Returns annotation file names corresponding to the passed xml file.
%

function [ annotFiles ] = getAnnotFileNames( xmlFileName, annotFileNameList )

fileNameForAnnot = cell(length(annotFileNameList),1);
for fil = 1 : length(annotFileNameList)
	fileNameForAnnot{fil} = extractAnnotFileName( annotFileNameList{fil} );
end

idx = 1; annotFiles = {};
for fil = 1 : length(annotFileNameList)
	if( strcmp( fileNameForAnnot{fil}, xmlFileName ) )
		annotFiles{idx} = annotFileNameList{fil};
		idx = idx + 1;
	end
end
	

end

function [ line ] = extractAnnotFileName( line )
	line = line( length('annot_') + 1 : end );		% remove annot_ first
	suffixIdx = strfind(line, '.xml');			% find first .xml
	suffixIdx = suffixIdx(1) -1;
	line = line(1:suffixIdx);				% remove _ and rest
	line = [line '.xml'];
end

