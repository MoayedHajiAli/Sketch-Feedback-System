function [ line ] = extractAnnotFileName( line )
	line = line( length('annot_') + 1 : end );		% remove annot_ first
	suffixIdx = findstr(line, '.xml');			% find first .xml
	suffixIdx = suffixIdx(1) -1;
	line = line(1:suffixIdx);				% remove _ and rest
	line = [line '.xml'];
end

