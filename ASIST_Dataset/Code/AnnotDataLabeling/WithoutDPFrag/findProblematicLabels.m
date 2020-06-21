function [missing, extra] = findProblematicLabels( labels, annotFiles )

% Parse labels
labelsPosIdx = find(labels ~= -1);

labelsPos = [];
for i = 1 : length(labelsPosIdx)
	labelsPos = [ labelsPos ; labels( labelsPosIdx(i) ) ];
end

% Parse annot files
annotClasses = [];
annotStructs = loadAnnotFiles(annotFiles);
for annot = 1 : length(annotStructs)
	annotClasses = [annotClasses; findClassNo(annotStructs{annot}.label)];
end

% Get max. values
maxBin = 0;
if( max(labelsPos) > max(annotClasses) )
	maxBin = max(labelsPos);
else
	maxBin = max(annotClasses);
end

% Count occurances
histLabels = countVec(labelsPos, maxBin);
histAnnot = countVec(annotClasses, maxBin);

% Differences
diff = histLabels - histAnnot;

% Record extras and missings
missingIdx = find(diff < 0);
extraIdx = find(diff > 0);

if(length(missingIdx) > 0)
	for i = 1 : length(missingIdx)
		missing.label{i} = findClassName( missingIdx(i) );
		missing.count(i) = diff( missingIdx(i) );
	end
else
	missing = [];
end

if(length(extraIdx) > 0)
	for i = 1 : length(extraIdx)
		extra.label{i} = findClassName( extraIdx(i) );
		extra.count(i) = diff( extraIdx(i) );
	end
else
	extra = [];
end

end

function h = countVec( x, hbin )

h = zeros( hbin, 1);

for i = 1 : length(x)
	c = x(i);
	h(c) = h(c) + 1;
end

end

function annotationStructs = loadAnnotFiles(annotFiles)
fprintf('Loading annotation files...\n');

if(isempty( annotFiles ) )
	annotationStructs = {};
else
	
	for file = 1 : length(annotFiles)
		tmp = load( annotFiles{file} );
		annotationStructs{file} = tmp.annotData;
	end
	
end
end

function className = findClassName( classNo )
	tmp = load('Labels\labels.mat');
	labels = tmp.labels;
	className = labels{classNo};
end

function classNo = findClassNo( className )
	tmp = load('Labels\labels.mat');
	labels = tmp.labels;
	
	for i = 1 : length(labels)
		lab = labels{i};
		if( strcmp( className, lab ) )
			classNo = i;
			return;
		end
	end
	
	%classNo = find(ismember(labels, className));
end