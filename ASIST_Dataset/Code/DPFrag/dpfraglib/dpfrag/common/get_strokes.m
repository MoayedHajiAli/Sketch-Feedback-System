function [trs tes trf tef] = get_strokes(filedir, patterns, exceptions, ntraining, ntesting)

    % usage : 
    % get_files('.', {'.xml'}, {'conflicted'}, 0.8);        % split [0.8, 0.2]
    % get_files('.', {'.mat'}, {'exception_file'}, 30, 20); % get [30 20]
    % get_files('.', {'.mat'});                             % get all files
    % get_files('.', {'.mat'}, {'exception_file'}, 30)      % get remainder
    
    
    trs={}; tes={}; trf={}; tef={};
    if (~exist('exceptions', 'var'))
        exceptions = {};
    end

    files = {};
    allfiles = dir(filedir);
    
    if (length(allfiles) == 1 && ~allfiles.isdir)
        trf = filedir;
        trs = read_sketch(trf);
        return;
    end

    filedir = addslash(filedir);
    for i=3:length(allfiles)
        curr_path = sprintf('%s%s', filedir, allfiles(i).name);
        if (allfiles(i).isdir)
            [trs_ tes_ trf_ tef_] = get_strokes(curr_path, patterns, exceptions);
            for j=1:length(trf_)
                files{end+1} = trf_{j};
            end
        elseif (has_pattern(curr_path, patterns))
            files{end+1} = curr_path;
        end
    end
    
    if (~isempty(exceptions))
        validind = [];
        for i=1:length(files)
            valid = true;
            for j=1:length(exceptions)
                if (~isempty(strfind(files{i}, exceptions{j})))
                    valid = false;
                    break;
                end
            end
            if (valid)
                validind(end+1) = i; 
            end
        end    
        files = files(validind);
    end

    
    nfiles = length(files);
    if (~exist('ntraining', 'var'))
        trf = files;
        tef = {};
        return;
    end
    
    if (~exist('ntesting', 'var') && ntraining <= 1.0)
        ntraining = ceil(nfiles * ntraining);
        ntesting = nfiles - ntraining;
    end
    
    if (~exist('ntesting', 'var') && ntraining > 1.0)
        ntesting = nfiles - ntraining;
    end
        
    
    if (nfiles < ntraining + ntesting)
        error('Number of testing and training files exceeds number of files available...');
    end
    
    randind = randperm(nfiles);
    trf = files(randind(1:ntraining));
    tef = files(randind(ntraining+1 : ntraining+ntesting));

    trs = [];
    for i=1:length(trf)
        trs = [trs read_sketch(trf{i})]; %#ok<AGROW>
    end

    tes = [];
    for i=1:length(tef)
        tes = [tes read_sketch(tef{i})]; %#ok<AGROW>
    end
    
end

function has = has_pattern(filename, patterns)
    has = false;
    for i=1:length(patterns)
        if (~isempty(strfind(filename, patterns{i})))
            has = true;
            break;
        end
    end
end

function filedir = addslash( filedir )
    
    if (filedir(end) ~= '/')
        filedir(end+1) = '/';
    end

end