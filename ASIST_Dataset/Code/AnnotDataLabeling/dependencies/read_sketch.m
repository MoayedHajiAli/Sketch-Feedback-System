function strokes = read_sketch(xmlfile)
    
    % load dat file if exists
    matfile = [xmlfile(1:end-3) 'mat'];
    if (exist(matfile, 'file'))
        eval(['load ' char(39) matfile char(39)]);
%         strokes = stroke; % rename in mat files later
        if (isfield(strokes, 'fragpts'))
            strokes = rmfield(strokes, 'fragpts');
        end
        if (isfield(strokes, 'dppoints'))
            strokes = rmfield(strokes, 'dppoints');
        end        
        return;
    end

    try
        V = xml_parseany(fileread(xmlfile));
    catch exception
        throw(exception)
    end

    for i=1:length(V.stroke)
        strokes(i) = struct('pids',[],  ...         % point ids 
                            'coords', [], ...       % coordinates
                            'times', [], ...        % times of each point in array
                            'primids', [], ...      % pirimitive id of each point
                            'primtypes', [] , ...   % types of the primitives
                            'corners', [], ...      % index of connections of two primitives
                            'npts', [], ...         % number of points
                            'nprims', []);          % distinct primitive ids
    end
    
    % point ids per stroke
    for i=1:length(V.stroke)
        strokes(i).pids = []; %#ok<*AGROW>
        for j=1:length(V.stroke{i}.arg)
            puid = V.stroke{i}.arg{j}.CONTENT;
            strokes(i).pids = [strokes(i).pids; puid];
        end
        strokes(i).npts = size(strokes(i).pids, 1);
    end

    % all point ids
    allids = [];
    for i=1:length(V.point)
        p = V.point{i}.ATTRIBUTE;
        allids = [allids; p.id];
    end
    
    % point coordinates and times
    for i=1:length(strokes)
        strokes(i).coords = [];
        strokes(i).times = [];
        for j=1:strokes(i).npts                        
            point_index = get_point_index(strokes(i).pids(j,:), allids);
            p = V.point{point_index}.ATTRIBUTE;
            strokes(i).coords = [strokes(i).coords; str2double(p.x) str2double(p.y)];
            strokes(i).times = [strokes(i).times; str2double(p.time)];                
        end
    end
    
    % sort each stroke
    for i=1:length(strokes)
        strokes(i) = sort_in_time(strokes(i));
    end

    if (isfield(V, 'label'))
        
        % ids, type and primid per primitive
        for i=1:length(V.label)
            pids = [];
            for j=1:length(V.label{i}.arg)
                pids = [pids; V.label{i}.arg{j}.CONTENT];
            end
            prim(i).pids = pids;
            prim(i).type = get_primitive_info(V.label{i}.ATTRIBUTE.primitiveType, 'label');
            prim(i).id = i;
        end

        % primitive id and type per primitive grouped by strokes 
        for i=1:length(strokes)
            strokes(i).primids = zeros(strokes(i).npts,1);
            strokes(i).primtypes = zeros(strokes(i).npts,1);   
            for j=1:strokes(i).npts
                pid = strokes(i).pids(j,:);
                for k=1:length(prim)
                    if (any(ismember(prim(k).pids, pid, 'rows')))
                        strokes(i).primids(j) = prim(k).id;
                        strokes(i).primtypes(j) = prim(k).type;
                        break;
                    end
                end
            end
            strokes(i).nprims = unique(strokes(i).primids);
            strokes(i).corners = find(conv(strokes(i).primids, [1 -1], 'valid'));
            strokes(i).corners = [1;strokes(i).corners;strokes(i).npts];
        end    
        
    end
        
    % pids no longer needed
    for i=1:length(strokes)
        strokes2(i) = rmfield(strokes(i), 'pids');
    end
    strokes = strokes2;
    
    % check validity
    if (~valid(strokes))
        exception = MException('VerifyOutput:OutOfBounds', ...
                               ['xml file is not valid : ' xmlfile]);
        throw(exception);
    else
%         eval(['save ' char(39) matfile char(39) ' strokes']); 
    end
    
    
end

function v = valid(strokes)
    v = true;
    for i=1:length(strokes)
        v = length(strokes(i).times) == strokes(i).npts;
        if (isempty(strokes(i).primids))
            continue;
        end
        v = v & length(strokes(i).primids) == strokes(i).npts;
        v = v & length(strokes(i).primtypes) == strokes(i).npts;
        v = v & all(strokes(i).primids);
        v = v & all(strokes(i).primtypes);
    end
end


function index = get_point_index(id, pids)
    
    index = find(ismember(pids, id, 'rows'));
    if (length(index) ~= 1)
        exception = MException('VerifyOutput:OutOfBounds', ...
                               ['point id not found or not unique : ' puid]);
        throw(exception);    
    end        
    
end


function stroke = sort_in_time(stroke)

    % make the sort stable (if time is same for two points they stay in same position)
    step = 1/(length(stroke.times)+1);
    increments = 0:step:1;    
    % make the increments same size as stroke.times
    increments = increments(1:stroke.npts);    
    time = stroke.times + increments';    
    [v index] = sort(time, 'ascend');  %#ok<ASGLU>    
    % set the start time to 0.
    stroke.times = stroke.times(index) - stroke.times(index(1)) ;
    stroke.coords = stroke.coords(index, :);
    stroke.pids = stroke.pids(index,:);
    
end