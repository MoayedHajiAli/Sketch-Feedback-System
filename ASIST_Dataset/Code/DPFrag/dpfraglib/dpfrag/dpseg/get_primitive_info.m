function info = get_primitive_info(arg, type)


    table(1).type = 'line';
    table(1).label = 1;
    table(1).color = [1 0 0];
    
    table(2).type = 'arc';
    table(2).label = 2;
    table(2).color = [0 1 0];   
    
    table(3).type = 'spiral';
    table(3).label = 3;
    table(3).color = [0 0 1];  
    
    table(4).type = 'bezier';
    table(4).label = 2;
    table(4).color = [0 1 0];
    
    table(5).type = 'ellipse';
    table(5).label = 2;
    table(5).color = [0 1 0];
    
    table(6).type = 'circle';
    table(6).label = 2;
    table(6).color = [0 1 0];
    
    
    
    info = [];
    switch type
        case 'label'
            for i=1:length(table)
                if (strcmp(table(i).type, arg))
                    info = table(i).label;
                    return;
                end
            end
            
        case 'color'
            for i=1:length(table)
                if (table(i).label == arg)
                    info = table(i).color;
                    return;
                end
            end
            
            
    end
    
    if (isempty(info))
        exception = MException('VerifyOutput:OutOfBounds', ...
                               ['Requested info not found in table : ' arg ',' type]);
        throw(exception);           
    end
    
end

