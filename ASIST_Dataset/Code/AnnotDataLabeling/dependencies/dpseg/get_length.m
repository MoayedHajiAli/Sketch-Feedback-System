function length = get_length(coords)   
   
    if (isempty(coords))
        length = 0;
        return;
    end
    shifted_coords = [coords; 0 0];
    shifted_coords(1,:) = [];
    diff = shifted_coords - coords;
    diff (end,:)=[];
    length = sum(sqrt(diag(diff*diff')));
    
end

