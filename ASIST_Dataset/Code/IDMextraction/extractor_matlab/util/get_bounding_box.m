function bounding_box = get_bounding_box(coords)    

    bounding_box.x1 = min(coords(:,1));
    bounding_box.y1 = min(coords(:,2));
    bounding_box.x2 = max(coords(:,1));
    bounding_box.y2 = max(coords(:,2));
    bounding_box.diagonal_length = get_distance([bounding_box.x1 bounding_box.y1],... 
                                                [bounding_box.x2 bounding_box.y2]);
    
    bounding_box.height = abs(bounding_box.y1 - bounding_box.y2);
    bounding_box.width = abs(bounding_box.x1 - bounding_box.x2);
    
    bounding_box.area = bounding_box.height*bounding_box.width;
    
end
