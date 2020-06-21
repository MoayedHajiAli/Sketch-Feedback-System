% Important Note:
% I added the isDebug flag in the parameter. Simply make it true to allow
% plotting
function [imcell] = get_combinations(primitiveNum,difCol,X,Y,T,isDebug)
    isDebug = false; %Kludge
    c = difCol;
    m = 1;
    for i=1:c-primitiveNum
        if(isDebug)
            hold off;
            figure(primitiveNum*1000+i);
        end
        
        for j = 0:primitiveNum-1
            if(isDebug)
                plot(X{i+j},-Y{i+j},'black.','LineWidth',3)
            end
            XX{m}=X{i+j};
            YY{m}=Y{i+j};
            TT{m}=T{i+j};
            m=m+1;
            if(isDebug)
                hold on;   
            end
        end
        imcell{i}.X = XX;
        imcell{i}.Y = YY;
        imcell{i}.T = TT;      
        XX = [];
        YY = [];
        TT = [];
        if(isDebug)
            axis equal;
        end
    end
end