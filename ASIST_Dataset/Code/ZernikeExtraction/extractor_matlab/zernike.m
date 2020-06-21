function feat = zernike(coords, order)
    
    feat = [];    
    Rcoef = get_coeff(order);
    M00 = 1/pi*length(coords);
    centroid = get_centroid(coords);
    maxdist = get_outlier(coords);
    % n starts from 2 since A00 and A11 are the same for all normalised symbols
    for n=2:order
        for m= mod(n,2):2:n
            vsum = [0 0];
            for i=1:length(coords)
                c = (coords(i,:) - centroid) / maxdist;
                v = V(n,m,c(1),c(2), Rcoef);
                vsum = vsum + v;
            end
            feat(end+1) = (n+1)/pi*sqrt(sum(vsum.*vsum))/M00;            
        end
    end				    
end


function res = R(n,m,rho,Rcoef)
    res = 0;
    for s=0:(n-m)/2
        res = res + Rcoef(n+1,m+1,s+1) * rho^(n-2*s);
    end
end

function res = V(n,m,x,y,Rcoef)
    
    r = R(n,m,sqrt(x*x+y*y), Rcoef);
    theta = atan2(y,x);
    res = [r*cos(m*theta) r*sin(m*theta)];
end

function Rcoef = get_coeff(order)

    Rcoef = zeros(order+1, order+1, floor((order+1)/2)+1);    
    for n=0:order 
        for m = mod(n,2):2:n  
			sign = 1;
			for s=0:floor((n-m)/2)
                Rcoef(n+1,m+1,s+1) = sign * factorial(n-s) / ( factorial(s) * factorial(floor((n+m)/2)-s) * factorial( floor((n-m)/2)-s));
% 				display(sprintf('%d, %d, %d, %d', n, m, s, Rcoef(n+1,m+1,s+1)));
                sign = sign*-1;
            end
        end
    end   
    
end