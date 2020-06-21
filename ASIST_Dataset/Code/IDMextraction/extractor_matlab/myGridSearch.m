function [bestc bestg] = myGridSearch(myOutput, myInput)

%log2c = -5:15
%log2g = -15:3
bestcv = 0;
for log2c = -1.1:3.1,
  for log2g = -4.1:1.1
    cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' -b 1'];
    cv = svmtrain(myOutput, myInput, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    %fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end