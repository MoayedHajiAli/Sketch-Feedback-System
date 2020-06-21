function ClassHist(Labels, ClassNames)

  NonNegClasses = unique( Labels( find(Labels > 0) ) );

  if(length(find(Labels == -1)) > 0)
    fprintf('(-1) Garbage:--------------#: %d...\n', length( find(Labels == -1) ) );
  end

  for cl = 1 : length( NonNegClasses )
    fprintf('(%d) %s:--------------#: %d...\n', NonNegClasses(cl), ClassNames{ NonNegClasses(cl) }, length( find(Labels == NonNegClasses(cl))));
  end

end
