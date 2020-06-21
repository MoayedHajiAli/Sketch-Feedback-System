function params = getparams(setname)

    set(1).name = 'C:\Users\KemalTugrul\Desktop\Dropbox\Bin\Study\Research\1003\SoftWorks\Modules\sketchBases\Annotated\14Partial';
    set(1).setdir = 'C:\Users\KemalTugrul\Desktop\Dropbox\Bin\Study\Research\1003\SoftWorks\Modules\sketchBases\Annotated\14Partial/';
    set(1).extension = {'.mat'};
    set(1).exceptions = {};
    set(1).douglasthresh = 10;
    
    

    for i=1:length(set)
        if (strcmp(set(i).name, setname))
            params = set(i);
        end
    end
    
    
    params.anglemeasure = 1:3;
    params.resample_interval = 3:5;    
    params.endwin = 3:5;   
    
end