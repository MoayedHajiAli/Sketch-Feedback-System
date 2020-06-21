function annot(trf)
    close all;
    annotated = annot_main(trf)
end

function annotated = annot_main(trf)

    annotated = 0;
    for i=1:length(trf)
        curr_file = trf{i};
        strokes = get_strokes(curr_file);
        strokes = douglas_peucker(strokes, 3);
        unannot = [];
        for j=1:length(strokes)
            close all;figure;
            
            for k=1:length(strokes)
                plot_stroke(strokes(k));
            end
            hold on;
            plot(strokes(j).coords(:,1), -strokes(j).coords(:,2), 'c-');
            coords = strokes(j).coords(strokes(j).dppoints,:);    
            for k=1:size(coords,1)
                text(coords(k,1), -coords(k,2)+1, num2str(k),'Color',[1,0.4,0.6],'fontSize',15);
            end            
            set(gcf, 'Position',[100 100 900 700]);
            prompt = {'Enter node sequence:','Enter primitives:'};
            dlg_title = 'Input for fragmentation';
            num_lines = 1;
            def = {'',''};
            answer = inputdlg(prompt,dlg_title,num_lines,def);
            if (isempty(answer))
                unannot = [unannot j];
                continue;
            end
            corners = textscan(answer{1}, '%d', 'Delimiter', ',');
            corners = corners{1};
            primitives = textscan(answer{2}, '%d', 'Delimiter', ',');
            primitives = primitives{1};
            strokes(j).corners = strokes(j).dppoints(corners);
            strokes(j).primids = zeros(strokes(j).npts,1);
            strokes(j).primtypes = zeros(strokes(j).npts,1);
            for l=1:length(strokes(j).corners)-1
                strokes(j).primids(strokes(j).corners(l):strokes(j).corners(l+1)) = l;
                strokes(j).primtypes(strokes(j).corners(l):strokes(j).corners(l+1)) = primitives(l);
            end
            strokes(j).nprims = 1:length(primitives);      
            hold off;
        end
        close;
        % delete unannotated strokes
        strokes(unannot) = [];
        matfile = [curr_file(1:end-4) '_lab.mat'];
        eval (['save ' char(39) matfile char(39) ' strokes']); 
        result = get_strokes(matfile);
        for k=1:length(strokes)            
            plot_stroke(result(k));
        end
        title('Resulting Corners');
        answer = inputdlg({'Is it true?'},'t/f',1,{'t'});
        if (isempty(answer) || ~strcmp(answer{1},'t'))
            eval(['delete ' char(39) matfile char(39)]);  
        else
            annotated = annotated+1;
            display(sprintf('Number of annotated strokes : %d', annotated));
        end
        close;
    end
end

