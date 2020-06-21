function [] = xmlToPNG(xml_path,img_path)
%%% save png files from xml_path
%xml_path = 'C:\Users\generic\Desktop\xml';
%img_path = 'C:\Users\generic\Desktop\img';

clc;
close all;

files = getAllFiles(xml_path);


parfor i=1:length(files)
    
	if(~strcmp( files{i}(end-3:end), '.xml') )
		fprintf('Continued... its not an .xml : %d\n', i);
		continue;
	end
	
	lastSlashIdx = findstr(files{i}, '\') + 1;
	name = files{i}(lastSlashIdx:end);
	savename = [name(1:end-3) 'png'];
	
	
	if( length( unique( ismember(files, [img_path '\' savename]) ) ) > 1 )
		%[img_path '\' savename]
		fprintf('File %d s png exists\n', i);
		%files{ ismember(files, [img_path '\' savename]) }
		continue;
	end
	 
    mat = get_strokes(files{i});
    
    f = figure('visible','off');
    colors = hsv(length(mat));
    for j=1:length(mat)
        plot(mat(j).coords(:,1),mat(j).coords(:,2), 'color', colors(j,:), 'LineWidth',3);
        hold on
        axis ij
        axis off
        axis equal
    end
    hold off
    %name = files(i,1).name;
    
    img = strcat(img_path,'\',savename);
    saveas(f,img);
	close all;
	
	fprintf('File: %d of %d\n', i, length(files));
end

end