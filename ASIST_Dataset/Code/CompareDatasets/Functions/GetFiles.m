function files = GetFiles(rootFolder, isExtension, templateText)

	allFiles = dir( rootFolder );

	filesIdx = 1;
	files = {};

	for i = 1: length(allFiles)

		if( allFiles(i).isdir )
			continue;
		end

		if( isExtension ) % Suffix

			if( length(allFiles(i).name) < length(templateText) )
				continue;
			end

			if(~strcmp(allFiles(i).name(end - length(templateText) + 1 : end), templateText ) )
				continue;
			else
				files{filesIdx} = allFiles(i).name;
				filesIdx = filesIdx + 1;
			end

		else % Prefix

			if( length(allFiles(i).name) < length(templateText) )
				continue;
			end

			if(~strcmp(allFiles(i).name(1 : length(templateText)), templateText ) )
				continue;
			else
				files{filesIdx} = allFiles(i).name;
				filesIdx = filesIdx + 1;
			end

		end

	end


end
