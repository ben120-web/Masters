%function generatingNoisyEcgDatabase(noiseSignalPath, ...
    ecgSignalPath, ecgFs, maxNosieSections, SNR, ...
    numberOfGeneratedNoisySignals, googleDriveFolderIDClean, snrFolderIDs)
    % generatingNoisyEcgDatabase - Models new noise sections, then corrupts the
    % clean ECG's with pre-defined SNR levels of electrode motion noise.
    %
    % Syntax: generatingNoisyEcgDatabase(noiseSignalPath, ...
    %    ecgSignalPath, ecgFs, maxNosieSections, SNR, ...
    %    numberOfGeneratedNoisySignals, googleDriveFolderIDClean, snrFolderIDs)
    %
    % Inputs:
    %    noiseSignalPath - Directory to real noise .mat file.
    %    ecgSignalPath - Directory to all clean ECG signals.
    %    ecgFs - Sampling frequency of ECG signals.
    %    maxNosieSections - Maximum number of noise segments per ECG file.
    %    SNR - Array containing the required signal to noise ratios.
    %    numberOfGeneratedNoisySignals - The number of noisy signals to
    %    generate from each clean ECG signal.
    %    googleDriveFolderIDClean - Google Drive Folder ID to save the clean files.
    %    snrFolderIDs - Struct containing Google Drive Folder IDs for SNR levels.
    %
    % Outputs: none.
    %
    % Other m-files required: noiseSignalModeling.m.
    % Subfunctions: computeRmsNoiseAmp, generateInitialTable, uploadToGoogleDrive.
    % MAT-files required: none.

    %------------- BEGIN CODE --------------
    %% Set constants
    QRS_SEARCH_WINDOW = 0.05; % [s]
    ECG_LENGTH_SECONDS = 30;

    % Get the information from the noise signals folder.
    noiseSignalDirInfo = dir(fullfile(noiseSignalPath, '*mat'));
    ecgSignalDirInfo = dir(fullfile(ecgSignalPath, '*mat'));

    % Read stored noise signals from mat file.
    initialNoiseData = load(fullfile(noiseSignalPath, noiseSignalDirInfo(1).name));

    % Calculate the minimum and maximum length of ecg signals which will be
    % used for noise generation.
    acceptableEcgLength500 = ECG_LENGTH_SECONDS * ecgFs;

    lengthOfNoiseSignal = numel(initialNoiseData.emNoise);

    % Define the noise signal.
    noiseSignal = initialNoiseData.emNoise;

    % Get the starting index of each noise section.
    noiseSectionStart = 1:acceptableEcgLength500:lengthOfNoiseSignal;

    % Generate all the noise signals for each section of data. These signals
    % are not scaled with respect to SNR right now.
    for iNoiseSection = 1:maxNosieSections

        noiseSectionEnd = noiseSectionStart(iNoiseSection) + ...
            acceptableEcgLength500 - 1;

        % All noise signals for this section.
        thisNoiseSection = noiseSignal(noiseSectionStart(iNoiseSection): ...
            noiseSectionEnd);

        initialGenerateNoise.("EM_Noise"){iNoiseSection, 1} = noiseSignalModeling(...
            thisNoiseSection, ...
            numberOfGeneratedNoisySignals);

    end

    % Filter out meta data 
    filterFlag = strcmp({ecgSignalDirInfo.name}, "AngleData.mat") | ...
        strcmp({ecgSignalDirInfo.name}, "widthData.mat") | ...
        strcmp({ecgSignalDirInfo.name}, "zData.mat");

    ecgSignalDirInfo(filterFlag) = [];

    % Number of valid ecg files.
    nEcgFiles = numel(ecgSignalDirInfo);

    % Get the length of SNR vector.
    nSNR = numel(SNR);

    % Process each valid ecg file.
    for iEcgFile = 1:nEcgFiles

        ecgSignalFileName = fullfile(ecgSignalPath, ...
            ecgSignalDirInfo(iEcgFile).name);

        TempData = load(ecgSignalFileName);
        tempDataFieldNames = fieldnames(TempData);

        % If the file contains one variable it will be taken as the ecg
        % signal.
        rawEcgSignal = TempData.(tempDataFieldNames{1});

        % Trim this signal to be 30 seconds in duration.
        rawEcgSignal = rawEcgSignal(1:acceptableEcgLength500);

        % Initial data table. One table will be generated for each valid
        % file and stored in a separate file in result folder.
        [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections);

        % Make sure the ecg signal is in column orientation and store the
        % data and file name in the data table.
        rawEcgSignal = rawEcgSignal(:);
        DataTable.ecgSignal(1, 1) = {rawEcgSignal};
        DataTable.FileName(1, 1) = ecgSignalDirInfo(iEcgFile).name(1:end-4);

        % Get the length of the current ecg signal.
        lengthOfThisEcgSignal = numel(rawEcgSignal);

        % Detect the r-peak locations.
        qrsLocations = TempData.(tempDataFieldNames{1}).qrsPeaks;

        % Trim this to 30 seconds of data
        qrsValidFlag = find(qrsLocations <= lengthOfThisEcgSignal);
        qrsLocations = qrsLocations(1:numel(qrsValidFlag));

        % Convert the qrs search window to samples. This window will be used to
        % calculate the qrs amplitude.
        qrsSearchWindow = round(QRS_SEARCH_WINDOW * ecgFs);

        % Pre-allocate memory.
        nQrsLocations = numel(qrsLocations);
        qrsPeakToPeak = nan(nQrsLocations, 1);

        % Calculate the peak to peak amplitude of the qrs complexes.
        for jQrsLocations = 1:nQrsLocations

            if (qrsLocations(jQrsLocations) + qrsSearchWindow <= lengthOfThisEcgSignal) && ...
               (qrsLocations(jQrsLocations) - qrsSearchWindow >= 1)

                qrsAmpMax = max(rawEcgSignal(qrsLocations(jQrsLocations) - qrsSearchWindow: ...
                                             qrsLocations(jQrsLocations) + qrsSearchWindow));
                qrsAmpMin = min(rawEcgSignal(qrsLocations(jQrsLocations) - qrsSearchWindow: ...
                                             qrsLocations(jQrsLocations) + qrsSearchWindow));
                qrsPeakToPeak(jQrsLocations, 1) = qrsAmpMax - qrsAmpMin;
            end

        end

        % Convert the peak to peak qrs amplitude to power. This is the correct
        % transformation for a sine wave. In this instance, it approximates close
        % enough for measurement. See the PDF of ecg SNR report for derivation.
        qrsPeakToPeakPower = (mean(qrsPeakToPeak, 'omitnan') ^ 2) / 8;

        % Now using the qrs peak to peak power all the noises in each
        % section will be scaled for every SNR level.
        for kSNR = 1:nSNR

            thisNoise = initialGenerateNoise.("EM_Noise");

            for mSection = 1:maxNosieSections

                thisNoiseSection = thisNoise{mSection, 1};

                for iGenSignal = 1:numberOfGeneratedNoisySignals + 1

                    thisNoiseSignal = thisNoiseSection{iGenSignal, 1};
                    thisNoiseSignal = thisNoiseSignal(1:lengthOfThisEcgSignal);

                    % Calculate power of noise signals.
                    thisNoiseSignalPower = computeRmsNoiseAmp(thisNoiseSignal) ^ 2;

                    % Calculate the scale factor which is required to
                    % achieve the desired SNR level with each noisy
                    % signal.
                    scaleFactor = sqrt(qrsPeakToPeakPower / ...
                        (thisNoiseSignalPower * (10 ^ (SNR(kSNR) / 10))));

                    scaledSignal = thisNoiseSignal * scaleFactor;

                    % Save the noise corrupt signals in the data table.
                    DataTable.(['SNR', num2str(SNR(kSNR))])(mSection, 1). ...
                        (string(noiseNames)){iGenSignal, 1} = ...
                        scaledSignal + rawEcgSignal;

                end

            end

        end

        % Call function to convert and save each signal to a readable format for
        % deep learning models and upload to Google Drive.
        convertToHDF5Format(DataTable, googleDriveFolderIDClean, snrFolderIDs);

    end

end

function convertToHDF5Format(DataTable, googleDriveFolderIDClean, snrFolderIDs)
    % Save all signals to a more readable format and upload to Google Drive

    % Constants.
    NUM_OF_SNR = 5;

    % Extract the filename.
    fileName = table2array(DataTable(1, 1));

    % Extract the clean signal.
    cleanSignal = DataTable.ecgSignal{1, 1};

    % Ensure the temp directory is valid and accessible
    tempDirectory = tempdir;
    if ~isfolder(tempDirectory)
        error('Temporary directory not found: %s', tempDirectory);
    end

    % Save the clean signal locally in H5 format.
    cleanFilePath = fullfile(tempDirectory, [fileName, '.h5']);
    h5create(cleanFilePath, '/ecgSignal', size(cleanSignal));
    h5write(cleanFilePath, '/ecgSignal', cleanSignal);

    % Upload the clean signal to Google Drive
    uploadToGoogleDrive(cleanFilePath, googleDriveFolderIDClean);

    % Now we need to loop through the noisy signals.
    for iSection = 1:height(DataTable)

        % Extract the section number 
        sectionNumber = iSection;

        % Loop through each SNR value.
        for iSNR = 1:NUM_OF_SNR

            % Index into corresponding SNR column
            SNRValue = ['SNR', num2str((iSNR-1) * 6)];

            % Extract the Data (nested in structs)
            SNRData = table2array(DataTable(:, iSNR + 3));

            % Grab the section specific noise structure.
            sectionNoise = SNRData(iSection).em;

            % Loop through each copy (From AR-Modelling)
            for iCopy = 1:numel(sectionNoise)

                % Specify the noise type.
                if iCopy == 1
                    noiseType = 'Original';
                else
                    noiseType = ['Copy', num2str(iCopy)];
                end

                % Extract the noisy ECG signal.
                noisyEcg = cell2mat(sectionNoise(iCopy));

                % Ensure the temp directory is valid and accessible
                if ~isfolder(tempDirectory)
                    error('Temporary directory not found: %s', tempDirectory);
                end

                % Let's create a filename based on the type of signal.
                outputFilename = fullfile(tempDirectory, ...
                    strcat(erase(fileName, '_cleanSignal'), '-', ...
                    noiseType, '-', num2str(sectionNumber), '.h5'));

                % Save the noisy signal locally in h5 format.
                h5create(outputFilename, '/ecgSignal', size(noisyEcg));
                h5write(outputFilename, '/ecgSignal', noisyEcg);

                % Upload the noisy signal to Google Drive
                uploadToGoogleDrive(outputFilename, snrFolderIDs.(SNRValue));
            end
        end
    end

end

function uploadToGoogleDrive(filePath, folderID)
    % uploadToGoogleDrive - Uploads a file to a specified Google Drive folder.
    %
    % Syntax: uploadToGoogleDrive(filePath, folderID)
    %
    % Inputs:
    %    filePath - Local path to the file.
    %    folderID - Google Drive Folder ID where the file should be uploaded.
    %
    % Outputs: none.

    %------------- BEGIN CODE --------------
    % Read OAuth 2.0 credentials from JSON file
    credentialsFile = '/Users/benrussell/Frameworks/Google-Drive/client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c.apps.googleusercontent.com.json'; % Update this path

    if ~isfile(credentialsFile)
        error('Credentials file not found: %s', credentialsFile);
    end
    
    fid = fopen(credentialsFile, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    credentials = jsondecode(str);

    % Authenticate and get access token
    refreshToken = getStoredRefreshToken();
    accessToken = getAccessTokenUsingRefreshToken(credentials, refreshToken);

    % Read the file content
    fid = fopen(filePath, 'rb');
    if fid == -1
        error('Cannot open file: %s', filePath);
    end
    fileContent = fread(fid, '*uint8');
    fclose(fid);

    % Define the metadata for the file
    [~, fileName, ext] = fileparts(filePath);
    fileName = strcat(fileName, ext);

    metadata = struct( ...
        'name', fileName, ...
        'parents', { {folderID} });

    % Convert metadata to JSON format
    metadataJson = jsonencode(metadata);

    % Create a multipart/form-data body for the HTTP request
    boundary = '-------314159265358979323846';
    CRLF = [char(13) char(10)];
    body = [ ...
        '--' boundary CRLF ...
        'Content-Type: application/json; charset=UTF-8' CRLF CRLF ...
        metadataJson CRLF ...
        '--' boundary CRLF ...
        'Content-Type: application/octet-stream' CRLF CRLF ...
        uint8(fileContent)' CRLF ...
        '--' boundary '--' CRLF];

    % Define HTTP header with authorization and content type
    headerFields = [
        matlab.net.http.field.GenericField('Authorization', ['Bearer ', accessToken]), ...
        matlab.net.http.field.ContentTypeField(['multipart/related; boundary="' boundary '"'])];

    % Define Google Drive API endpoint for uploading a file
    apiEndpoint = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart';

    % Upload the file
    request = matlab.net.http.RequestMessage('POST', headerFields, body);
    response = request.send(matlab.net.URI(apiEndpoint));

    if response.StatusCode == 200 || response.StatusCode == 201
        fileResponse = jsondecode(response.Body.Data);
        fileID = fileResponse.id;
        % Display the uploaded file ID
        fprintf('Uploaded file ID: %s\n', fileID);
    else
        % Enhanced debugging output
        fprintf('Failed to upload file. Status Code: %d\n', response.StatusCode);
        fprintf('Response: %s\n', char(response.Body.Data));
        error('Failed to upload file: %s', char(response.Body.Data));
    end
end

function [noiseAmp] = computeRmsNoiseAmp(noiseSection)
    % Computes the Root-Mean-Squared amplitude of a signal by measuring the RMS
    % value for each second and then discarding the top and bottom 5% of
    % values.

    % In this function, sampling rate is fixed at 500 Hz.
    SIGNAL_FS = 500; % [Hz]

    % Get the length of the signal in complete seconds.
    nSignalSeconds = floor(numel(noiseSection) / SIGNAL_FS);

    % Calculate RMS value for each second.
    rmsVals = nan(nSignalSeconds, 1);

    for iSec = 1:nSignalSeconds

        % Get indexes for start and end of section.
        thisSectionStart = (iSec - 1) * SIGNAL_FS + 1;
        thisSectionEnd = iSec * SIGNAL_FS;

        % Grab the section from the noise signal.
        thisSignalSection = noiseSection(thisSectionStart:thisSectionEnd);

        % Compute the RMS amplitude difference between the mean of the section
        % and the section itself.
        rmsVals(iSec) = rms(thisSignalSection - mean(thisSignalSection));

    end

    % Discard highest and lowest 5% of values.
    noiseAmp = trimmean(rmsVals, 5, 'round');

end

function [noiseNames, DataTable] = generateInitialTable(nSNR, SNR, maxNosieSections)
    % Create initial table.

    variableNames = {'FileName', 'ecgSignal', 'SectionNum', 'SNR0'};
    fileName = "";
    ecgSignal = cell(1, 1);
    noiseStructure.em = {};

    noiseNames = fieldnames(noiseStructure);
    SectionNum = 1;
    DataTable = table(fileName, ecgSignal, SectionNum, noiseStructure, ...
        'VariableNames', variableNames);

    nInitialVariableNames = numel(variableNames);

    for i = 2:nSNR
        variableNames{1, i + nInitialVariableNames - 1} = ['SNR', num2str(SNR(i))];
    end

    DataTable(1, nInitialVariableNames + 1:nSNR + nInitialVariableNames - 1) = DataTable(1, 4);
    DataTable.Properties.VariableNames = variableNames;
    DataTable(maxNosieSections, :) = DataTable(1, :);
    DataTable.SectionNum(1:end, 1) = 1:maxNosieSections;
    DataTable.FileName(:, :) = "";

end