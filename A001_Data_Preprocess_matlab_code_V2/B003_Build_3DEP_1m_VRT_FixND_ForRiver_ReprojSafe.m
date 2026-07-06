function Summary = B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe(riverName, varargin)
% B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe
% Revised 2026-07-03
%
% REPROJECT_SAFE_VERSION_MARKER: this version detects mixed SRS / mixed UTM zones and reprojects 3DEP tiles to Bathy_1m_FixND SRS before building VRT.
% Purpose
%   Build 3DEP 1 m DEM VRTs from downloaded DEM_1m_raw files, then create
%   a unified NoData VRT for later resampling/clipping to the bathymetry grid.
%
% Key revision
%   Some downloaded 3DEP tiles may be in different UTM zones for the same river
%   extent. Example: MD Potomac contains both UTM 17N and UTM 18N tiles.
%   A direct gdalbuildvrt would skip heterogeneous projection tiles and produce
%   an incomplete DEM VRT. This version checks the source SRS groups and, when
%   needed, reprojects all source 3DEP tiles to the corresponding bathy 1 m SRS
%   before building the formal VRT.
%
% Input
%   USGS_3DEP_bathymetry_DEM/<river>/DEM_1m_raw/*.tif or *.tiff
%   For Milwaukee children, all four river sections share:
%   USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/DEM_1m_raw
%
% Output
%   Processed_Results/3DEP_1m_VRT/<demName>/DEM_3DEP_1m.vrt
%   Processed_Results/3DEP_1m_VRT/<demName>/DEM_3DEP_1m_FixND.vrt
%
% Also writes / refreshes:
%   USGS_3DEP_bathymetry_DEM/<demName>/DEM_1m_raw/Filelist.txt
%   Processed_Results/3DEP_1m_VRT/<demName>/SRS_report.txt
%   Processed_Results/3DEP_1m_VRT/<demName>/Filelist_reproj_to_bathy_srs.txt, if reprojection is triggered
%
% Usage examples
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('MD_PotomacRiver_Bathy_2019');
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('CA_KlamathRiver_TopoBathy_2018_D18', 'numThreads', 4);
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('milwaukee_river_3DEP');
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('BadgerFinNull');  % maps to milwaukee_river_3DEP
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('ALL');
%   B003_Build_3DEP_1m_VRT_FixND_ForRiver_ReprojSafe('LIST');
%
% Options
%   'reprojectToBathySRS', 'auto'   default; reproject only if source SRS is heterogeneous or differs from bathy SRS
%   'reprojectToBathySRS', true     always reproject to bathy SRS
%   'reprojectToBathySRS', false    never reproject; not recommended for mixed UTM rivers
%   'targetBathyRiver', '<river>'   manually choose bathy SRS reference; useful for milwaukee_river_3DEP if needed
%
% Design notes
%   - This step does NOT download 3DEP data.
%   - This step does NOT perform final resampling/clipping to the bathy grid. That should be B004.
%   - This step organizes downloaded 3DEP files into a projection-consistent VRT and normalizes NoData metadata to -999999.
%   - The raw folder remains an input source. The formal product is under Processed_Results/3DEP_1m_VRT.

if nargin < 1 || isempty(riverName)
    riverName = 'LIST';
end

cfg = defaultConfig();
cfg = parseOptions(cfg, varargin{:});

setupGdalPaths();
ensureDir(cfg.outRoot);
ensureDir(cfg.logRoot);

validBathyRivers = selectedBathyRivers();
validDemTasks    = selectedDemTasks();

if ischar(riverName) || isstring(riverName)
    riverName = char(riverName);
    if strcmpi(riverName, 'LIST')
        printLists(validBathyRivers, validDemTasks);
        Summary = table();
        return;
    elseif strcmpi(riverName, 'ALL')
        tasksToRun = validDemTasks;
    else
        tasksToRun = {mapRiverToDemName(riverName)};
        if isempty(cfg.targetBathyRiver) && ~strcmp(tasksToRun{1}, riverName)
            cfg.targetBathyRiver = char(riverName);
        end
    end
elseif iscell(riverName)
    tasksToRun = cell(size(riverName));
    for k = 1:numel(riverName)
        tasksToRun{k} = mapRiverToDemName(riverName{k});
    end
else
    error('riverName must be char/string, cell array, ''ALL'', or ''LIST''.');
end

% unique while preserving order
[~, ia] = unique(tasksToRun, 'stable');
tasksToRun = tasksToRun(sort(ia));

for k = 1:numel(tasksToRun)
    if ~ismember(tasksToRun{k}, validDemTasks)
        fprintf('\nUnknown 3DEP task/demName: %s\n', tasksToRun{k});
        printLists(validBathyRivers, validDemTasks);
        error('Input is not in selected valid 3DEP task list.');
    end
end

fprintf('\n============================================================\n');
fprintf('B003 build 3DEP 1m VRT + FixND\n');
fprintf('Raw root   : %s\n', cfg.rawRoot);
fprintf('Output root: %s\n', cfg.outRoot);
fprintf('Tasks      : %s\n', strjoin(tasksToRun, ', '));
fprintf('Overwrite  : %d\n', cfg.overwrite);
fprintf('RebuildList: %d\n', cfg.rebuildList);
fprintf('Unzip first: %d\n', cfg.unzipFirst);
fprintf('GDAL thread: %s\n', cfg.numThreads);
fprintf('Reproject  : %s\n', optionToString(cfg.reprojectToBathySRS));
fprintf('============================================================\n');

summaryRows = {};
rowId = 0;

for i = 1:numel(tasksToRun)
    demName = tasksToRun{i};

    [demRawDir, demSourceNote] = findDemRawDir(cfg, demName);
    outDir = fullfile(cfg.outRoot, demName);
    ensureDir(outDir);

    listFile = fullfile(demRawDir, 'Filelist.txt');
    srsReportFile = fullfile(outDir, 'SRS_report.txt');
    vrtRaw   = fullfile(outDir, 'DEM_3DEP_1m.vrt');
    vrtFixND = fullfile(outDir, 'DEM_3DEP_1m_FixND.vrt');

    fprintf('\n============================================================\n');
    fprintf('[%d/%d] 3DEP task: %s\n', i, numel(tasksToRun), demName);
    fprintf('DEM raw dir : %s\n', demRawDir);
    fprintf('Source note : %s\n', demSourceNote);
    fprintf('Filelist    : %s\n', listFile);
    fprintf('Raw VRT     : %s\n', vrtRaw);
    fprintf('FixND VRT   : %s\n', vrtFixND);
    fprintf('============================================================\n');

    if exist(demRawDir, 'dir') ~= 7
        msg = sprintf('Missing DEM raw folder: %s', demRawDir);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            rowId = rowId + 1;
            summaryRows(rowId,:) = {demName, demRawDir, listFile, 0, '', vrtRaw, vrtFixND, 'MISSING_RAW_DIR', demSourceNote}; %#ok<AGROW>
            continue;
        end
    end

    if cfg.unzipFirst
        unzipDownloadedZipFiles(demRawDir);
    end

    if cfg.rebuildList || exist(listFile, 'file') ~= 2
        build3DEPFilelist(demRawDir, listFile);
    else
        fprintf('[INFO] Use existing Filelist.txt: %s\n', listFile);
    end

    nFiles = countLines(listFile);
    fprintf('[3DEP tif count] %d\n', nFiles);

    if nFiles == 0
        msg = sprintf('No 3DEP tif/tiff found under: %s', demRawDir);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            rowId = rowId + 1;
            summaryRows(rowId,:) = {demName, demRawDir, listFile, nFiles, '', vrtRaw, vrtFixND, 'EMPTY_LIST', demSourceNote}; %#ok<AGROW>
            continue;
        end
    end

    if ~cfg.overwrite && exist(vrtFixND, 'file') == 2
        fprintf('[SKIP] Existing FixND VRT and overwrite=false: %s\n', vrtFixND);
        rowId = rowId + 1;
        summaryRows(rowId,:) = {demName, demRawDir, listFile, nFiles, listFile, vrtRaw, vrtFixND, 'SKIPPED_EXISTING', demSourceNote}; %#ok<AGROW>
        continue;
    end

    % Build SRS report and determine whether direct VRT build is safe.
    srsInfo = buildSRSReport(listFile, srsReportFile);
    fprintf('[SRS report] %s\n', srsReportFile);
    fprintf('[SRS unique EPSG count] %d\n', numel(srsInfo.uniqueEPSG));
    for s = 1:numel(srsInfo.uniqueEPSG)
        fprintf('  %s : %d files\n', srsInfo.uniqueEPSG{s}, srsInfo.counts(s));
    end

    [targetBathyRiver, bathyRef] = chooseTargetBathyRef(cfg, demName);
    [targetSRSFile, targetEPSG, targetResX, targetResY, hasBathyRef] = prepareTargetSRS(cfg, outDir, bathyRef);

    if hasBathyRef
        fprintf('[Target bathy SRS] river=%s\n', targetBathyRiver);
        fprintf('[Target bathy VRT] %s\n', bathyRef);
        fprintf('[Target EPSG] %s\n', targetEPSG);
        fprintf('[Target resolution] %.12g / %.12g\n', targetResX, targetResY);
    else
        fprintf('[Target bathy SRS] Not available for this task. Direct VRT only unless reprojectToBathySRS=true.\n');
    end

    needReproj = decideNeedReproject(cfg, srsInfo, targetEPSG, hasBathyRef);
    listForVRT = listFile;
    reprojNote = 'direct_source_filelist';

    if needReproj
        if ~hasBathyRef
            error('Reprojection requested/needed, but target bathy reference is unavailable for task: %s', demName);
        end

        fprintf('\n[Action] Reproject 3DEP source tiles to bathy SRS before VRT build.\n');
        fprintf('This avoids gdalbuildvrt skipping mixed UTM-zone tiles.\n');

        reprojDir  = fullfile(outDir, 'DEM_1m_raw_reproj_to_bathy_srs');
        reprojList = fullfile(outDir, 'Filelist_reproj_to_bathy_srs.txt');
        ensureDir(reprojDir);

        reprojectSourceTilesToTargetSRS(listFile, reprojDir, reprojList, targetSRSFile, targetResX, targetResY, cfg);
        listForVRT = reprojList;
        reprojNote = sprintf('reprojected_to_bathy_srs_%s', targetBathyRiver);
        fprintf('[Reprojected tif count] %d\n', countLines(reprojList));
    else
        fprintf('\n[Action] Direct VRT build. Source SRS is consistent enough for this B003 step.\n');
    end

    status1 = buildRawVRT(listForVRT, vrtRaw);
    if status1 ~= 0
        rowId = rowId + 1;
        summaryRows(rowId,:) = {demName, demRawDir, listFile, nFiles, listForVRT, vrtRaw, vrtFixND, 'FAILED_BUILD_RAW_VRT', reprojNote}; %#ok<AGROW>
        error('gdalbuildvrt failed for %s.', demName);
    end

    printRasterInfo(vrtRaw, 'Raw 3DEP VRT');

    status2 = buildFixNDVRT(vrtRaw, vrtFixND, cfg);
    if status2 ~= 0
        rowId = rowId + 1;
        summaryRows(rowId,:) = {demName, demRawDir, listFile, nFiles, listForVRT, vrtRaw, vrtFixND, 'FAILED_FIXND_VRT', reprojNote}; %#ok<AGROW>
        error('gdalwarp FixND VRT failed for %s.', demName);
    end

    printRasterInfo(vrtFixND, 'FixND 3DEP VRT');

    rowId = rowId + 1;
    summaryRows(rowId,:) = {demName, demRawDir, listFile, nFiles, listForVRT, vrtRaw, vrtFixND, 'OK', reprojNote}; %#ok<AGROW>
end

Summary = rowsToTable(summaryRows);
summaryFile = fullfile(cfg.logRoot, sprintf('B003_Build_3DEP_1m_VRT_FixND_%s_%s.csv', safeName(strjoin(tasksToRun, '__')), datestr(now, 'yyyymmdd_HHMMSS')));
writeSummaryCSV(summaryFile, summaryRows);

fprintf('\n============================================================\n');
fprintf('B003 finished. Summary written:\n%s\n', summaryFile);
fprintf('Next step should use:\n  %s/<demName>/DEM_3DEP_1m_FixND.vrt\n', cfg.outRoot);
fprintf('============================================================\n');

end

%% ============================================================
% Local functions
% ============================================================

function cfg = defaultConfig()
    cfg.rawRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM';
    cfg.prRoot  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
    cfg.outRoot = fullfile(cfg.prRoot, '3DEP_1m_VRT');
    cfg.logRoot = fullfile(cfg.prRoot, 'Logs');

    cfg.globalND = -999999;
    cfg.overwrite = true;
    cfg.rebuildList = true;
    cfg.unzipFirst = true;
    cfg.stopOnMissing = true;

    % auto = reproject when source SRS groups are heterogeneous or differ from bathy SRS.
    cfg.reprojectToBathySRS = 'auto';
    cfg.targetBathyRiver = '';

    % Safer for multiple MATLAB sessions.
    cfg.numThreads = '4';
    cfg.warpMemoryMB = 1024;
end

function cfg = parseOptions(cfg, varargin)
    if mod(numel(varargin), 2) ~= 0
        error('Options must be name-value pairs.');
    end

    for i = 1:2:numel(varargin)
        key = lower(char(varargin{i}));
        val = varargin{i+1};
        switch key
            case 'rawroot'
                cfg.rawRoot = char(val);
            case 'prroot'
                cfg.prRoot = char(val);
                cfg.outRoot = fullfile(cfg.prRoot, '3DEP_1m_VRT');
                cfg.logRoot = fullfile(cfg.prRoot, 'Logs');
            case 'outroot'
                cfg.outRoot = char(val);
            case 'overwrite'
                cfg.overwrite = logical(val);
            case 'rebuildlist'
                cfg.rebuildList = logical(val);
            case 'unzipfirst'
                cfg.unzipFirst = logical(val);
            case 'stoponmissing'
                cfg.stopOnMissing = logical(val);
            case 'numthreads'
                if isnumeric(val)
                    cfg.numThreads = num2str(val);
                else
                    cfg.numThreads = char(val);
                end
            case 'warpmemorymb'
                cfg.warpMemoryMB = val;
            case 'reprojecttobathysrs'
                cfg.reprojectToBathySRS = val;
            case 'targetbathyriver'
                cfg.targetBathyRiver = char(val);
            otherwise
                error('Unknown option: %s', key);
        end
    end
end

function rivers = selectedBathyRivers()
    rivers = { ...
        'BadgerFinNull'
        'Estabrook_Combined'
        'KewaFix2Null'
        'Kletzch_Combined_UpMax3Null'
        'CA_KlamathRiver_TopoBathy_2018_D18'
        'CO_UpperColorado_Topobathy_1_2020'
        'MD_PotomacRiver_Bathy_2019'
        'NE_Niobrara_Topobathy_2018'
        'OR_MKRC_Topobathy_2021'
        'OR_SantiamRiverTB_Topobathy_1_D23'
        'WA_ChehalisRiverTB_Topobathy_1_D23'
        'WA_Nisqually_Bathymetric_2020'
        };
end

function tasks = selectedDemTasks()
    tasks = { ...
        'milwaukee_river_3DEP'
        'CA_KlamathRiver_TopoBathy_2018_D18'
        'CO_UpperColorado_Topobathy_1_2020'
        'MD_PotomacRiver_Bathy_2019'
        'NE_Niobrara_Topobathy_2018'
        'OR_MKRC_Topobathy_2021'
        'OR_SantiamRiverTB_Topobathy_1_D23'
        'WA_ChehalisRiverTB_Topobathy_1_D23'
        'WA_Nisqually_Bathymetric_2020'
        };
end

function demName = mapRiverToDemName(riverName)
    riverName = char(riverName);
    milwaukeeChildren = { ...
        'BadgerFinNull', ...
        'Estabrook_Combined', ...
        'KewaFix2Null', ...
        'Kletzch_Combined_UpMax3Null'};

    if any(strcmp(riverName, milwaukeeChildren))
        demName = 'milwaukee_river_3DEP';
    else
        demName = riverName;
    end
end

function printLists(validBathyRivers, validDemTasks)
    fprintf('\nSelected bathy rivers:\n');
    for i = 1:numel(validBathyRivers)
        fprintf('  %2d. %s\n', i, validBathyRivers{i});
    end
    fprintf('\n3DEP DEM build tasks:\n');
    for i = 1:numel(validDemTasks)
        fprintf('  %2d. %s\n', i, validDemTasks{i});
    end
    fprintf('\nMilwaukee note: BadgerFinNull / Estabrook_Combined / KewaFix2Null / Kletzch_Combined_UpMax3Null share milwaukee_river_3DEP.\n\n');
end

function setupGdalPaths()
    try
        cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
        GDALLoad();
        addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
    catch ME
        warning('GDALLoad/setup path warning: %s', ME.message);
    end
end

function ensureDir(p)
    if exist(p, 'dir') ~= 7
        mkdir(p);
    end
end

function [demRawDir, note] = findDemRawDir(cfg, demName)
    rootDir = fullfile(cfg.rawRoot, demName);
    preferred = fullfile(rootDir, 'DEM_1m_raw');
    if exist(preferred, 'dir') == 7
        demRawDir = preferred;
        note = 'using DEM_1m_raw subfolder';
    elseif exist(rootDir, 'dir') == 7
        demRawDir = rootDir;
        note = 'DEM_1m_raw not found; using river root folder as fallback';
    else
        demRawDir = preferred;
        note = 'missing source folder';
    end
end

function unzipDownloadedZipFiles(demRawDir)
    cmd = sprintf('find "%s" -maxdepth 1 -type f -iname "*.zip" -exec unzip -n {} -d "%s" \\;', demRawDir, demRawDir);
    fprintf('Unzip command:\n%s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Failed to unzip zip files under: %s', demRawDir);
    end
end

function build3DEPFilelist(demRawDir, listFile)
    cmd = sprintf('find "%s" -type f \\( -iname "*.tif" -o -iname "*.tiff" \\) | sort > "%s"', demRawDir, listFile);
    fprintf('Build 3DEP Filelist.txt:\n%s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('Failed to build 3DEP Filelist.txt for: %s', demRawDir);
    end
end

function n = countLines(filePath)
    if exist(filePath, 'file') ~= 2
        n = 0;
        return;
    end
    [status, out] = system(sprintf('wc -l < "%s"', filePath));
    if status ~= 0
        n = 0;
    else
        n = str2double(strtrim(out));
        if isnan(n); n = 0; end
    end
end

function srsInfo = buildSRSReport(listFile, srsReportFile)
    cmd = sprintf(['bash -lc ''while IFS= read -r f; do ' ...
                   'epsg=$(gdalsrsinfo -o epsg "$f" 2>/dev/null | head -n 1 | tr -d "\\r"); ' ...
                   'if [ -z "$epsg" ]; then epsg="UNKNOWN"; fi; ' ...
                   'printf "%%s\t%%s\\n" "$epsg" "$f"; ' ...
                   'done < "%s" > "%s"'''], listFile, srsReportFile);
    status = system(cmd);
    if status ~= 0
        error('Failed to build SRS report for: %s', listFile);
    end

    txt = fileread(srsReportFile);
    lines = regexp(txt, '\r?\n', 'split');
    epsgs = {};
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line); continue; end
        parts = regexp(line, '\t', 'split');
        epsgs{end+1} = parts{1}; %#ok<AGROW>
    end

    if isempty(epsgs)
        srsInfo.uniqueEPSG = {};
        srsInfo.counts = [];
        srsInfo.hasMixed = false;
        return;
    end

    [u, ~, ic] = unique(epsgs);
    counts = accumarray(ic(:), 1);
    srsInfo.uniqueEPSG = u;
    srsInfo.counts = counts(:)';
    srsInfo.hasMixed = numel(u) > 1;
end

function [targetBathyRiver, bathyRef] = chooseTargetBathyRef(cfg, demName)
    if ~isempty(cfg.targetBathyRiver)
        targetBathyRiver = cfg.targetBathyRiver;
    elseif strcmp(demName, 'milwaukee_river_3DEP')
        % Usually not needed for shared Milwaukee DEM. If reprojection is needed,
        % pass 'targetBathyRiver','BadgerFinNull' or another Milwaukee child explicitly.
        targetBathyRiver = '';
    else
        targetBathyRiver = demName;
    end

    if isempty(targetBathyRiver)
        bathyRef = '';
    else
        bathyRef = fullfile(cfg.prRoot, 'Bathy_1m_FixND', targetBathyRiver, 'Bathy_1m.vrt');
    end
end

function [targetSRSFile, targetEPSG, resX, resY, ok] = prepareTargetSRS(cfg, outDir, bathyRef)
    targetSRSFile = fullfile(outDir, 'target_bathy_srs.wkt');
    targetEPSG = '';
    resX = 1;
    resY = 1;
    ok = false;

    if isempty(bathyRef) || exist(bathyRef, 'file') ~= 2
        return;
    end

    cmdSRS = sprintf('gdalsrsinfo -o wkt "%s" > "%s"', bathyRef, targetSRSFile);
    status = system(cmdSRS);
    if status ~= 0
        error('Failed to write target bathy SRS WKT from: %s', bathyRef);
    end

    [statusEPSG, outEPSG] = system(sprintf('gdalsrsinfo -o epsg "%s" | head -n 1', bathyRef));
    if statusEPSG == 0
        targetEPSG = strtrim(outEPSG);
    else
        targetEPSG = 'UNKNOWN';
    end

    try
        [~,~,~,geo,~,~,~] = RasterInfo(bathyRef);
        resX = abs(geo(2));
        resY = abs(geo(6));
    catch
        [resX, resY] = getPixelSizeByGdalInfo(bathyRef);
    end

    ok = true;
end

function [resX, resY] = getPixelSizeByGdalInfo(rasterPath)
    cmd = sprintf('gdalinfo "%s" | grep "Pixel Size" | head -n 1', rasterPath);
    [status, out] = system(cmd);
    if status ~= 0
        resX = 1; resY = 1; return;
    end
    tok = regexp(out, 'Pixel Size = \(([-+0-9.eE]+),([-+0-9.eE]+)\)', 'tokens');
    if isempty(tok)
        resX = 1; resY = 1;
    else
        resX = abs(str2double(tok{1}{1}));
        resY = abs(str2double(tok{1}{2}));
    end
end

function need = decideNeedReproject(cfg, srsInfo, targetEPSG, hasBathyRef)
    opt = cfg.reprojectToBathySRS;
    if islogical(opt)
        need = opt;
        return;
    elseif isnumeric(opt)
        need = logical(opt);
        return;
    end

    opt = lower(char(opt));
    switch opt
        case {'true','yes','on','always'}
            need = true;
        case {'false','no','off','never'}
            need = false;
        case 'auto'
            if ~hasBathyRef
                need = false;
                return;
            end
            if srsInfo.hasMixed
                need = true;
                return;
            end
            if isempty(srsInfo.uniqueEPSG) || isempty(targetEPSG) || strcmp(targetEPSG, 'UNKNOWN')
                need = false;
                return;
            end
            srcEPSG = srsInfo.uniqueEPSG{1};
            need = ~strcmp(strtrim(srcEPSG), strtrim(targetEPSG));
        otherwise
            error('Unsupported reprojectToBathySRS option: %s', opt);
    end
end

function reprojectSourceTilesToTargetSRS(listFile, reprojDir, reprojList, targetSRSFile, resX, resY, cfg)
    files = readLines(listFile);
    if isempty(files)
        error('Empty source list for reprojection: %s', listFile);
    end

    if exist(reprojList, 'file') == 2
        delete(reprojList);
    end

    fid = fopen(reprojList, 'w');
    if fid < 0
        error('Cannot write reprojection list: %s', reprojList);
    end

    for i = 1:numel(files)
        src = files{i};
        [~, bn, ~] = fileparts(src);
        outTif = fullfile(reprojDir, [safeName(bn), '_to_bathySRS.tif']);

        if cfg.overwrite || exist(outTif, 'file') ~= 2
            fprintf('[Reproject %d/%d]\n', i, numel(files));
            fprintf('  IN : %s\n', src);
            fprintf('  OUT: %s\n', outTif);

            cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                           'export PROJ_NETWORK=ON; ' ...
                           'gdalwarp -overwrite -of GTiff ' ...
                           '-multi -wo NUM_THREADS=%s -wm %d ' ...
                           '-t_srs "%s" ' ...
                           '-tr %.17g %.17g ' ...
                           '-r bilinear ' ...
                           '-dstnodata %.17g ' ...
                           '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ' ...
                           '"%s" "%s"'], ...
                           cfg.numThreads, cfg.warpMemoryMB, targetSRSFile, resX, resY, cfg.globalND, src, outTif);
            status = system(cmd);
            if status ~= 0
                fclose(fid);
                error('gdalwarp reprojection failed for source: %s', src);
            end
        else
            fprintf('[Reproject %d/%d] existing output, skip: %s\n', i, numel(files), outTif);
        end

        fprintf(fid, '%s\n', outTif);
    end

    fclose(fid);
end

function lines = readLines(filePath)
    txt = fileread(filePath);
    raw = regexp(txt, '\r?\n', 'split');
    lines = {};
    for i = 1:numel(raw)
        s = strtrim(raw{i});
        if ~isempty(s)
            lines{end+1} = s; %#ok<AGROW>
        end
    end
end

function status = buildRawVRT(listFile, vrtRaw)
    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalbuildvrt -overwrite ' ...
                   '-input_file_list "%s" "%s"'], ...
                   listFile, vrtRaw);
    fprintf('Build raw 3DEP VRT CMD:\n%s\n', cmd);
    status = system(cmd);
end

function status = buildFixNDVRT(vrtRaw, vrtFixND, cfg)
    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalwarp -overwrite -of VRT ' ...
                   '-multi -wo NUM_THREADS=%s -wm %d ' ...
                   '-dstnodata %.17g ' ...
                   '"%s" "%s"'], ...
                   cfg.numThreads, cfg.warpMemoryMB, cfg.globalND, vrtRaw, vrtFixND);
    fprintf('Build FixND 3DEP VRT CMD:\n%s\n', cmd);
    status = system(cmd);
end

function printRasterInfo(rasterPath, label)
    fprintf('[Check] %s:\n%s\n', label, rasterPath);
    cmd = sprintf('gdalinfo "%s" | grep -E "Size is|Coordinate System is|Pixel Size|Upper Left|Lower Right|NoData"', rasterPath);
    system(cmd);
end

function T = rowsToTable(rows)
    if isempty(rows)
        T = cell2table(cell(0,9), 'VariableNames', {'demName','demRawDir','sourceListFile','nFiles','listForVRT','vrtRaw','vrtFixND','status','note'});
    else
        T = cell2table(rows, 'VariableNames', {'demName','demRawDir','sourceListFile','nFiles','listForVRT','vrtRaw','vrtFixND','status','note'});
    end
end

function writeSummaryCSV(pathOut, rows)
    T = rowsToTable(rows);
    try
        writetable(T, pathOut);
    catch ME
        warning('Failed to write summary CSV: %s', ME.message);
    end
end

function s = safeName(s)
    s = char(s);
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if numel(s) > 160
        s = s(1:160);
    end
end

function s = optionToString(v)
    if islogical(v)
        s = mat2str(v);
    elseif isnumeric(v)
        s = num2str(v);
    else
        s = char(v);
    end
end
