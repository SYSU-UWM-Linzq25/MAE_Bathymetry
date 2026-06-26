function C001_Clip_USDetailed_SHP_To_BathyExtent_All(varargin)
%% ============================================================
%  C001_Clip_USDetailed_SHP_To_BathyExtent_All.m
%
%  Purpose:
%    Clip the CONUS / large-domain US detailed RiverOnly water-body SHP
%    to each river's final bathy-grid extent, and save one clipped SHP
%    per river in a new folder.
%
%  Why:
%    This is the vector counterpart of the LCC extraction workflow.
%    For LCC, we warp the global raster to the final bathy grid.
%    For US detailed water bodies, we first extract the vector features
%    intersecting each river's bathy / bathy+3DEP extent. These clipped
%    SHPs can then be rasterized to the exact bathy grid in the next step.
%
%  Main logic:
%    For each river:
%      1) Read reference raster grid info from Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%         by default, matching the LCC rebuilding script.
%      2) Get bbox and projection from the reference raster.
%      3) Use ogr2ogr to:
%           - spatially filter source SHP by bbox
%           - reproject to the reference raster CRS
%           - clip geometries to the bbox
%      4) Save clipped SHP under:
%           Processed_Results/USDetailed_RiverOnly_SHP/<river>/USRiverClip.shp
%
%  Important:
%    - This script does NOT rasterize the SHP yet.
%    - This script does NOT change existing LCC / bathy / tile outputs.
%    - The output SHP is reprojected to the target bathy grid CRS.
%    - The output extent is controlled by the reference raster bbox.
%
%  Example:
%    C001_Clip_USDetailed_SHP_To_BathyExtent_All()
%
%    C001_Clip_USDetailed_SHP_To_BathyExtent_All( ...
%       'selectedRivers', {'MD_PotomacRiver_Bathy_2019'}, ...
%       'extentRes', 1, ...
%       'overwrite', true)
%
%    C001_Clip_USDetailed_SHP_To_BathyExtent_All( ...
%       'usDetailedShp', '/tank/data/SFS/xinyis/data/US_detalied_water_bodies/USA_Detailed_Water_Bodies_RiverOnly.shp')
% ============================================================

%% -------------------- Parse inputs --------------------
p = inputParser;

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

% Can be either:
%   1) a direct .shp path, or
%   2) a folder containing one or more .shp files.
% If a folder is provided, the script prefers a file whose name contains
% both "River" and "Only".
addParameter(p, 'usDetailedShp', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask/US_Detailed', ...
    @(x) ischar(x) || isstring(x));

% Reference grid resolution used only to define each river bbox and CRS.
% Default is 1 m, consistent with the current LCC rebuild workflow.
addParameter(p, 'extentRes', 1, @isnumeric);

% selectedRivers = {} means process all rivers under Bathy_1m_FixND.
addParameter(p, 'selectedRivers', {}, @(x) iscell(x) || isstring(x));

% Output folder name under rootPR.
addParameter(p, 'outFolderName', 'USDetailed_RiverOnly_SHP', ...
    @(x) ischar(x) || isstring(x));

% Optional bbox buffer in target raster coordinate units.
% Keep 0 for strict clipping to the bathy extent.
addParameter(p, 'bboxBuffer', 0, @isnumeric);

addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'backupOld', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, varargin{:});

rootPR        = char(p.Results.rootPR);
usDetailedIn  = char(p.Results.usDetailedShp);
extentRes     = p.Results.extentRes;
selectedRivers = p.Results.selectedRivers;
outFolderName = char(p.Results.outFolderName);
bboxBuffer    = p.Results.bboxBuffer;
overwrite     = p.Results.overwrite;
backupOld     = p.Results.backupOld;
doPathSetup   = p.Results.doPathSetup;

%% -------------------- Path setup --------------------
if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

usDetailedShp = resolveUSDetailedShp(usDetailedIn);

if exist(usDetailedShp, 'file') ~= 2
    error('Missing US detailed SHP: %s', usDetailedShp);
end

%% -------------------- River list --------------------
if isempty(selectedRivers)
    d = dir(fullfile(rootPR, 'Bathy_1m_FixND'));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));

    rivers = cell(numel(d), 1);
    for i = 1:numel(d)
        rivers{i} = d(i).name;
    end
else
    rivers = cellstr(selectedRivers);
end

%% -------------------- Logs --------------------
outRoot = fullfile(rootPR, outFolderName);
if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

logDir = fullfile(rootPR, 'Z030_USDetailed_RiverOnly_SHP_Clip_Log');
if exist(logDir, 'dir') ~= 7
    mkdir(logDir);
end

logCSV = fullfile(logDir, 'C001_Clip_USDetailed_SHP_To_BathyExtent_Log.csv');
fid = fopen(logCSV, 'w');
fprintf(fid, ['River,ExtentResolution_m,Status,Rows,Cols,' ...
              'Xmin,Ymin,Xmax,Ymax,FeatureCount,' ...
              'SourceSHP,TargetGrid,OutputSHP,Message\n']);
fclose(fid);

fprintf('\n============================================================\n');
fprintf('C001 clip US detailed RiverOnly SHP to river bathy extent\n');
fprintf('Source SHP       : %s\n', usDetailedShp);
fprintf('Reference res    : %g m\n', extentRes);
fprintf('Output root      : %s\n', outRoot);
fprintf('Number of rivers : %d\n', numel(rivers));
fprintf('Bbox buffer      : %.6g target CRS units\n', bboxBuffer);
fprintf('============================================================\n');

%% ============================================================
%  Main loop
% ============================================================
for iRiver = 1:numel(rivers)

    river = rivers{iRiver};

    if contains(river, 'NoNeed')
        continue;
    end

    fprintf('\n============================================================\n');
    fprintf('[%d/%d] River: %s\n', iRiver, numel(rivers), river);
    fprintf('============================================================\n');

    %% -------------------- Reference bathy grid --------------------
    refGrid = fullfile(rootPR, sprintf('Bathy_%dm_FixND', extentRes), ...
        river, sprintf('Bathy_%dm.vrt', extentRes));

    if exist(refGrid, 'file') ~= 2
        refTif = fullfile(rootPR, sprintf('Bathy_%dm_FixND', extentRes), ...
            river, sprintf('Bathy_%dm.tif', extentRes));

        if exist(refTif, 'file') == 2
            refGrid = refTif;
        else
            msg = sprintf('Missing reference bathy grid: %s', refGrid);
            warning('[%s] %s', river, msg);
            appendLog(logCSV, river, extentRes, 'SKIP_MISSING_REFGRID', ...
                NaN, NaN, NaN, NaN, NaN, NaN, NaN, ...
                usDetailedShp, refGrid, '', msg);
            continue;
        end
    end

    [~, rowsB, colsB, geoB, projB, ~, ~] = RasterInfo(refGrid);

    xmin = geoB(1);
    xres = geoB(2);
    ymax = geoB(4);
    yres = geoB(6);

    xmax = xmin + colsB * xres;
    ymin = ymax + rowsB * yres;

    if bboxBuffer > 0
        xmin = xmin - bboxBuffer;
        ymin = ymin - bboxBuffer;
        xmax = xmax + bboxBuffer;
        ymax = ymax + bboxBuffer;
    end

    %% -------------------- Output paths --------------------
    outSub = fullfile(outRoot, river);
    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    outShp = fullfile(outSub, 'USRiverClip.shp');
    srsWkt = fullfile(outSub, 'target_bathy_srs.wkt');

    if exist(outShp, 'file') == 2 && ~overwrite
        nFeat = getFeatureCount(outShp);
        msg = 'Output exists and overwrite=false';
        fprintf('[%s] Skip: %s\n', river, msg);
        appendLog(logCSV, river, extentRes, 'SKIP_EXISTS', ...
            rowsB, colsB, xmin, ymin, xmax, ymax, nFeat, ...
            usDetailedShp, refGrid, outShp, msg);
        continue;
    end

    %% -------------------- Backup old output --------------------
    if backupOld && exist(outShp, 'file') == 2
        backupDir = fullfile(rootPR, 'Z021_Backup_USDetailed_RiverOnly_SHP_Before_Clip', river);
        if exist(backupDir, 'dir') ~= 7
            mkdir(backupDir);
        end
        stamp = datestr(now, 'yyyymmdd_HHMMSS');
        backupSub = fullfile(backupDir, stamp);
        mkdir(backupSub);
        copyShpSidecars(outSub, 'USRiverClip', backupSub);
        fprintf('Backup old clipped SHP to:\n%s\n', backupSub);
    end

    %% -------------------- Clean old products --------------------
    deleteShpSidecars(outSub, 'USRiverClip');

    %% -------------------- Write target SRS WKT --------------------
    fid = fopen(srsWkt, 'w');
    if fid < 0
        error('Cannot write SRS WKT: %s', srsWkt);
    end
    fprintf(fid, '%s', projB);
    fclose(fid);

    %% -------------------- Clip source SHP to target bbox --------------------
    % Strategy:
    %   -spat_srs + -spat uses the bathy CRS bbox as a spatial filter.
    %   -t_srs reprojects features to the bathy CRS.
    %   -clipdst clips output geometries to the bathy bbox.
    % If the local GDAL does not support -spat_srs, we retry a slower
    % fallback without the source-side spatial filter.

    cmd = sprintf([ ...
        'ogr2ogr -overwrite -f "ESRI Shapefile" ', ...
        '-skipfailures ', ...
        '-t_srs %s ', ...
        '-spat_srs %s -spat %.10f %.10f %.10f %.10f ', ...
        '-lco ENCODING=UTF-8 ', ...
        '%s %s' ], ...
        qpath(srsWkt), ...
        qpath(srsWkt), xmin, ymin, xmax, ymax, ...
        qpath(outShp), qpath(usDetailedShp));

    fprintf('[%s] Clip US detailed SHP to bathy extent\n', river);
    fprintf('Reference grid: %s\n', refGrid);
    fprintf('Output SHP    : %s\n', outShp);
    fprintf('Rows/Cols     : %d / %d\n', rowsB, colsB);
    fprintf('BBox          : %.6f %.6f %.6f %.6f\n', xmin, ymin, xmax, ymax);
    fprintf('%s\n', cmd);

    status = system(cmd);

    if status ~= 0
        warning('[%s] First ogr2ogr command failed. Retry without -spat_srs/-spat filter.', river);

        cmdFallback = sprintf([ ...
            'ogr2ogr -overwrite -f "ESRI Shapefile" ', ...
            '-skipfailures ', ...
            '-spat_srs %s -spat %.10f %.10f %.10f %.10f ', ...
            '-t_srs %s ', ...
            '-lco ENCODING=UTF-8 ', ...
            '%s %s' ], ...
            qpath(srsWkt), xmin, ymin, xmax, ymax, ...
            qpath(srsWkt), ...
            qpath(outShp), qpath(usDetailedShp));

        fprintf('%s\n', cmdFallback);
        status = system(cmdFallback);
    end

    if status ~= 0 || exist(outShp, 'file') ~= 2
        msg = 'ogr2ogr clipping failed';
        warning('[%s] %s', river, msg);
        appendLog(logCSV, river, extentRes, 'FAIL_OGR2OGR', ...
            rowsB, colsB, xmin, ymin, xmax, ymax, NaN, ...
            usDetailedShp, refGrid, outShp, msg);
        continue;
    end

    %% -------------------- Feature count and final log --------------------
    nFeat = getFeatureCount(outShp);

    fprintf('[%s] Feature count after clipping: %g\n', river, nFeat);
    fprintf('[%s] Output SHP: %s\n', river, outShp);

    if isnan(nFeat)
        statusStr = 'PASS_NO_COUNT';
        msg = 'Output exists; feature count could not be parsed';
    elseif nFeat == 0
        statusStr = 'PASS_EMPTY';
        msg = 'Output exists but contains zero features';
    else
        statusStr = 'PASS';
        msg = 'OK';
    end

    appendLog(logCSV, river, extentRes, statusStr, ...
        rowsB, colsB, xmin, ymin, xmax, ymax, nFeat, ...
        usDetailedShp, refGrid, outShp, msg);
end

fprintf('\n============================================================\n');
fprintf('C001 US detailed RiverOnly SHP clipping done.\n');
fprintf('Log written to:\n%s\n', logCSV);
fprintf('============================================================\n');

end

%% ============================================================
%  Local helper: resolve source SHP path
% ============================================================
function shpPath = resolveUSDetailedShp(inPath)
    inPath = char(inPath);

    if exist(inPath, 'file') == 2
        [~, ~, ext] = fileparts(inPath);
        if strcmpi(ext, '.shp')
            shpPath = inPath;
            return;
        else
            error('usDetailedShp must be a .shp file or a folder containing .shp files: %s', inPath);
        end
    end

    if exist(inPath, 'dir') ~= 7
        error('usDetailedShp does not exist as file or folder: %s', inPath);
    end

    d = dir(fullfile(inPath, '**', '*.shp'));
    if isempty(d)
        error('No .shp file found under folder: %s', inPath);
    end

    names = {d.name};
    score = zeros(numel(d), 1);
    for i = 1:numel(d)
        nm = lower(names{i});
        if contains(nm, 'river')
            score(i) = score(i) + 10;
        end
        if contains(nm, 'only')
            score(i) = score(i) + 10;
        end
        if contains(nm, 'water')
            score(i) = score(i) + 1;
        end
    end

    [bestScore, idx] = max(score);

    if numel(d) > 1 && bestScore == 0
        fprintf('Found multiple SHPs but no RiverOnly-like name:\n');
        for i = 1:numel(d)
            fprintf('  %s\n', fullfile(d(i).folder, d(i).name));
        end
        error('Please provide the exact RiverOnly .shp path via ''usDetailedShp'' parameter.');
    end

    shpPath = fullfile(d(idx).folder, d(idx).name);

    if numel(d) > 1
        fprintf('Auto-selected US detailed SHP:\n%s\n', shpPath);
    end
end

%% ============================================================
%  Local helper: quoted path for shell command
% ============================================================
function s = qpath(p)
    p = char(p);
    p = strrep(p, '"', '\"');
    s = ['"' p '"'];
end

%% ============================================================
%  Local helper: delete shapefile sidecars
% ============================================================
function deleteShpSidecars(folder, baseName)
    exts = {'.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj', '.fix', '.sbn', '.sbx'};
    for i = 1:numel(exts)
        f = fullfile(folder, [baseName exts{i}]);
        if exist(f, 'file') == 2
            delete(f);
        end
    end
end

%% ============================================================
%  Local helper: copy shapefile sidecars
% ============================================================
function copyShpSidecars(srcFolder, baseName, dstFolder)
    exts = {'.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj', '.fix', '.sbn', '.sbx'};
    for i = 1:numel(exts)
        f = fullfile(srcFolder, [baseName exts{i}]);
        if exist(f, 'file') == 2
            copyfile(f, fullfile(dstFolder, [baseName exts{i}]));
        end
    end
end

%% ============================================================
%  Local helper: feature count
% ============================================================
function nFeat = getFeatureCount(shpPath)
    nFeat = NaN;

    if exist(shpPath, 'file') ~= 2
        return;
    end

    [~, baseName, ~] = fileparts(shpPath);
    cmd = sprintf('ogrinfo -ro -so %s %s', qpath(shpPath), qpath(baseName));
    [status, txt] = system(cmd);

    if status ~= 0
        return;
    end

    tok = regexp(txt, 'Feature Count:\s*(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        nFeat = str2double(tok{1});
    end
end

%% ============================================================
%  Local helper: append log row
% ============================================================
function appendLog(logCSV, river, res, statusStr, rows, cols, ...
                   xmin, ymin, xmax, ymax, nFeat, ...
                   srcSHP, targetGrid, outSHP, msg)

    fid = fopen(logCSV, 'a');

    if fid < 0
        warning('Cannot open log CSV: %s', logCSV);
        return;
    end

    msg = csvsafe(msg);
    srcSHP = csvsafe(srcSHP);
    targetGrid = csvsafe(targetGrid);
    outSHP = csvsafe(outSHP);

    fprintf(fid, '%s,%g,%s,%g,%g,%.10f,%.10f,%.10f,%.10f,%g,%s,%s,%s,%s\n', ...
        river, res, statusStr, rows, cols, xmin, ymin, xmax, ymax, nFeat, ...
        srcSHP, targetGrid, outSHP, msg);

    fclose(fid);
end

function s = csvsafe(s)
    s = char(s);
    s = strrep(s, ',', ';');
    s = strrep(s, sprintf('\n'), ' ');
    s = strrep(s, sprintf('\r'), ' ');
end
