function Summary = B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver_FIX_Near(riverName, varargin)
% FIX3_NEAR_MARKER: default resampling is nearest neighbour (-r near), matching previous B001_01 / OR / Kewa 3DEP resample-to-bathy-grid code.
% B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver_FIX3_Near
% Revised 2026-07-03 FIX2: removed target_bathy_grid_gdalinfo.json copyfile block completely
%
% Purpose
%   Resample and clip the projection-consistent 3DEP 1 m DEM from B003 to
%   the exact Bathy_1m_FixND grid for each selected river.
%
% Input
%   Bathy reference grid:
%     Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt
%
%   3DEP source grid from B003:
%     Processed_Results/3DEP_1m_VRT/<demName>/DEM_3DEP_1m_FixND.vrt
%
%   For Milwaukee child rivers, demName is shared:
%     milwaukee_river_3DEP
%
% Output
%   Processed_Results/3DEP_1m_ResampleClip/<river>/DEM_3DEP_1m_ResampleandClip.vrt
%
% Key design
%   - The output grid is exactly the bathy 1 m grid: same CRS, extent,
%     rows/cols, and pixel size.
%   - Default uses nearest neighbour (-r near) to reproduce the previous 3DEP-to-bathy-grid workflow.
%   - NoData is standardized to -999999.
%   - Zero is NOT treated as NoData.
%   - This step does not fuse bathy and 3DEP. Fusion should be B005.
%
% Usage
%   B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver('MD_PotomacRiver_Bathy_2019');
%   B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver('CA_KlamathRiver_TopoBathy_2018_D18', 'numThreads', 4);
%   B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver('BadgerFinNull');  % uses milwaukee_river_3DEP source
%   B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver('ALL');
%   B004_ResampleClip_3DEP_1m_To_BathyGrid_ForRiver('LIST');
%
% Options
%   'processedRoot'  default /tank/data/SFS/xinyis/data/bathymetry/Processed_Results
%   'overwrite'      default true
%   'numThreads'     default '4'
%   'wmMB'           default 2048
%   'resampleAlg'    default 'near' (same as previous 3DEP-to-bathy-grid code)
%   'outputFormat'   default 'VRT'; optional 'GTiff'
%   'dryRun'         default false
%   'stopOnMissing'  default true
%
% Note
%   OutputFormat='VRT' is preferred for huge rivers to avoid materializing
%   extremely large 1 m rasters. If outputFormat='GTiff', a GeoTIFF will be
%   written and a VRT wrapper will also be built for consistent downstream use.

if nargin < 1 || isempty(riverName)
    riverName = 'LIST';
end

cfg = defaultConfig();
cfg = parseOptions(cfg, varargin{:});

setupGdalPaths();
ensureDir(cfg.outRoot);
ensureDir(cfg.logRoot);

validRivers = selectedBathyRivers();

if ischar(riverName) || isstring(riverName)
    riverName = char(riverName);
    if strcmpi(riverName, 'LIST')
        printRiverList(validRivers);
        Summary = table();
        return;
    elseif strcmpi(riverName, 'ALL')
        riversToRun = validRivers;
    else
        riversToRun = {riverName};
    end
elseif iscell(riverName)
    riversToRun = riverName;
else
    error('riverName must be char/string, cell array, ''ALL'', or ''LIST''.');
end

% unique while preserving order
[~, ia] = unique(riversToRun, 'stable');
riversToRun = riversToRun(sort(ia));

for k = 1:numel(riversToRun)
    if ~ismember(riversToRun{k}, validRivers)
        fprintf('\nUnknown river: %s\n', riversToRun{k});
        printRiverList(validRivers);
        error('Input river is not in selected valid river list.');
    end
end

fprintf('\n============================================================\n');
fprintf('B004 resample/clip 3DEP 1m to Bathy_1m_FixND grid\n');
fprintf('Processed root : %s\n', cfg.processedRoot);
fprintf('3DEP source root: %s\n', cfg.depRoot);
fprintf('Output root    : %s\n', cfg.outRoot);
fprintf('Rivers         : %s\n', strjoin(riversToRun, ', '));
fprintf('Output format  : %s\n', cfg.outputFormat);
fprintf('Resample alg   : %s\n', cfg.resampleAlg);
fprintf('Overwrite      : %d\n', cfg.overwrite);
fprintf('GDAL thread    : %s\n', cfg.numThreads);
fprintf('============================================================\n');

summaryRows = {};
rowId = 0;

for i = 1:numel(riversToRun)
    river = riversToRun{i};
    demName = mapRiverToDemName(river);

    bathyRef = fullfile(cfg.processedRoot, 'Bathy_1m_FixND', river, 'Bathy_1m.vrt');
    depSrc   = fullfile(cfg.depRoot, demName, 'DEM_3DEP_1m_FixND.vrt');

    outDir = fullfile(cfg.outRoot, river);
    ensureDir(outDir);

    srsWkt = fullfile(outDir, 'target_bathy_srs.wkt');
    gridJson = fullfile(outDir, 'target_bathy_grid_gdalinfo.json');

    outVrt = fullfile(outDir, 'DEM_3DEP_1m_ResampleandClip.vrt');
    outTif = fullfile(outDir, 'DEM_3DEP_1m_ResampleandClip.tif');

    fprintf('\n============================================================\n');
    fprintf('[%d/%d] River: %s\n', i, numel(riversToRun), river);
    fprintf('3DEP demName : %s\n', demName);
    fprintf('Bathy ref   : %s\n', bathyRef);
    fprintf('3DEP source : %s\n', depSrc);
    fprintf('Output VRT  : %s\n', outVrt);
    fprintf('============================================================\n');

    statusLabel = 'OK';
    msg = '';

    if exist(bathyRef, 'file') ~= 2
        msg = sprintf('Missing bathy reference grid: %s', bathyRef);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            rowId = rowId + 1;
            summaryRows(rowId,:) = makeRow(river, demName, bathyRef, depSrc, outVrt, NaN, NaN, NaN, NaN, NaN, NaN, 'MISSING_BATHY', msg); %#ok<AGROW>
            continue;
        end
    end

    if exist(depSrc, 'file') ~= 2
        msg = sprintf('Missing 3DEP FixND VRT from B003: %s', depSrc);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            rowId = rowId + 1;
            summaryRows(rowId,:) = makeRow(river, demName, bathyRef, depSrc, outVrt, NaN, NaN, NaN, NaN, NaN, NaN, 'MISSING_3DEP', msg); %#ok<AGROW>
            continue;
        end
    end

    if ~cfg.overwrite && exist(outVrt, 'file') == 2
        fprintf('[SKIP] Existing output and overwrite=false: %s\n', outVrt);
        info = getRasterInfoGDAL(outVrt, outDir, 'existing_output_gdalinfo.json');
        rowId = rowId + 1;
        summaryRows(rowId,:) = makeRow(river, demName, bathyRef, depSrc, outVrt, info.cols, info.rows, info.xmin, info.ymin, info.xmax, info.ymax, 'SKIPPED_EXISTING', ''); %#ok<AGROW>
        continue;
    end

    % Get bathy target grid and SRS.
    % FIX2: getRasterInfoGDAL writes directly to target_bathy_grid_gdalinfo.json
    % under outDir, so there is no need to copy it. Copying here can fail
    % when source and destination are the same path.
    bathyInfo = getRasterInfoGDAL(bathyRef, outDir, 'target_bathy_grid_gdalinfo.json');
    if exist(gridJson, 'file') ~= 2
        error('Target bathy grid JSON was not created: %s', gridJson);
    end
    writeRasterSRS(bathyRef, srsWkt, bathyInfo);

    fprintf('Target rows/cols : %d / %d\n', bathyInfo.rows, bathyInfo.cols);
    fprintf('Target bbox      : %.10f %.10f %.10f %.10f\n', bathyInfo.xmin, bathyInfo.ymin, bathyInfo.xmax, bathyInfo.ymax);
    fprintf('Target pixsize   : %.12g / %.12g\n', bathyInfo.px, bathyInfo.py);
    fprintf('Target SRS WKT   : %s\n', srsWkt);

    deleteOutputs(outVrt, outTif, cfg.overwrite);

    if strcmpi(cfg.outputFormat, 'GTiff') || strcmpi(cfg.outputFormat, 'TIF') || strcmpi(cfg.outputFormat, 'TIFF')
        warpOut = outTif;
        ofFlag = 'GTiff';
        createOpts = '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS ';
    else
        warpOut = outVrt;
        ofFlag = 'VRT';
        createOpts = '';
    end

    cmd = sprintf([ ...
        'export PROJ_USE_PROJ4_INIT_RULES=YES; ', ...
        'export PROJ_NETWORK=ON; ', ...
        'gdalwarp -overwrite -of %s ', ...
        '-multi -wo NUM_THREADS=%s -wo INIT_DEST=NO_DATA -wm %d ', ...
        '-r %s -ot Float32 ', ...
        '-t_srs %s -te_srs %s ', ...
        '-te %.10f %.10f %.10f %.10f ', ...
        '-ts %d %d ', ...
        '-srcnodata %g -dstnodata %g ', ...
        '%s', ...
        '%s %s'], ...
        ofFlag, cfg.numThreads, cfg.wmMB, cfg.resampleAlg, ...
        qpath(srsWkt), qpath(srsWkt), ...
        bathyInfo.xmin, bathyInfo.ymin, bathyInfo.xmax, bathyInfo.ymax, ...
        bathyInfo.cols, bathyInfo.rows, ...
        cfg.noData, cfg.noData, ...
        createOpts, ...
        qpath(depSrc), qpath(warpOut));

    fprintf('B004 gdalwarp CMD:\n%s\n', cmd);

    if cfg.dryRun
        statusLabel = 'DRYRUN';
    else
        [status, sysMsg] = system(cmd, '-echo');
        if status ~= 0
            error('B004 gdalwarp failed for %s:\n%s', river, sysMsg);
        end

        if strcmpi(ofFlag, 'GTiff')
            % Build VRT wrapper with the formal downstream name.
            cmdVrt = sprintf('gdalbuildvrt -overwrite -srcnodata %g -vrtnodata %g %s %s', ...
                cfg.noData, cfg.noData, qpath(outVrt), qpath(outTif));
            fprintf('Build VRT wrapper CMD:\n%s\n', cmdVrt);
            [statusV, msgV] = system(cmdVrt, '-echo');
            if statusV ~= 0
                error('gdalbuildvrt wrapper failed for %s:\n%s', river, msgV);
            end
        end

        fprintf('[Check] Aligned 3DEP VRT:\n%s\n', outVrt);
        system(sprintf('gdalinfo %s | grep -E "Size is|Coordinate System is|Pixel Size|Upper Left|Lower Right|NoData"', qpath(outVrt)), '-echo');

        outInfo = getRasterInfoGDAL(outVrt, outDir, 'aligned_3dep_grid_gdalinfo.json');
        if ~sameGrid(bathyInfo, outInfo)
            statusLabel = 'GRID_MISMATCH_WARNING';
            msg = 'Output grid does not exactly match bathy reference grid. Inspect gdalinfo JSON.';
            warning('[%s] %s', river, msg);
        else
            fprintf('[OK] Output grid exactly matches bathy reference grid.\n');
        end
    end

    rowId = rowId + 1;
    summaryRows(rowId,:) = makeRow(river, demName, bathyRef, depSrc, outVrt, bathyInfo.cols, bathyInfo.rows, bathyInfo.xmin, bathyInfo.ymin, bathyInfo.xmax, bathyInfo.ymax, statusLabel, msg); %#ok<AGROW>
end

Summary = cell2table(summaryRows, 'VariableNames', { ...
    'river','demName','bathyRef','depSource','outVRT', ...
    'cols','rows','xmin','ymin','xmax','ymax','status','message'});

ts = datestr(now, 'yyyymmdd_HHMMSS');
if numel(riversToRun) == 1
    logName = sprintf('B004_ResampleClip_3DEP_1m_To_BathyGrid_%s_%s.csv', riversToRun{1}, ts);
else
    logName = sprintf('B004_ResampleClip_3DEP_1m_To_BathyGrid_MULTI_%s.csv', ts);
end
logCsv = fullfile(cfg.logRoot, logName);
writetable(Summary, logCsv);

fprintf('\n============================================================\n');
fprintf('B004 finished. Summary written:\n%s\n', logCsv);
fprintf('Next step should use:\n');
fprintf('  %s/3DEP_1m_ResampleClip/<river>/DEM_3DEP_1m_ResampleandClip.vrt\n', cfg.processedRoot);
fprintf('============================================================\n');

end

% =====================================================================
% Config and helper functions
% =====================================================================

function cfg = defaultConfig()
cfg.processedRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
cfg.depRoot = fullfile(cfg.processedRoot, '3DEP_1m_VRT');
cfg.outRoot = fullfile(cfg.processedRoot, '3DEP_1m_ResampleClip');
cfg.logRoot = fullfile(cfg.processedRoot, 'Logs');
cfg.overwrite = true;
cfg.stopOnMissing = true;
cfg.dryRun = false;
cfg.numThreads = '4';
cfg.wmMB = 2048;
cfg.resampleAlg = 'near';
cfg.noData = -999999;
cfg.outputFormat = 'VRT';
end

function cfg = parseOptions(cfg, varargin)
if mod(numel(varargin), 2) ~= 0
    error('Options must be name/value pairs.');
end
for i = 1:2:numel(varargin)
    key = char(varargin{i});
    val = varargin{i+1};
    switch lower(key)
        case 'processedroot'
            cfg.processedRoot = char(val);
            cfg.depRoot = fullfile(cfg.processedRoot, '3DEP_1m_VRT');
            cfg.outRoot = fullfile(cfg.processedRoot, '3DEP_1m_ResampleClip');
            cfg.logRoot = fullfile(cfg.processedRoot, 'Logs');
        case 'deproot'
            cfg.depRoot = char(val);
        case 'outroot'
            cfg.outRoot = char(val);
        case 'logroot'
            cfg.logRoot = char(val);
        case 'overwrite'
            cfg.overwrite = logical(val);
        case 'stoponmissing'
            cfg.stopOnMissing = logical(val);
        case 'dryrun'
            cfg.dryRun = logical(val);
        case 'numthreads'
            if isnumeric(val)
                cfg.numThreads = num2str(val);
            else
                cfg.numThreads = char(val);
            end
        case 'wmmb'
            cfg.wmMB = double(val);
        case 'resamplealg'
            cfg.resampleAlg = char(val);
        case 'nodata'
            cfg.noData = double(val);
        case 'outputformat'
            cfg.outputFormat = char(val);
        otherwise
            error('Unknown option: %s', key);
    end
end
end

function setupGdalPaths()
try
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
catch ME
    warning('GDALLoad / CREST_Prep setup skipped or failed: %s', ME.message);
end
end

function rivers = selectedBathyRivers()
rivers = { ...
    'BadgerFinNull', ...
    'Estabrook_Combined', ...
    'KewaFix2Null', ...
    'Kletzch_Combined_UpMax3Null', ...
    'CA_KlamathRiver_TopoBathy_2018_D18', ...
    'CO_UpperColorado_Topobathy_1_2020', ...
    'MD_PotomacRiver_Bathy_2019', ...
    'NE_Niobrara_Topobathy_2018', ...
    'OR_MKRC_Topobathy_2021', ...
    'OR_SantiamRiverTB_Topobathy_1_D23', ...
    'WA_ChehalisRiverTB_Topobathy_1_D23', ...
    'WA_Nisqually_Bathymetric_2020' ...
    };
end

function demName = mapRiverToDemName(river)
river = char(river);
if ismember(river, {'BadgerFinNull','Estabrook_Combined','KewaFix2Null','Kletzch_Combined_UpMax3Null'})
    demName = 'milwaukee_river_3DEP';
else
    demName = river;
end
end

function printRiverList(rivers)
fprintf('\nValid B004 rivers:\n');
for i = 1:numel(rivers)
    fprintf('  %2d. %s\n', i, rivers{i});
end
fprintf('\nMilwaukee child rivers use 3DEP source task: milwaukee_river_3DEP\n');
end

function ensureDir(p)
if exist(p, 'dir') ~= 7
    mkdir(p);
end
end

function info = getRasterInfoGDAL(refGrid, outDir, jsonName)
if nargin < 3 || isempty(jsonName)
    jsonName = 'raster_gdalinfo.json';
end
ensureDir(outDir);
jsonFile = fullfile(outDir, jsonName);
cmd = sprintf('gdalinfo -json %s > %s', qpath(refGrid), qpath(jsonFile));
[status, msg] = system(cmd);
if status ~= 0
    error('gdalinfo -json failed for %s:\n%s', refGrid, msg);
end
jtxt = fileread(jsonFile);
j = jsondecode(jtxt);

info.cols = double(j.size(1));
info.rows = double(j.size(2));

if ~isfield(j, 'geoTransform') || numel(j.geoTransform) < 6
    error('No geoTransform found in gdalinfo json for: %s', refGrid);
end

gt = double(j.geoTransform(:)');
x0 = gt(1); px = gt(2); rx = gt(3);
y0 = gt(4); ry = gt(5); py = gt(6);

xs = [x0, x0 + info.cols*px, x0 + info.rows*rx, x0 + info.cols*px + info.rows*rx];
ys = [y0, y0 + info.cols*ry, y0 + info.rows*py, y0 + info.cols*ry + info.rows*py];

info.xmin = min(xs);
info.xmax = max(xs);
info.ymin = min(ys);
info.ymax = max(ys);
info.geoTransform = gt;
info.px = abs(px);
info.py = abs(py);

if isfield(j, 'coordinateSystem') && isfield(j.coordinateSystem, 'wkt')
    info.wkt = j.coordinateSystem.wkt;
else
    info.wkt = '';
end
end

function writeRasterSRS(refGrid, srsWkt, info)
cmd = sprintf('gdalsrsinfo -o wkt %s > %s', qpath(refGrid), qpath(srsWkt));
[status, msg] = system(cmd);
if status ~= 0 || exist(srsWkt, 'file') ~= 2 || dir(srsWkt).bytes == 0
    if isfield(info, 'wkt') && ~isempty(info.wkt)
        fid = fopen(srsWkt, 'w');
        if fid < 0
            error('Cannot write SRS WKT: %s', srsWkt);
        end
        fprintf(fid, '%s\n', info.wkt);
        fclose(fid);
    else
        error('Cannot get SRS WKT. gdalsrsinfo message: %s', msg);
    end
end
end

function tf = sameGrid(a, b)
tol = max([a.px, a.py, b.px, b.py, 1]) * 1e-6;
tf = true;
tf = tf && a.cols == b.cols;
tf = tf && a.rows == b.rows;
tf = tf && abs(a.xmin - b.xmin) <= tol;
tf = tf && abs(a.xmax - b.xmax) <= tol;
tf = tf && abs(a.ymin - b.ymin) <= tol;
tf = tf && abs(a.ymax - b.ymax) <= tol;
tf = tf && abs(a.px - b.px) <= tol;
tf = tf && abs(a.py - b.py) <= tol;
end

function deleteOutputs(outVrt, outTif, overwrite)
if ~overwrite
    return;
end
if exist(outVrt, 'file') == 2
    delete(outVrt);
end
if exist(outTif, 'file') == 2
    delete(outTif);
end
if exist([outTif '.aux.xml'], 'file') == 2
    delete([outTif '.aux.xml']);
end
if exist([outTif '.ovr'], 'file') == 2
    delete([outTif '.ovr']);
end
end

function row = makeRow(river, demName, bathyRef, depSrc, outVrt, cols, rows, xmin, ymin, xmax, ymax, status, message)
row = {string(river), string(demName), string(bathyRef), string(depSrc), string(outVrt), ...
    double(cols), double(rows), double(xmin), double(ymin), double(xmax), double(ymax), string(status), string(message)};
end

function s = qpath(x)
s = ['"', char(x), '"'];
end
