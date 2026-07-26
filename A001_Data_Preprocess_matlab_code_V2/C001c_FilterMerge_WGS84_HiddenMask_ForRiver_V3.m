function Summary = C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3(riverName, targetResM, varargin)
% C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3
% 2026-07-24
%
% V2 update
%   - If C001b output is already a single whole-river GeoTIFF, skip it by default.
%   - Only tiled WGS84 outputs are filtered/merged.
%   - For tiled outputs, empty tiles are removed from the source list; the final
%     merged GeoTIFF is built from non-empty tiles only. Deleted internal tile
%     footprints become NoData=255 in the merged product, while deleted tiles
%     outside the kept-tile bounding box are cropped out unless preserveFullGrid=true.
%
% Purpose
%   Post-process C001b WGS84 HiddenMask results:
%     1) scan WGS84 HiddenMask tiles;
%     2) remove tiles that contain no hidden river pixels, i.e. tiles that are
%        all 0 background and/or 255 NoData;
%     3) build a VRT from only non-empty tiles;
%     4) translate that VRT into one merged WGS84 GeoTIFF.
%
% This script does NOT modify or delete the original C001b outputs. It writes
% a clean merged product into a new folder.
%
% Input from C001b
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/
%       HiddenMask_ByRiver_<res>m_WGS84/<river>/
%           HiddenMask_<res>m_WGS84_tiles/*.tif   [large rivers]
%           HiddenMask_<res>m_WGS84.tif           [small rivers]
%           HiddenMask_<res>m_WGS84.vrt
%
% Output
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/
%       HiddenMask_ByRiver_<res>m_WGS84_CleanMerged/<river>/
%           HiddenMask_<res>m_WGS84_nonempty.vrt
%           HiddenMask_<res>m_WGS84_clean_merged.tif
%           Lists/HiddenMask_<res>m_WGS84_nonempty_tile_list.txt
%           Lists/HiddenMask_<res>m_WGS84_tile_filter_manifest.csv
%
% Mask values
%   1   = hidden / river area to keep
%   0   = background
%   255 = NoData
%
% Empty tile definition
%   A tile is removed if it has no valid pixel with value 1.
%   In normal 0/1/255 mask tiles, this means all values are 0 and/or 255.
%
% Usage
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('WA_Nisqually_Bathymetric_2020', 1);
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('ALL', 1);
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('LIST', 1);
%
% Optional examples
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('ALL', 1, 'numThreads', 4);
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('ALL', 1, 'copyKeptTiles', true);
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('ALL', 1, 'preserveFullGrid', true);
%   C001c_FilterMerge_WGS84_HiddenMask_ForRiver_V3('ALL', 1, 'overwrite', true);

if nargin < 1 || isempty(riverName)
    riverName = 'LIST';
end
if nargin < 2 || isempty(targetResM)
    targetResM = 1;
end

cfg = defaultConfig();
cfg = parseOptions(cfg, varargin{:});

setupGdalPaths();
ensureDir(cfg.logRoot);

targetResM = unique(double(targetResM(:)'));

if ischar(riverName) || isstring(riverName)
    riverName = char(riverName);
    if strcmpi(riverName, 'LIST')
        printRiverList(scanRiverFolders(cfg, targetResM));
        Summary = table();
        return;
    elseif strcmpi(riverName, 'ALL')
        riversToRun = scanRiverFolders(cfg, targetResM);
    else
        riversToRun = {riverName};
    end
elseif iscell(riverName)
    riversToRun = riverName;
else
    error('riverName must be char/string, cell array, ''ALL'', or ''LIST''.');
end

fprintf('\n============================================================\n');
fprintf('C001c V3 Filter empty WGS84 HiddenMask tiles and merge to one GeoTIFF\n');
fprintf('Input root  : %s/HiddenMask_ByRiver_<res>m_WGS84/<river>\n', cfg.maskRoot);
fprintf('Output root : %s/HiddenMask_ByRiver_<res>m_WGS84_CleanMerged/<river>\n', cfg.maskRoot);
fprintf('Rivers      : %s\n', strjoin(riversToRun, ', '));
fprintf('Resolutions : %s m\n', mat2str(targetResM));
fprintf('Keep rule   : keep tile if computed maximum >= %.3f and < NoData %.0f\n', cfg.keepValue, cfg.maskNoData);
fprintf('NoData      : %d\n', cfg.maskNoData);
fprintf('Copy kept tiles: %d\n', cfg.copyKeptTiles);
fprintf('Skip single tif : %d\n', cfg.skipSingleTif);
fprintf('Preserve full grid extent: %d\n', cfg.preserveFullGrid);
fprintf('Overwrite   : %d\n', cfg.overwrite);
fprintf('============================================================\n');

summaryRows = {};
rowId = 0;

for r = 1:numel(riversToRun)
    river = riversToRun{r};

    fprintf('\n############################################################\n');
    fprintf('River: %s\n', river);
    fprintf('############################################################\n');

    for j = 1:numel(targetResM)
        resM = targetResM(j);

        try
            R = processOneRiver(river, resM, cfg);
            statusText = R.status;
            msg = R.message;
        catch ME
            if cfg.stopOnError
                rethrow(ME);
            end
            warning('[%s %dm] C001c failed: %s', river, resM, ME.message);
            R = emptyResult(river, resM, cfg);
            statusText = 'FAIL';
            msg = ME.message;
        end

        rowId = rowId + 1;
        summaryRows(rowId,:) = { ... %#ok<AGROW>
            river, resM, R.sourceMode, R.nCandidate, R.nKept, R.nRemoved, ...
            R.outVrt, R.outTif, R.manifestCsv, R.keptList, statusText, msg};
    end
end

Summary = rowsToTable(summaryRows);
summaryFile = fullfile(cfg.logRoot, sprintf('C001c_WGS84_HiddenMask_CleanMerge_%s_%s.csv', ...
    safeName(strjoin(riversToRun, '__')), datestr(now, 'yyyymmdd_HHMMSS')));
writeSummaryCSV(summaryFile, summaryRows);

fprintf('\n============================================================\n');
fprintf('C001c finished. Summary written:\n%s\n', summaryFile);
fprintf('Outputs are under:\n%s/HiddenMask_ByRiver_<res>m_WGS84_CleanMerged/<river>/\n', cfg.maskRoot);
fprintf('============================================================\n');

end

%% ============================================================
% Main processing
% ============================================================

function R = processOneRiver(river, resM, cfg)
    inRoot = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm_WGS84', resM), river);
    inTileDir = fullfile(inRoot, sprintf('HiddenMask_%dm_WGS84_tiles', resM));
    inSingleTif = fullfile(inRoot, sprintf('HiddenMask_%dm_WGS84.tif', resM));
    inVrt = fullfile(inRoot, sprintf('HiddenMask_%dm_WGS84.vrt', resM));

    outRoot = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm_WGS84_CleanMerged', resM), river);
    listDir = fullfile(outRoot, 'Lists');
    ensureDir(outRoot);
    ensureDir(listDir);

    keptTileDir = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84_nonempty_tiles', resM));
    if cfg.copyKeptTiles
        ensureDir(keptTileDir);
    end

    outVrt = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84_nonempty.vrt', resM));
    outTif = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84_clean_merged.tif', resM));
    keptList = fullfile(listDir, sprintf('HiddenMask_%dm_WGS84_nonempty_tile_list.txt', resM));
    manifestCsv = fullfile(listDir, sprintf('HiddenMask_%dm_WGS84_tile_filter_manifest.csv', resM));

    R = emptyResult(river, resM, cfg);
    R.outRoot = outRoot;
    R.outVrt = outVrt;
    R.outTif = outTif;
    R.keptList = keptList;
    R.manifestCsv = manifestCsv;

    fprintf('\n============================================================\n');
    fprintf('[%s] C001c clean + merge WGS84 HiddenMask %dm\n', river, resM);
    fprintf('Input folder : %s\n', inRoot);
    fprintf('Output folder: %s\n', outRoot);
    fprintf('============================================================\n');

    if exist(inRoot, 'dir') ~= 7
        msg = sprintf('Missing C001b WGS84 input folder: %s', inRoot);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            R.status = 'MISSING_INPUT_FOLDER';
            R.message = msg;
            return;
        end
    end

    % Detect source candidates.
    tileFiles = dir(fullfile(inTileDir, sprintf('HiddenMask_%dm_WGS84_tile_*.tif', resM)));
    if ~isempty(tileFiles)
        sourceMode = 'tiled';
        candidates = cell(numel(tileFiles), 1);
        for i = 1:numel(tileFiles)
            candidates{i} = fullfile(tileFiles(i).folder, tileFiles(i).name);
        end
    elseif exist(inSingleTif, 'file') == 2
        sourceMode = 'single_tif';
        candidates = {inSingleTif};
    elseif exist(inVrt, 'file') == 2
        sourceMode = 'vrt_only';
        candidates = {inVrt};
    else
        msg = sprintf('No WGS84 tif/tile/vrt found under: %s', inRoot);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            R.status = 'MISSING_INPUT_RASTER';
            R.message = msg;
            return;
        end
    end

    R.sourceMode = sourceMode;
    R.nCandidate = numel(candidates);

    fprintf('Source mode       : %s\n', sourceMode);
    fprintf('Candidate rasters : %d\n', numel(candidates));

    if strcmpi(sourceMode, 'single_tif') && cfg.skipSingleTif
        fprintf('[SKIP] Single whole-river WGS84 GeoTIFF already exists; no filtering/merging needed.\n');
        R.nKept = 1;
        R.nRemoved = 0;
        R.outTif = inSingleTif;
        R.outVrt = inVrt;
        R.status = 'SKIP_SINGLE_TIF_ALREADY_MERGED';
        R.message = 'C001b output is already one whole-river GeoTIFF; C001c does not modify it by default.';
        return;
    end

    if cfg.overwrite
        deleteIfExists(outVrt);
        deleteIfExists(outTif);
        if exist(keptList, 'file') == 2; delete(keptList); end
        if exist(manifestCsv, 'file') == 2; delete(manifestCsv); end
        if cfg.copyKeptTiles && exist(keptTileDir, 'dir') == 7
            oldCopies = dir(fullfile(keptTileDir, '*.tif'));
            for k = 1:numel(oldCopies)
                deleteIfExists(fullfile(oldCopies(k).folder, oldCopies(k).name));
            end
        end
    elseif exist(outTif, 'file') == 2
        fprintf('[SKIP] Existing merged output and overwrite=false: %s\n', outTif);
        R.status = 'SKIP_EXISTING';
        R.message = 'existing output and overwrite=false';
        return;
    end

    % Filter candidates.
    fidManifest = fopen(manifestCsv, 'w');
    if fidManifest < 0
        error('Cannot open manifest CSV for writing: %s', manifestCsv);
    end
    fprintf(fidManifest, 'river,res_m,source_mode,tile_path,computed_min,computed_max,keep,reason\n');

    fidList = fopen(keptList, 'w');
    if fidList < 0
        fclose(fidManifest);
        error('Cannot open kept tile list for writing: %s', keptList);
    end

    nKept = 0;
    nRemoved = 0;
    keptPaths = {};
    t0 = tic;

    for i = 1:numel(candidates)
        tilePath = candidates{i};
        [minVal, maxVal, statsOK, statsMsg] = computedMinMax(tilePath);
        [keep, reason] = decideKeep(minVal, maxVal, statsOK, statsMsg, cfg);

        if keep
            nKept = nKept + 1;
            if cfg.copyKeptTiles && strcmpi(sourceMode, 'tiled')
                dstTile = fullfile(keptTileDir, sprintf('keep_%06d_%s', nKept, getFileName(tilePath)));
                copyfile(tilePath, dstTile);
                copySidecars(tilePath, dstTile);
                keptPathForList = dstTile;
            else
                keptPathForList = tilePath;
            end
            keptPaths{end+1,1} = keptPathForList; %#ok<AGROW>
            fprintf(fidList, '%s\n', keptPathForList);
        else
            nRemoved = nRemoved + 1;
        end

        fprintf(fidManifest, '"%s",%g,"%s","%s",%s,%s,%d,"%s"\n', ...
            river, resM, sourceMode, tilePath, numOrNaN(minVal), numOrNaN(maxVal), keep, escapeCsv(reason));

        if mod(i, cfg.progressEvery) == 0 || i == numel(candidates)
            fprintf('  checked %d/%d, kept=%d, removed=%d, elapsed=%.1f min\n', ...
                i, numel(candidates), nKept, nRemoved, toc(t0)/60);
        end
    end

    fclose(fidManifest);
    fclose(fidList);

    R.nKept = nKept;
    R.nRemoved = nRemoved;

    fprintf('Filter finished: kept=%d, removed=%d\n', nKept, nRemoved);
    fprintf('Manifest: %s\n', manifestCsv);
    fprintf('Kept list: %s\n', keptList);

    if nKept == 0
        R.status = 'EMPTY_NO_RIVER_TILE';
        R.message = 'No tile contains value 1; no merged GeoTIFF generated.';
        warning('[%s %dm] No non-empty tile found. Output not created.', river, resM);
        return;
    end

    % Build non-empty VRT.
    buildNonEmptyVRT(keptList, outVrt, cfg);

    % Merge into one compressed/sparse GeoTIFF.
    if cfg.preserveFullGrid
        translateVRTToMergedTifPreserveGrid(outVrt, outTif, inVrt, cfg);
    else
        translateVRTToMergedTif(outVrt, outTif, cfg);
    end

    printRasterInfo(outTif);
    quickMaskStats(outTif, cfg);

    R.status = 'OK_CLEAN_MERGED_TIF';
    R.message = sprintf('kept %d of %d source rasters; merged single WGS84 GeoTIFF created', nKept, numel(candidates));
end

%% ============================================================
% Config and option parsing
% ============================================================

function cfg = defaultConfig()
    cfg.maskRoot = '/tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw';
    cfg.logRoot = fullfile(cfg.maskRoot, 'Logs');

    cfg.maskNoData = 255;
    cfg.keepValue = 1;       % river/hidden value to keep
    cfg.overwrite = true;
    cfg.stopOnMissing = true;
    cfg.stopOnError = true;

    cfg.copyKeptTiles = false;
    cfg.skipSingleTif = true;       % Single whole-river tif is already final; do not rewrite by default.
    cfg.preserveFullGrid = false;   % false = crop to kept-tile bounding box; true = preserve original WGS84 VRT grid extent/size.
    cfg.progressEvery = 20;

    % gdal_translate options for the final merged GeoTIFF.
    cfg.numThreads = 'ALL_CPUS';
    cfg.creationOptions = { ...
        'TILED=YES', ...
        'BLOCKXSIZE=512', ...
        'BLOCKYSIZE=512', ...
        'COMPRESS=LZW', ...
        'BIGTIFF=YES', ...
        'SPARSE_OK=YES'};
end

function cfg = parseOptions(cfg, varargin)
    if mod(numel(varargin), 2) ~= 0
        error('Options must be name-value pairs.');
    end
    for i = 1:2:numel(varargin)
        key = lower(char(varargin{i}));
        val = varargin{i+1};
        switch key
            case 'maskroot'
                cfg.maskRoot = char(val);
                cfg.logRoot = fullfile(cfg.maskRoot, 'Logs');
            case 'logroot'
                cfg.logRoot = char(val);
            case 'masknodata'
                cfg.maskNoData = double(val);
            case 'keepvalue'
                cfg.keepValue = double(val);
            case 'overwrite'
                cfg.overwrite = logical(val);
            case 'stoponmissing'
                cfg.stopOnMissing = logical(val);
            case 'stoponerror'
                cfg.stopOnError = logical(val);
            case 'copykepttiles'
                cfg.copyKeptTiles = logical(val);
            case 'skipsingletif'
                cfg.skipSingleTif = logical(val);
            case 'preservefullgrid'
                cfg.preserveFullGrid = logical(val);
            case 'progressevery'
                cfg.progressEvery = max(1, round(double(val)));
            case 'numthreads'
                if isnumeric(val)
                    cfg.numThreads = sprintf('%d', round(double(val)));
                else
                    cfg.numThreads = char(val);
                end
            otherwise
                error('Unknown option: %s', key);
        end
    end
end

%% ============================================================
% GDAL helpers
% ============================================================

function setupGdalPaths()
    try
        oldDir = pwd;
        cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
        GDALLoad();
        cd(oldDir);
    catch ME
        fprintf('[WARN] GDALLoad setup skipped or failed: %s\n', ME.message);
    end
end

function [minVal, maxVal, ok, msg] = computedMinMax(rasterPath)
    minVal = NaN;
    maxVal = NaN;
    ok = false;
    msg = '';

    cmd = sprintf('gdalinfo -mm %s', qpath(rasterPath));
    [status, txt] = system(cmd);
    if status ~= 0
        msg = sprintf('gdalinfo -mm failed: %s', strtrim(txt));
        return;
    end

    tok = regexp(txt, 'Computed Min/Max=([-+0-9\.eE]+),([-+0-9\.eE]+)', 'tokens', 'once');
    if isempty(tok)
        tok = regexp(txt, 'Minimum=([-+0-9\.eE]+), Maximum=([-+0-9\.eE]+)', 'tokens', 'once');
    end

    if isempty(tok)
        msg = 'No computed min/max found. This can happen for all-NoData rasters.';
        return;
    end

    minVal = str2double(tok{1});
    maxVal = str2double(tok{2});
    ok = isfinite(minVal) && isfinite(maxVal);
    if ~ok
        msg = 'Parsed min/max are not finite.';
    end
end

function [keep, reason] = decideKeep(minVal, maxVal, statsOK, statsMsg, cfg)
    if ~statsOK
        keep = false;
        reason = sprintf('remove: %s', statsMsg);
        return;
    end

    % Normal C001b mask values are 0/1/255. gdalinfo computed min/max should
    % ignore the 255 NoData value. Therefore, max >= 1 indicates a non-empty
    % tile. If max is 255, treat it as suspicious and keep it conservatively,
    % rather than risking loss of a real river tile.
    if maxVal >= (cfg.maskNoData - 0.5)
        keep = true;
        reason = sprintf('keep_suspicious_max_nodata_or_valid: min=%.6g max=%.6g', minVal, maxVal);
    elseif maxVal >= (cfg.keepValue - 0.5)
        keep = true;
        reason = sprintf('keep_has_value_%g: min=%.6g max=%.6g', cfg.keepValue, minVal, maxVal);
    else
        keep = false;
        reason = sprintf('remove_all_background_or_nodata: min=%.6g max=%.6g', minVal, maxVal);
    end
end

function buildNonEmptyVRT(keptList, outVrt, cfg)
    deleteIfExists(outVrt);
    cmd = sprintf('gdalbuildvrt -overwrite -srcnodata %d -vrtnodata %d -input_file_list %s %s', ...
        cfg.maskNoData, cfg.maskNoData, qpath(keptList), qpath(outVrt));
    fprintf('[Build non-empty VRT]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(outVrt, 'file') ~= 2
        error('gdalbuildvrt failed: %s', msg);
    end
end

function translateVRTToMergedTif(outVrt, outTif, cfg)
    deleteIfExists(outTif);

    co = '';
    for i = 1:numel(cfg.creationOptions)
        co = sprintf('%s -co %s', co, cfg.creationOptions{i});
    end

    % gdal_translate is streaming and usually safer than gdalwarp for merging
    % an already aligned VRT into one GeoTIFF.
    cmd = sprintf(['gdal_translate -of GTiff -ot Byte -a_nodata %d ', ...
                   '-co NUM_THREADS=%s %s %s %s'], ...
                   cfg.maskNoData, cfg.numThreads, co, qpath(outVrt), qpath(outTif));
    fprintf('[Translate VRT -> merged GeoTIFF]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(outTif, 'file') ~= 2
        error('gdal_translate failed: %s', msg);
    end
end

function translateVRTToMergedTifPreserveGrid(nonEmptyVrt, outTif, referenceVrt, cfg)
    % Preserve the full original C001b WGS84 grid. Areas corresponding to
    % removed empty tiles are written as NoData=255. This keeps dimensions
    % and extent consistent with the original WGS84 VRT, but the output can
    % be larger than the cropped default product.
    deleteIfExists(outTif);

    tmpJson = [tempname '.json'];
    info = getRasterInfoGDAL(referenceVrt, tmpJson);
    if exist(tmpJson, 'file') == 2
        delete(tmpJson);
    end

    co = '';
    for i = 1:numel(cfg.creationOptions)
        co = sprintf('%s -co %s', co, cfg.creationOptions{i});
    end

    cmd = sprintf([ ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-te %.12f %.12f %.12f %.12f ', ...
        '-ts %d %d ', ...
        '-r near -ot Byte ', ...
        '-srcnodata %d -dstnodata %d ', ...
        '-multi -wo NUM_THREADS=%s ', ...
        '%s %s %s'], ...
        info.xmin, info.ymin, info.xmax, info.ymax, info.cols, info.rows, ...
        cfg.maskNoData, cfg.maskNoData, cfg.numThreads, co, qpath(nonEmptyVrt), qpath(outTif));

    fprintf('[Warp non-empty VRT -> merged GeoTIFF preserving original grid]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(outTif, 'file') ~= 2
        error('gdalwarp preserve-full-grid merge failed: %s', msg);
    end
end

function info = getRasterInfoGDAL(rasterPath, jsonFile)
    cmd = sprintf('gdalinfo -json %s > %s', qpath(rasterPath), qpath(jsonFile));
    [status, msg] = system(cmd);
    if status ~= 0
        error('gdalinfo -json failed: %s', msg);
    end
    jtxt = fileread(jsonFile);
    j = jsondecode(jtxt);
    info.cols = double(j.size(1));
    info.rows = double(j.size(2));
    if ~isfield(j, 'geoTransform') || numel(j.geoTransform) < 6
        error('No geoTransform found in gdalinfo json for: %s', rasterPath);
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
end

function printRasterInfo(rasterPath)
    fprintf('[Raster info]\n');
    cmd = sprintf('gdalinfo %s | grep -E "Size is|Coordinate System is|GEOGCS|GCS_WGS_1984|WGS 84|ID\\[\\"EPSG\\"|Pixel Size|Upper Left|Lower Right|NoData"', qpath(rasterPath));
    system(cmd);
end

function quickMaskStats(rasterPath, cfg)
    % Use exact min/max instead of -approx_stats. For very sparse merged
    % mask rasters, GDAL approximate sampling can miss all valid pixels and
    % incorrectly print "no valid pixels found in sampling" even when the
    % raster contains hidden=1 cells. Exact min/max is slower, but reliable
    % for this final QA step.
    fprintf('[Exact mask min/max QA]\n');

    cmd = sprintf('gdalinfo -mm %s | grep -E "Computed Min/Max|Minimum|Maximum|NoData|STATISTICS"', qpath(rasterPath));
    system(cmd);

    [minVal, maxVal, ok, msg] = computedMinMax(rasterPath);
    if ok
        fprintf('Computed min/max parsed: min=%.6g, max=%.6g\n', minVal, maxVal);
        if maxVal >= (cfg.keepValue - 0.5)
            fprintf('[OK] Output contains hidden-mask value >= %.3g.\n', cfg.keepValue);
        else
            fprintf('[WARN] Output max < %.3g. This merged output may contain no hidden=1 pixels.\n', cfg.keepValue);
        end
    else
        fprintf('[WARN] Exact min/max failed: %s\n', msg);
    end

    fprintf('Expected values: hidden=1, background=0, NoData(meta)=%d\n', cfg.maskNoData);
end

%% ============================================================
% Utility helpers
% ============================================================

function rivers = scanRiverFolders(cfg, targetResM)
    if isempty(targetResM)
        targetResM = 1;
    end
    resM = targetResM(1);
    root = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm_WGS84', resM));
    if exist(root, 'dir') ~= 7
        rivers = {};
        fprintf('[WARN] Cannot scan rivers because folder is missing: %s\n', root);
        return;
    end
    D = dir(root);
    rivers = {};
    for i = 1:numel(D)
        if D(i).isdir && ~startsWith(D(i).name, '.') && ~startsWith(D(i).name, '_')
            rivers{end+1,1} = D(i).name; %#ok<AGROW>
        end
    end
    rivers = sort(rivers);
end

function printRiverList(rivers)
    fprintf('\nRivers found under C001b WGS84 folders:\n');
    if isempty(rivers)
        fprintf('  [none found]\n');
    else
        for i = 1:numel(rivers)
            fprintf('  %2d. %s\n', i, rivers{i});
        end
    end
    fprintf('\n');
end

function R = emptyResult(river, resM, cfg) %#ok<INUSD>
    R = struct();
    R.river = river;
    R.resM = resM;
    R.sourceMode = '';
    R.nCandidate = 0;
    R.nKept = 0;
    R.nRemoved = 0;
    R.outRoot = '';
    R.outVrt = '';
    R.outTif = '';
    R.keptList = '';
    R.manifestCsv = '';
    R.status = '';
    R.message = '';
end

function ensureDir(d)
    if exist(d, 'dir') ~= 7
        mkdir(d);
    end
end

function deleteIfExists(path0)
    [folder, base, ext] = fileparts(path0);
    if isempty(ext)
        if exist(path0, 'dir') == 7
            rmdir(path0, 's');
        end
        return;
    end
    if strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        exts = {ext, [ext '.aux.xml'], '.aux.xml', '.ovr'};
    elseif strcmpi(ext, '.vrt')
        exts = {'.vrt', '.vrt.aux.xml'};
    else
        exts = {ext};
    end
    for i = 1:numel(exts)
        f = fullfile(folder, [base exts{i}]);
        if exist(f, 'file') == 2
            delete(f);
        end
    end
end

function copySidecars(srcTif, dstTif)
    sidecars = {[srcTif '.aux.xml'], [srcTif '.ovr']};
    for i = 1:numel(sidecars)
        if exist(sidecars{i}, 'file') == 2
            [~, dstBase, dstExt] = fileparts(dstTif);
            dstSidecar = fullfile(fileparts(dstTif), [dstBase dstExt strrep(sidecars{i}, srcTif, '')]);
            try
                copyfile(sidecars{i}, dstSidecar);
            catch
            end
        end
    end
end

function s = qpath(x)
    s = ['"', char(x), '"'];
end

function name = getFileName(path0)
    [~, b, e] = fileparts(path0);
    name = [b e];
end

function s = numOrNaN(x)
    if isfinite(x)
        s = sprintf('%.12g', x);
    else
        s = 'NaN';
    end
end

function s = escapeCsv(s)
    s = strrep(char(s), '"', '''');
    s = strrep(s, sprintf('\n'), ' ');
    s = strrep(s, sprintf('\r'), ' ');
end

function T = rowsToTable(rows)
    if isempty(rows)
        T = table();
        return;
    end
    T = cell2table(rows, 'VariableNames', { ...
        'river','res_m','sourceMode','nCandidate','nKept','nRemoved', ...
        'outVrt','outTif','manifestCsv','keptList','status','message'});
end

function writeSummaryCSV(summaryFile, rows)
    ensureDir(fileparts(summaryFile));
    fid = fopen(summaryFile, 'w');
    if fid < 0
        warning('Cannot write summary CSV: %s', summaryFile);
        return;
    end
    fprintf(fid, 'river,res_m,sourceMode,nCandidate,nKept,nRemoved,outVrt,outTif,manifestCsv,keptList,status,message\n');
    for i = 1:size(rows,1)
        fprintf(fid, '"%s",%g,"%s",%g,%g,%g,"%s","%s","%s","%s","%s","%s"\n', ...
            rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, ...
            rows{i,7}, rows{i,8}, rows{i,9}, rows{i,10}, rows{i,11}, escapeCsv(rows{i,12}));
    end
    fclose(fid);
end

function s = safeName(s)
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if numel(s) > 120
        s = s(1:120);
    end
end
