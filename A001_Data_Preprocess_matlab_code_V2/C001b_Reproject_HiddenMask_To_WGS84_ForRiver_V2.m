function Summary = C001b_Reproject_HiddenMask_To_WGS84_ForRiver_V2(riverName, targetResM, varargin)
% C001b_Reproject_HiddenMask_To_WGS84_ForRiver_V2
% 2026-07-14
%
% V2 update
%   - In auto mode, detect whether the source C001 output is tiled by
%     scanning HiddenMask_<res>m_tiles/*.tif under each river folder.
%   - If source is tiled, use tiled WGS84 output automatically.
%   - If source is single, use single WGS84 output unless the WGS84 output
%     grid exceeds autoTilePixelThreshold.
%
% Purpose
%   Reproject existing C001 HiddenMask rasters from bathy-grid CRS to
%   WGS84 geographic coordinates (EPSG:4326 / GCS_WGS_1984).
%
%   This script does NOT re-rasterize Hidden_Mask.shp. It starts from the
%   already-created HiddenMask_<res>m.vrt/tif produced by C001.
%
% Input
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/
%       HiddenMask_ByRiver_<res>m/<river>/HiddenMask_<res>m.vrt
%   If the VRT does not exist, it tries HiddenMask_<res>m.tif.
%
% Output
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/
%       HiddenMask_ByRiver_<res>m_WGS84/<river>/HiddenMask_<res>m_WGS84.tif
%       HiddenMask_ByRiver_<res>m_WGS84/<river>/HiddenMask_<res>m_WGS84.vrt
%
% For very large rivers, output is tiled:
%   HiddenMask_ByRiver_<res>m_WGS84/<river>/HiddenMask_<res>m_WGS84_tiles/*.tif
%   HiddenMask_ByRiver_<res>m_WGS84/<river>/HiddenMask_<res>m_WGS84.vrt
%
% Mask values
%   1   = hidden area
%   0   = non-hidden background
%   255 = NoData metadata value
%
% Notes
%   - Output CRS is EPSG:4326, which ArcMap should show as GCS_WGS_1984.
%   - Nearest-neighbor resampling is used, so values remain categorical.
%   - 0 remains a valid background value, not NoData.
%   - For tiled mode, a global WGS84 VRT grid is first created only for
%     georeference/extent. Then output tiles are warped to that same grid
%     and mosaicked with gdalbuildvrt.
%
% Usage
%   C001b_Reproject_HiddenMask_To_WGS84_ForRiver('BadgerFinNull', 1);
%   C001b_Reproject_HiddenMask_To_WGS84_ForRiver('ALL', 1);
%   C001b_Reproject_HiddenMask_To_WGS84_ForRiver('ALL', [1 3 5 10]);
%   C001b_Reproject_HiddenMask_To_WGS84_ForRiver('CA_KlamathRiver_TopoBathy_2018_D18', 1, 'outputMode', 'tiled');
%   C001b_Reproject_HiddenMask_To_WGS84_ForRiver('LIST');

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

validRivers = selectedRivers();

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

targetResM = unique(double(targetResM(:)'));

for i = 1:numel(riversToRun)
    if ~ismember(riversToRun{i}, validRivers)
        fprintf('\nUnknown river: %s\n', riversToRun{i});
        printRiverList(validRivers);
        error('River is not in selected valid river list.');
    end
end

fprintf('\n============================================================\n');
fprintf('C001b Reproject existing HiddenMask rasters to WGS84\n');
fprintf('Input root  : %s/HiddenMask_ByRiver_<res>m/<river>\n', cfg.maskRoot);
fprintf('Output root : %s/HiddenMask_ByRiver_<res>m_WGS84/<river>\n', cfg.maskRoot);
fprintf('Target CRS  : %s\n', cfg.targetSRS);
fprintf('Rivers      : %s\n', strjoin(riversToRun, ', '));
fprintf('Resolutions : %s m\n', mat2str(targetResM));
fprintf('Output mode : %s (auto is source-layout-aware)\n', cfg.outputMode);
fprintf('Tile size   : %d target pixels, auto threshold = %.3g pixels\n', cfg.tileSize, cfg.autoTilePixelThreshold);
fprintf('Resampling  : %s\n', cfg.resampling);
fprintf('NoData      : %d\n', cfg.maskNoData);
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

        inRoot = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm', resM), river);
        inVrt  = fullfile(inRoot, sprintf('HiddenMask_%dm.vrt', resM));
        inTif  = fullfile(inRoot, sprintf('HiddenMask_%dm.tif', resM));
        srsWkt = fullfile(inRoot, 'target_bathy_srs.wkt');

        if exist(inVrt, 'file') == 2
            inRaster = inVrt;
        elseif exist(inTif, 'file') == 2
            inRaster = inTif;
        else
            msg = sprintf('Missing input HiddenMask raster: %s or %s', inVrt, inTif);
            if cfg.stopOnMissing
                error(msg);
            else
                warning(msg);
                rowId = rowId + 1;
                summaryRows(rowId,:) = {river, resM, '', '', '', '', 'MISSING_INPUT', msg}; %#ok<AGROW>
                continue;
            end
        end

        sourceLayout = detectSourceLayout(inRoot, resM);

        outRoot = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm_WGS84', resM), river);
        ensureDir(outRoot);
        outTif = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84.tif', resM));
        outVrt = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84.vrt', resM));
        gridVrt = fullfile(outRoot, sprintf('_HiddenMask_%dm_WGS84_global_grid.vrt', resM));
        infoJson = fullfile(outRoot, sprintf('_HiddenMask_%dm_WGS84_global_grid_gdalinfo.json', resM));

        fprintf('\n============================================================\n');
        fprintf('[%s] C001b reproject HiddenMask %dm to WGS84\n', river, resM);
        fprintf('Input : %s\n', inRaster);
        fprintf('Source layout detected: %s\n', sourceLayout);
        fprintf('Output: %s\n', outRoot);
        fprintf('============================================================\n');

        try
            if cfg.overwrite
                deleteIfExists(outTif);
                deleteIfExists(outVrt);
                deleteIfExists(gridVrt);
            elseif exist(outVrt, 'file') == 2 || exist(outTif, 'file') == 2
                fprintf('[SKIP] Existing WGS84 output found and overwrite=false.\n');
                statusText = 'SKIP_EXISTING';
                msg = 'existing output and overwrite=false';
                rowId = rowId + 1;
                summaryRows(rowId,:) = {river, resM, inRaster, outTif, outVrt, '', statusText, msg}; %#ok<AGROW>
                continue;
            end

            sourceSrsArg = sourceSRSArg(cfg, srsWkt);

            % Step 1: create a WGS84 VRT grid. This is cheap and gives a
            % common target extent/resolution for both single and tiled outputs.
            createGlobalWgs84GridVRT(inRaster, gridVrt, sourceSrsArg, cfg);
            info = getRasterInfoGDAL(gridVrt, infoJson);
            fprintf('WGS84 grid rows/cols : %d / %d\n', info.rows, info.cols);
            fprintf('WGS84 BBox           : %.12f %.12f %.12f %.12f\n', info.xmin, info.ymin, info.xmax, info.ymax);
            fprintf('WGS84 Pixel Size     : %.12g / %.12g degrees\n', info.xres, abs(info.yres));

            modeUsed = chooseOutputMode(cfg, info, sourceLayout);
            fprintf('Mode used            : %s\n', modeUsed);

            if strcmpi(modeUsed, 'single')
                reprojectSingle(inRaster, outTif, sourceSrsArg, info, cfg);
                buildVRTFromSingleTif(outTif, outVrt, cfg);
                statusText = 'OK_SINGLE';
                outTifSummary = outTif;
                msg = 'single GeoTIFF + VRT, WGS84';
            else
                outVrt = reprojectTiled(inRaster, outRoot, outVrt, sourceSrsArg, info, resM, cfg);
                outTifSummary = '';
                statusText = 'OK_TILED_VRT';
                msg = 'tiled GeoTIFFs + WGS84 VRT';
            end

            printRasterInfo(outVrt);
            quickMaskStats(outVrt, cfg);

        catch ME
            if cfg.stopOnError
                rethrow(ME);
            end
            warning('[%s %dm] C001b failed: %s', river, resM, ME.message);
            outTifSummary = '';
            statusText = 'FAIL';
            msg = ME.message;
        end

        rowId = rowId + 1;
        summaryRows(rowId,:) = {river, resM, inRaster, outTifSummary, outVrt, modeUsedSafe(statusText), statusText, msg}; %#ok<AGROW>
    end
end

Summary = rowsToTable(summaryRows);
summaryFile = fullfile(cfg.logRoot, sprintf('C001b_HiddenMask_Reproject_WGS84_%s_%s.csv', safeName(strjoin(riversToRun, '__')), datestr(now, 'yyyymmdd_HHMMSS')));
writeSummaryCSV(summaryFile, summaryRows);

fprintf('\n============================================================\n');
fprintf('C001b HiddenMask WGS84 reprojection finished. Summary written:\n%s\n', summaryFile);
fprintf('Outputs are under:\n%s/HiddenMask_ByRiver_<res>m_WGS84/<river>/\n', cfg.maskRoot);
fprintf('============================================================\n');

end

%% ============================================================
% Local functions
% ============================================================

function cfg = defaultConfig()
    cfg.maskRoot = '/tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw';
    cfg.logRoot = fullfile(cfg.maskRoot, 'Logs');
    cfg.targetSRS = 'EPSG:4326';
    cfg.resampling = 'near';
    cfg.maskNoData = 255;
    cfg.overwrite = true;
    cfg.stopOnMissing = true;
    cfg.stopOnError = true;
    cfg.forceSourceSRS = true;
    cfg.numThreads = 'ALL_CPUS';
    cfg.warpMemoryMB = 512;

    % outputMode:
    %   'single' = write one HiddenMask_<res>m_WGS84.tif + VRT
    %   'tiled'  = write HiddenMask_<res>m_WGS84_tiles/*.tif + VRT
    %   'auto'   = use tiled when WGS84 output has more pixels than threshold
    cfg.outputMode = 'auto';
    cfg.autoTilePixelThreshold = 2.0e9;
    cfg.tileSize = 10000;  % target WGS84 pixels per tile dimension
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
            case 'targetsrs'
                cfg.targetSRS = char(val);
            case 'resampling'
                cfg.resampling = char(val);
            case 'overwrite'
                cfg.overwrite = logical(val);
            case 'stoponmissing'
                cfg.stopOnMissing = logical(val);
            case 'stoponerror'
                cfg.stopOnError = logical(val);
            case 'forcesourcesrs'
                cfg.forceSourceSRS = logical(val);
            case 'numthreads'
                if isnumeric(val)
                    cfg.numThreads = sprintf('%d', val);
                else
                    cfg.numThreads = char(val);
                end
            case 'warpmemorymb'
                cfg.warpMemoryMB = round(double(val));
            case 'outputmode'
                cfg.outputMode = lower(char(val));
            case 'autotilepixelthreshold'
                cfg.autoTilePixelThreshold = double(val);
            case 'tilesize'
                cfg.tileSize = round(double(val));
            otherwise
                error('Unknown option: %s', key);
        end
    end

    if ~ismember(lower(cfg.outputMode), {'auto','single','tiled'})
        error('outputMode must be auto, single, or tiled.');
    end
    if cfg.tileSize <= 0
        error('tileSize must be positive.');
    end
end

function rivers = selectedRivers()
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

function printRiverList(rivers)
    fprintf('\nSelected valid rivers for C001b HiddenMask WGS84 reprojection:\n');
    for i = 1:numel(rivers)
        fprintf('  %2d. %s\n', i, rivers{i});
    end
    fprintf('\n');
end

function setupGdalPaths()
    try
        cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
        GDALLoad();
    catch ME
        fprintf('[WARN] GDALLoad setup skipped or failed: %s\n', ME.message);
    end
end

function ensureDir(d)
    if exist(d, 'dir') ~= 7
        mkdir(d);
    end
end

function sourceSrsArg = sourceSRSArg(cfg, srsWkt)
    sourceSrsArg = '';
    if cfg.forceSourceSRS && exist(srsWkt, 'file') == 2
        sourceSrsArg = sprintf('-s_srs %s ', qpath(srsWkt));
        fprintf('Source SRS forced from: %s\n', srsWkt);
    elseif cfg.forceSourceSRS
        fprintf('[WARN] forceSourceSRS=true but missing WKT: %s\n', srsWkt);
    end
end

function createGlobalWgs84GridVRT(inRaster, gridVrt, sourceSrsArg, cfg)
    deleteIfExists(gridVrt);
    cmd = sprintf([ ...
        'gdalwarp -overwrite -of VRT ', ...
        '%s', ...
        '-t_srs %s ', ...
        '-r %s ', ...
        '-ot Byte ', ...
        '-srcnodata %d -dstnodata %d ', ...
        '%s %s'], ...
        sourceSrsArg, cfg.targetSRS, cfg.resampling, cfg.maskNoData, cfg.maskNoData, ...
        qpath(inRaster), qpath(gridVrt));
    fprintf('[Build global WGS84 grid VRT]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(gridVrt, 'file') ~= 2
        error('Failed to build global WGS84 grid VRT: %s', msg);
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
    info.geoTransform = gt;

    x0 = gt(1); px = gt(2); rx = gt(3);
    y0 = gt(4); ry = gt(5); py = gt(6);

    xs = [x0, x0 + info.cols*px, x0 + info.rows*rx, x0 + info.cols*px + info.rows*rx];
    ys = [y0, y0 + info.cols*ry, y0 + info.rows*py, y0 + info.cols*ry + info.rows*py];

    info.xmin = min(xs);
    info.xmax = max(xs);
    info.ymin = min(ys);
    info.ymax = max(ys);
    info.xres = px;
    info.yres = py;
end

function sourceLayout = detectSourceLayout(inRoot, resM)
    tileDir = fullfile(inRoot, sprintf('HiddenMask_%dm_tiles', resM));
    tileList = fullfile(tileDir, sprintf('HiddenMask_%dm_tile_list.txt', resM));
    tileFiles = dir(fullfile(tileDir, sprintf('HiddenMask_%dm_tile_*.tif', resM)));

    singleTif = fullfile(inRoot, sprintf('HiddenMask_%dm.tif', resM));

    if exist(tileDir, 'dir') == 7 && (~isempty(tileFiles) || exist(tileList, 'file') == 2)
        sourceLayout = 'tiled';
    elseif exist(singleTif, 'file') == 2
        sourceLayout = 'single';
    else
        sourceLayout = 'vrt_only';
    end
end

function modeUsed = chooseOutputMode(cfg, info, sourceLayout)
    switch lower(cfg.outputMode)
        case 'single'
            modeUsed = 'single';
        case 'tiled'
            modeUsed = 'tiled';
        otherwise
            % Auto mode: first respect the source C001 layout. If the source
            % HiddenMask was already tiled, keep the WGS84 output tiled too.
            % Otherwise write a single GeoTIFF unless the target WGS84 grid is
            % very large. This avoids manually specifying outputMode for big
            % rivers such as CA/CO/NE.
            if strcmpi(sourceLayout, 'tiled')
                modeUsed = 'tiled';
                return;
            end

            nPix = double(info.rows) * double(info.cols);
            if nPix > cfg.autoTilePixelThreshold
                modeUsed = 'tiled';
            else
                modeUsed = 'single';
            end
    end
end

function reprojectSingle(inRaster, outTif, sourceSrsArg, info, cfg)
    deleteIfExists(outTif);

    cmd = sprintf([ ...
        'gdalwarp -overwrite ', ...
        '%s', ...
        '-t_srs %s ', ...
        '-te %.12f %.12f %.12f %.12f ', ...
        '-ts %d %d ', ...
        '-r %s ', ...
        '-ot Byte ', ...
        '-srcnodata %d -dstnodata %d ', ...
        '-multi -wo NUM_THREADS=%s -wm %d ', ...
        '-co TILED=YES ', ...
        '-co BLOCKXSIZE=512 -co BLOCKYSIZE=512 ', ...
        '-co COMPRESS=LZW ', ...
        '-co BIGTIFF=YES ', ...
        '-co SPARSE_OK=YES ', ...
        '%s %s'], ...
        sourceSrsArg, cfg.targetSRS, ...
        info.xmin, info.ymin, info.xmax, info.ymax, info.cols, info.rows, ...
        cfg.resampling, cfg.maskNoData, cfg.maskNoData, cfg.numThreads, cfg.warpMemoryMB, ...
        qpath(inRaster), qpath(outTif));

    fprintf('[gdalwarp single]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(outTif, 'file') ~= 2
        error('gdalwarp single failed: %s', msg);
    end
end

function outVrt = reprojectTiled(inRaster, outRoot, outVrt, sourceSrsArg, info, resM, cfg)
    tileDir = fullfile(outRoot, sprintf('HiddenMask_%dm_WGS84_tiles', resM));
    ensureDir(tileDir);
    tileList = fullfile(tileDir, sprintf('HiddenMask_%dm_WGS84_tile_list.txt', resM));

    if cfg.overwrite
        oldTiles = dir(fullfile(tileDir, sprintf('HiddenMask_%dm_WGS84_tile_*.tif', resM)));
        for i = 1:numel(oldTiles)
            deleteIfExists(fullfile(oldTiles(i).folder, oldTiles(i).name));
        end
        if exist(tileList, 'file') == 2; delete(tileList); end
        deleteIfExists(outVrt);
    end

    fid = fopen(tileList, 'w');
    if fid < 0
        error('Cannot open tile list: %s', tileList);
    end

    nTiles = 0;
    t0 = tic;
    fprintf('[gdalwarp tiled] target tileSize=%d pixels\n', cfg.tileSize);

    for r0 = 0:cfg.tileSize:(info.rows-1)
        r1 = min(r0 + cfg.tileSize, info.rows);
        tileRows = r1 - r0;

        for c0 = 0:cfg.tileSize:(info.cols-1)
            c1 = min(c0 + cfg.tileSize, info.cols);
            tileCols = c1 - c0;

            % Use the global WGS84 grid geotransform, so all tiles align.
            [xMinTile, yMinTile, xMaxTile, yMaxTile] = tileExtentFromGrid(info, r0, r1, c0, c1);

            tileName = sprintf('HiddenMask_%dm_WGS84_tile_r%06d_c%06d.tif', resM, r0, c0);
            tilePath = fullfile(tileDir, tileName);
            deleteIfExists(tilePath);

            cmd = sprintf([ ...
                'gdalwarp -overwrite ', ...
                '%s', ...
                '-t_srs %s ', ...
                '-te %.12f %.12f %.12f %.12f ', ...
                '-ts %d %d ', ...
                '-r %s ', ...
                '-ot Byte ', ...
                '-srcnodata %d -dstnodata %d ', ...
                '-multi -wo NUM_THREADS=%s -wm %d ', ...
                '-co TILED=YES ', ...
                '-co BLOCKXSIZE=512 -co BLOCKYSIZE=512 ', ...
                '-co COMPRESS=LZW ', ...
                '-co BIGTIFF=YES ', ...
                '-co SPARSE_OK=YES ', ...
                '%s %s'], ...
                sourceSrsArg, cfg.targetSRS, ...
                xMinTile, yMinTile, xMaxTile, yMaxTile, tileCols, tileRows, ...
                cfg.resampling, cfg.maskNoData, cfg.maskNoData, cfg.numThreads, cfg.warpMemoryMB, ...
                qpath(inRaster), qpath(tilePath));

            fprintf('[Tile %d] r=%d:%d c=%d:%d, size=%dx%d\n', nTiles+1, r0, r1, c0, c1, tileRows, tileCols);
            [status, msg] = system(cmd, '-echo');
            if status ~= 0 || exist(tilePath, 'file') ~= 2
                fclose(fid);
                error('gdalwarp tile failed: %s\n%s', tilePath, msg);
            end

            fprintf(fid, '%s\n', tilePath);
            nTiles = nTiles + 1;
        end
    end

    fclose(fid);

    cmdVRT = sprintf('gdalbuildvrt -overwrite -srcnodata %d -vrtnodata %d -input_file_list %s %s', ...
        cfg.maskNoData, cfg.maskNoData, qpath(tileList), qpath(outVrt));
    fprintf('[Build WGS84 tiled VRT]\n%s\n', cmdVRT);
    [statusVRT, msgVRT] = system(cmdVRT, '-echo');
    if statusVRT ~= 0 || exist(outVrt, 'file') ~= 2
        error('gdalbuildvrt tiled failed: %s', msgVRT);
    end

    fprintf('[gdalwarp tiled] Done. nTiles=%d, elapsed=%.1f min\n', nTiles, toc(t0)/60);
end

function [xmin, ymin, xmax, ymax] = tileExtentFromGrid(info, r0, r1, c0, c1)
    gt = info.geoTransform;
    x0 = gt(1); px = gt(2); rx = gt(3);
    y0 = gt(4); ry = gt(5); py = gt(6);

    xs = [x0 + c0*px + r0*rx, x0 + c1*px + r0*rx, x0 + c0*px + r1*rx, x0 + c1*px + r1*rx];
    ys = [y0 + c0*ry + r0*py, y0 + c1*ry + r0*py, y0 + c0*ry + r1*py, y0 + c1*ry + r1*py];

    xmin = min(xs);
    xmax = max(xs);
    ymin = min(ys);
    ymax = max(ys);
end

function buildVRTFromSingleTif(outTif, outVrt, cfg)
    deleteIfExists(outVrt);
    cmd = sprintf('gdalbuildvrt -overwrite -srcnodata %d -vrtnodata %d %s %s', ...
        cfg.maskNoData, cfg.maskNoData, qpath(outVrt), qpath(outTif));
    fprintf('[Build VRT]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(outVrt, 'file') ~= 2
        error('gdalbuildvrt failed: %s', msg);
    end
end

function printRasterInfo(rasterPath)
    if exist(rasterPath, 'file') ~= 2
        fprintf('[WARN] Missing raster: %s\n', rasterPath);
        return;
    end
    fprintf('[Raster info]\n');
    cmd = sprintf('gdalinfo %s | grep -E "Size is|Coordinate System is|GEOGCS|GCS_WGS_1984|WGS 84|ID\\[\"EPSG\"|Pixel Size|Upper Left|Lower Right|NoData"', qpath(rasterPath));
    system(cmd);
end

function quickMaskStats(rasterPath, cfg)
    fprintf('[Quick mask stats]\n');
    cmd = sprintf('gdalinfo -approx_stats %s | grep -E "Minimum|Maximum|Mean|StdDev|NoData|STATISTICS"', qpath(rasterPath));
    system(cmd);
    fprintf('Expected values: hidden=1, background=0, NoData(meta)=%d\n', cfg.maskNoData);
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

function s = qpath(x)
    s = ['"', char(x), '"'];
end

function T = rowsToTable(rows)
    if isempty(rows)
        T = table();
        return;
    end
    T = cell2table(rows, 'VariableNames', {'river','res_m','inRaster','outTif','outVrt','mode','status','message'});
end

function writeSummaryCSV(summaryFile, rows)
    ensureDir(fileparts(summaryFile));
    fid = fopen(summaryFile, 'w');
    if fid < 0
        warning('Cannot write summary CSV: %s', summaryFile);
        return;
    end
    fprintf(fid, 'river,res_m,inRaster,outTif,outVrt,mode,status,message\n');
    for i = 1:size(rows,1)
        fprintf(fid, '"%s",%g,"%s","%s","%s","%s","%s","%s"\n', ...
            rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, rows{i,7}, strrep(rows{i,8}, '"', ''''));
    end
    fclose(fid);
end

function s = safeName(s)
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if numel(s) > 120
        s = s(1:120);
    end
end

function m = modeUsedSafe(statusText)
    if contains(statusText, 'TILED')
        m = 'tiled';
    elseif contains(statusText, 'SINGLE')
        m = 'single';
    else
        m = '';
    end
end
