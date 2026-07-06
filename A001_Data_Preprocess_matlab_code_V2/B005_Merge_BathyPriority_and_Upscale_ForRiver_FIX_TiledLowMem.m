function B005_Merge_BathyPriority_and_Upscale_ForRiver_FIX_TiledLowMem(rivers, upscaleRes, varargin)
% B005_Merge_BathyPriority_ByVRT_Then_Upscale_ForRiver_FIX2_TiledLowMem
%
% FIX2_TiledLowMem:
%   - Keep original/simple merge logic: gdalbuildvrt, 3DEP first, Bathy second.
%   - Avoid MATLAB/Python raster writing.
%   - For very large 3/5/10 m outputs, write tiled GeoTIFFs and build a VRT.
%     This avoids status 137 / Killed from one huge gdalwarp job.
%
% Merge logic:
%   valid Bathy wins; Bathy NoData (-999999) is filled by valid 3DEP;
%   both NoData -> -999999. Zero is NOT treated as NoData.
%
% Usage:
%   B005_Merge_BathyPriority_ByVRT_Then_Upscale_ForRiver_FIX2_TiledLowMem('CA_KlamathRiver_TopoBathy_2018_D18');
%   B005_Merge_BathyPriority_ByVRT_Then_Upscale_ForRiver_FIX2_TiledLowMem('CA_KlamathRiver_TopoBathy_2018_D18',[3 5 10]);
%   B005_Merge_BathyPriority_ByVRT_Then_Upscale_ForRiver_FIX2_TiledLowMem('ALL');
%   B005_Merge_BathyPriority_ByVRT_Then_Upscale_ForRiver_FIX2_TiledLowMem('LIST');
%
% Key options:
%   'upscaleMode'              'auto' | 'single' | 'tiled'
%   'largeOutputPixelThreshold' 2e8
%   'coarseTilePixels'          4096
%   'warpMemoryMB'              256
%   'numThreads'                1   % safer for large VRT/average warps
%
% Notes:
%   This version is intended for large rivers such as CA/CO/MD where one
%   monolithic gdalwarp can be killed by the system OOM killer.

if nargin < 2 || isempty(upscaleRes)
    upscaleRes = [3 5 10];
end

p = inputParser;
addRequired(p, 'rivers');
addOptional(p, 'upscaleRes', [3 5 10], @(x) isnumeric(x) || isempty(x));
addParameter(p, 'rootPR', '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', @(x) ischar(x) || isstring(x));
addParameter(p, 'bathyRoot', 'Bathy_1m_FixND', @(x) ischar(x) || isstring(x));
addParameter(p, 'demRoot', '3DEP_1m_ResampleClip', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRootPattern', 'Bathy3DEP_Merged_%dm_FixND', @(x) ischar(x) || isstring(x));
addParameter(p, 'globalND', -999999, @isnumeric);
addParameter(p, 'upscaleAlg', 'average', @(x) ischar(x) || isstring(x));
addParameter(p, 'numThreads', 1, @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doUpscale', true, @islogical);
addParameter(p, 'upscaleMode', 'auto', @(x) ismember(lower(char(x)), {'auto','single','tiled'}));
addParameter(p, 'largeOutputPixelThreshold', 2e8, @isnumeric);
addParameter(p, 'coarseTilePixels', 4096, @isnumeric);
addParameter(p, 'warpMemoryMB', 256, @isnumeric);
addParameter(p, 'gdalCacheMB', 256, @isnumeric);
addParameter(p, 'envPrefix', 'export PROJ_USE_PROJ4_INIT_RULES=YES; export PROJ_NETWORK=ON;', @(x) ischar(x) || isstring(x));
parse(p, rivers, upscaleRes, varargin{:});

cfg = p.Results;
cfg.rootPR = char(cfg.rootPR);
cfg.bathyRoot = char(cfg.bathyRoot);
cfg.demRoot = char(cfg.demRoot);
cfg.outRootPattern = char(cfg.outRootPattern);
cfg.upscaleAlg = char(cfg.upscaleAlg);
cfg.upscaleMode = lower(char(cfg.upscaleMode));
cfg.envPrefix = char(cfg.envPrefix);
cfg.globalND = double(cfg.globalND);
cfg.numThreads = max(1, round(double(cfg.numThreads)));
cfg.coarseTilePixels = max(512, round(double(cfg.coarseTilePixels)));
cfg.warpMemoryMB = max(64, round(double(cfg.warpMemoryMB)));
cfg.gdalCacheMB = max(64, round(double(cfg.gdalCacheMB)));

allRivers = selectedRiverList();

if ischar(rivers) || isstring(rivers)
    r0 = char(rivers);
    if strcmpi(r0, 'LIST')
        fprintf('Selected rivers:\n');
        for i = 1:numel(allRivers)
            fprintf('  %2d. %s\n', i, allRivers{i});
        end
        return;
    elseif strcmpi(r0, 'ALL')
        riverList = allRivers;
    else
        riverList = {r0};
    end
elseif iscell(rivers)
    riverList = rivers;
else
    error('rivers must be char/string/cellstr.');
end

fprintf('\n============================================================\n');
fprintf('B005 bathy-priority merge by VRT, then upscale -- FIX2 TILED LOWMEM\n');
fprintf('Processed root : %s\n', cfg.rootPR);
fprintf('Bathy root     : %s\n', fullfile(cfg.rootPR, cfg.bathyRoot));
fprintf('3DEP root      : %s\n', fullfile(cfg.rootPR, cfg.demRoot));
fprintf('Output pattern : %s\n', fullfile(cfg.rootPR, cfg.outRootPattern));
fprintf('Rivers         : %s\n', strjoin(riverList, ', '));
fprintf('Upscale res    : [%s] m\n', num2str(cfg.upscaleRes));
fprintf('Merge method   : gdalbuildvrt, order = 3DEP first, Bathy second\n');
fprintf('Upscale alg    : %s\n', cfg.upscaleAlg);
fprintf('Upscale mode   : %s\n', cfg.upscaleMode);
fprintf('Large threshold: %.3g output pixels\n', cfg.largeOutputPixelThreshold);
fprintf('Tile pixels    : %d x %d output pixels\n', cfg.coarseTilePixels, cfg.coarseTilePixels);
fprintf('Warp memory    : %d MB\n', cfg.warpMemoryMB);
fprintf('GDAL cache     : %d MB\n', cfg.gdalCacheMB);
fprintf('GDAL thread    : %d\n', cfg.numThreads);
fprintf('Overwrite      : %d\n', cfg.overwrite);
fprintf('NoData         : %.17g\n', cfg.globalND);
fprintf('Zero as NoData : NO\n');
fprintf('MATLAB/Python raster writing: NOT USED\n');
fprintf('============================================================\n');

logRoot = fullfile(cfg.rootPR, 'Logs');
if exist(logRoot, 'dir') ~= 7; mkdir(logRoot); end
logCsv = fullfile(logRoot, sprintf('B005_Merge_BathyPriority_ByVRT_FIX2_TiledLowMem_%s.csv', datestr(now,'yyyymmdd_HHMMSS')));
lfid = fopen(logCsv, 'w');
fprintf(lfid, 'river,status,merged1m,note\n');

for i = 1:numel(riverList)
    river = char(riverList{i});
    try
        outVrt = processOneRiver(river, cfg);
        fprintf(lfid, '%s,OK,%s,%s\n', csvq(river), csvq(outVrt), '');
    catch ME
        fprintf('\n[ERROR] %s\n%s\n', river, getReport(ME, 'extended', 'hyperlinks', 'off'));
        fprintf(lfid, '%s,ERROR,,%s\n', csvq(river), csvq(ME.message));
        fclose(lfid);
        rethrow(ME);
    end
end

fclose(lfid);
fprintf('\n============================================================\n');
fprintf('B005 finished. Log written:\n%s\n', logCsv);
fprintf('============================================================\n');
end

function outVrt = processOneRiver(river, cfg)
fprintf('\n============================================================\n');
fprintf('[B005] River: %s\n', river);
fprintf('============================================================\n');

bathyVrt = fullfile(cfg.rootPR, cfg.bathyRoot, river, 'Bathy_1m.vrt');
demVrt = fullfile(cfg.rootPR, cfg.demRoot, river, 'DEM_3DEP_1m_ResampleandClip.vrt');

out1mRoot = fullfile(cfg.rootPR, sprintf(cfg.outRootPattern, 1), river);
if exist(out1mRoot, 'dir') ~= 7; mkdir(out1mRoot); end
outVrt = fullfile(out1mRoot, 'Combined_BathyPriority_1m.vrt');
listTxt = fullfile(out1mRoot, 'B005_merge_input_order_3DEP_then_Bathy.txt');
summaryTxt = fullfile(out1mRoot, 'B005_merge_method_summary.txt');

if exist(bathyVrt, 'file') ~= 2; error('Missing bathy input: %s', bathyVrt); end
if exist(demVrt, 'file') ~= 2; error('Missing 3DEP input. Run B004 first: %s', demVrt); end

binfo = getRasterInfoJSON(bathyVrt);
dinfo = getRasterInfoJSON(demVrt);
checkSameGrid(binfo, dinfo, river);

fprintf('Bathy input : %s\n', bathyVrt);
fprintf('3DEP input  : %s\n', demVrt);
fprintf('Output VRT  : %s\n', outVrt);
fprintf('Grid rows/cols : %d / %d\n', binfo.rows, binfo.cols);

if cfg.overwrite && exist(outVrt, 'file') == 2
    delete(outVrt);
elseif ~cfg.overwrite && exist(outVrt, 'file') == 2
    fprintf('[SKIP] merged 1m exists: %s\n', outVrt);
end

fid = fopen(listTxt, 'w');
if fid < 0; error('Cannot write list file: %s', listTxt); end
fprintf(fid, '%s\n', demVrt);   % lower priority: 3DEP
fprintf(fid, '%s\n', bathyVrt); % higher priority: Bathy
fclose(fid);

cmd = sprintf(['gdalbuildvrt -overwrite ', ...
    '-resolution user -tr %.17g %.17g ', ...
    '-te %.17g %.17g %.17g %.17g ', ...
    '-srcnodata %.17g -vrtnodata %.17g ', ...
    '-input_file_list %s %s'], ...
    binfo.xres, binfo.yres, ...
    binfo.xmin, binfo.ymin, binfo.xmax, binfo.ymax, ...
    cfg.globalND, cfg.globalND, ...
    shellq(listTxt), shellq(outVrt));
runCmd(cmd, cfg.envPrefix);

fid = fopen(summaryTxt, 'w');
fprintf(fid, 'B005 merge method summary\n');
fprintf(fid, 'River: %s\n', river);
fprintf(fid, 'Method: gdalbuildvrt mosaic VRT\n');
fprintf(fid, 'Input order 1: 3DEP lower priority: %s\n', demVrt);
fprintf(fid, 'Input order 2: Bathy higher priority: %s\n', bathyVrt);
fprintf(fid, 'Logic: valid Bathy wins; Bathy NoData (-999999) is filled by valid 3DEP; both NoData -> -999999.\n');
fprintf(fid, 'Zero is valid elevation and is not treated as NoData.\n');
fprintf(fid, 'Grid: cols=%d rows=%d xmin=%.17g ymin=%.17g xmax=%.17g ymax=%.17g xres=%.17g yres=%.17g\n', ...
    binfo.cols, binfo.rows, binfo.xmin, binfo.ymin, binfo.xmax, binfo.ymax, binfo.xres, binfo.yres);
fclose(fid);

minfo = getRasterInfoJSON(outVrt);
checkSameGrid(binfo, minfo, river);

if cfg.doUpscale && ~isempty(cfg.upscaleRes)
    for rr = cfg.upscaleRes(:)'
        upscaleMerged(outVrt, binfo, river, rr, cfg);
    end
end
end

function upscaleMerged(inVrt, srcInfo, river, res, cfg)
outRoot = fullfile(cfg.rootPR, sprintf(cfg.outRootPattern, res), river);
if exist(outRoot, 'dir') ~= 7; mkdir(outRoot); end
outTif = fullfile(outRoot, sprintf('Combined_BathyPriority_%dm.tif', res));
outVrt = fullfile(outRoot, sprintf('Combined_BathyPriority_%dm.vrt', res));
tileDir = fullfile(outRoot, sprintf('Combined_BathyPriority_%dm_tiles', res));
tileList = fullfile(outRoot, sprintf('Combined_BathyPriority_%dm_tile_list.txt', res));

outCols = ceil((srcInfo.xmax - srcInfo.xmin) / res);
outRows = ceil((srcInfo.ymax - srcInfo.ymin) / res);
outPixels = outCols * outRows;

mode = cfg.upscaleMode;
if strcmp(mode, 'auto')
    if outPixels > cfg.largeOutputPixelThreshold
        mode = 'tiled';
    else
        mode = 'single';
    end
end

fprintf('\n[B005 upscale] %s -> %dm\n', river, res);
fprintf('  target approx size: %d cols x %d rows = %.3g pixels\n', outCols, outRows, outPixels);
fprintf('  selected mode: %s\n', mode);

if cfg.overwrite
    if exist(outTif, 'file') == 2; delete(outTif); end
    if exist(outVrt, 'file') == 2; delete(outVrt); end
    if exist(tileList, 'file') == 2; delete(tileList); end
    if exist(tileDir, 'dir') == 7
        % Remove previous tiles only inside this known output directory.
        delete(fullfile(tileDir, '*.tif'));
    end
elseif exist(outVrt, 'file') == 2
    fprintf('[SKIP] upscale %dm exists: %s\n', res, outVrt);
    return;
end

if strcmp(mode, 'single')
    cmd = baseWarpCmd(inVrt, outTif, res, cfg, '');
    runCmd(cmd, cfg.envPrefix);
    cmd2 = sprintf('gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g %s %s', ...
        cfg.globalND, cfg.globalND, shellq(outVrt), shellq(outTif));
    runCmd(cmd2, cfg.envPrefix);
else
    if exist(tileDir, 'dir') ~= 7; mkdir(tileDir); end
    fid = fopen(tileList, 'w');
    if fid < 0; error('Cannot write tile list: %s', tileList); end

    nTileX = ceil(outCols / cfg.coarseTilePixels);
    nTileY = ceil(outRows / cfg.coarseTilePixels);
    fprintf('  tiled output: %d x %d tiles, tile=%d output pixels\n', nTileX, nTileY, cfg.coarseTilePixels);

    tileCount = 0;
    for ty = 1:nTileY
        row0 = (ty-1) * cfg.coarseTilePixels;
        row1 = min(ty * cfg.coarseTilePixels, outRows);
        yTop = srcInfo.ymax - row0 * res;
        yBottom = srcInfo.ymax - row1 * res;
        for tx = 1:nTileX
            col0 = (tx-1) * cfg.coarseTilePixels;
            col1 = min(tx * cfg.coarseTilePixels, outCols);
            xLeft = srcInfo.xmin + col0 * res;
            xRight = srcInfo.xmin + col1 * res;
            tileCount = tileCount + 1;
            tileTif = fullfile(tileDir, sprintf('tile_y%04d_x%04d.tif', ty, tx));
            teOpt = sprintf('-te %.17g %.17g %.17g %.17g ', xLeft, yBottom, xRight, yTop);
            cmd = baseWarpCmd(inVrt, tileTif, res, cfg, teOpt);
            fprintf('  tile %d/%d: y=%d x=%d\n', tileCount, nTileX*nTileY, ty, tx);
            runCmd(cmd, cfg.envPrefix);
            fprintf(fid, '%s\n', tileTif);
        end
    end
    fclose(fid);

    cmd2 = sprintf('gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g -input_file_list %s %s', ...
        cfg.globalND, cfg.globalND, shellq(tileList), shellq(outVrt));
    runCmd(cmd2, cfg.envPrefix);
end
end

function cmd = baseWarpCmd(inVrt, outTif, res, cfg, teOpt)
% No -multi by default. Multi-threading can increase memory and trigger OOM.
% Use --config GDAL_CACHEMAX and -wm to keep memory bounded.
if cfg.numThreads > 1
    threadOpt = sprintf('-multi -wo NUM_THREADS=%d ', cfg.numThreads);
else
    threadOpt = '';
end
cmd = sprintf(['gdalwarp -overwrite -of GTiff ', ...
    '--config GDAL_CACHEMAX %d ', ...
    '%s', ...
    '-tr %.17g %.17g -r %s ', ...
    '-srcnodata %.17g -dstnodata %.17g ', ...
    '%s', ...
    '-wm %d ', ...
    '-co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 ', ...
    '-co COMPRESS=LZW -co BIGTIFF=YES -co SPARSE_OK=YES ', ...
    '%s %s'], ...
    cfg.gdalCacheMB, ...
    teOpt, ...
    res, res, cfg.upscaleAlg, ...
    cfg.globalND, cfg.globalND, ...
    threadOpt, ...
    cfg.warpMemoryMB, ...
    shellq(inVrt), shellq(outTif));
end

function info = getRasterInfoJSON(path)
tmp = [tempname, '.json'];
cmd = sprintf('gdalinfo -json %s > %s', shellq(path), shellq(tmp));
runCmd(cmd, '');
raw = fileread(tmp);
try
    delete(tmp);
catch
end
j = jsondecode(raw);
info.cols = double(j.size(1));
info.rows = double(j.size(2));
gt = double(j.geoTransform(:));
info.gt = gt;
x0 = gt(1); px = gt(2); rx = gt(3); y0 = gt(4); ry = gt(5); py = gt(6);
xs = [x0, x0 + info.cols*px, x0 + info.rows*rx, x0 + info.cols*px + info.rows*rx];
ys = [y0, y0 + info.cols*ry, y0 + info.rows*py, y0 + info.cols*ry + info.rows*py];
info.xmin = min(xs); info.xmax = max(xs);
info.ymin = min(ys); info.ymax = max(ys);
info.xres = abs(px);
info.yres = abs(py);
if isfield(j, 'coordinateSystem') && isfield(j.coordinateSystem, 'wkt')
    info.wkt = j.coordinateSystem.wkt;
else
    info.wkt = '';
end
end

function checkSameGrid(a, b, river)
tol = max([a.xres, a.yres, b.xres, b.yres, 1]) * 1e-7;
if a.cols ~= b.cols || a.rows ~= b.rows
    error('Grid size mismatch for %s: A=%d/%d, B=%d/%d', river, a.rows, a.cols, b.rows, b.cols);
end
if any(abs(a.gt(:) - b.gt(:)) > tol)
    error('GeoTransform mismatch for %s. Max abs diff = %.17g', river, max(abs(a.gt(:)-b.gt(:))));
end
if abs(a.xmin-b.xmin)>tol || abs(a.xmax-b.xmax)>tol || abs(a.ymin-b.ymin)>tol || abs(a.ymax-b.ymax)>tol
    error('Extent mismatch for %s.', river);
end
if ~isempty(a.wkt) && ~isempty(b.wkt) && ~strcmp(a.wkt, b.wkt)
    warning('Projection WKT differs for %s, but grid transform matches. Inspect if needed.', river);
end
end

function runCmd(cmd, envPrefix)
if ~isempty(envPrefix)
    fullcmd = sprintf('%s %s', envPrefix, cmd);
else
    fullcmd = cmd;
end
fprintf('\nCMD:\n%s\n', fullcmd);
[status, out] = system(fullcmd);
if status ~= 0
    error('Command failed with status %d:\n%s\nOutput:\n%s', status, fullcmd, out);
end
if ~isempty(strtrim(out))
    fprintf('%s\n', out);
end
end

function s = shellq(x)
x = char(x);
s = ['''', strrep(x, '''', '''"''"'''), ''''];
end

function s = csvq(x)
x = char(x);
s = ['"', strrep(x, '"', '""'), '"'];
end

function rivers = selectedRiverList()
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
    'WA_Nisqually_Bathymetric_2020'};
end
