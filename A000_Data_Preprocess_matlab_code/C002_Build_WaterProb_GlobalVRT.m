function C002_Build_WaterProb_GlobalVRT(varargin)
% C002_Build_WaterProb_GlobalVRT
%
% Build a single VRT mosaic for CONUS Water Probability tiles.
%
% This is the second step of the BetterMask C-series pipeline.
% C001 clipped US_Detailed RiverOnly SHP by river bathy extent.
% C002 only builds the Water_Prob global/CONUS VRT. It does NOT clip or
% resample to any river grid. River-grid alignment will be done in C003.
%
% Default input:
%   /tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask/Water_Prob
%
% Default output:
%   /tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask/Water_Prob_VRT/Water_Prob_CONUS.vrt
%
% Example:
%   C002_Build_WaterProb_GlobalVRT()
%
% Optional:
%   C002_Build_WaterProb_GlobalVRT('overwrite', true, 'srcNoData', 255, 'vrtNoData', 255)
%
% Notes:
%   Water probability / occurrence tiles are usually Byte rasters.
%   Common convention:
%       0      = non-water / occurrence 0
%       1-100  = water occurrence / probability
%       255    = NoData / outside valid domain
%   Therefore C002 sets 255 as VRT NoData by default.

p = inputParser;
addParameter(p, 'betterMaskRoot', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask', @ischar);
addParameter(p, 'waterProbDir', '', @ischar);
addParameter(p, 'outDir', '', @ischar);
addParameter(p, 'outVRTName', 'Water_Prob_CONUS.vrt', @ischar);
addParameter(p, 'overwrite', true, @(x)islogical(x) || isnumeric(x));
addParameter(p, 'srcNoData', 255, @isnumeric);
addParameter(p, 'vrtNoData', 255, @isnumeric);
addParameter(p, 'runGdalInfo', true, @(x)islogical(x) || isnumeric(x));
parse(p, varargin{:});
opt = p.Results;
opt.overwrite = logical(opt.overwrite);
opt.runGdalInfo = logical(opt.runGdalInfo);

if isempty(opt.waterProbDir)
    opt.waterProbDir = fullfile(opt.betterMaskRoot, 'Water_Prob');
end
if isempty(opt.outDir)
    opt.outDir = fullfile(opt.betterMaskRoot, 'Water_Prob_VRT');
end

if ~exist(opt.waterProbDir, 'dir')
    error('Water_Prob directory does not exist: %s', opt.waterProbDir);
end
if ~exist(opt.outDir, 'dir')
    mkdir(opt.outDir);
end

outVRT   = fullfile(opt.outDir, opt.outVRTName);
tileList = fullfile(opt.outDir, 'Water_Prob_Tile_List.txt');
logFile  = fullfile(opt.outDir, 'C002_Build_WaterProb_GlobalVRT_Log.txt');

if exist(outVRT, 'file') && ~opt.overwrite
    fprintf('Output VRT already exists and overwrite=false:\n%s\n', outVRT);
    return;
end

% -------------------------------------------------------------------------
% 1. Collect all tif/tiff tiles recursively.
% -------------------------------------------------------------------------
files = collect_tiff_files(opt.waterProbDir);
if isempty(files)
    error('No .tif/.tiff files found under: %s', opt.waterProbDir);
end

fid = fopen(tileList, 'w');
if fid < 0
    error('Cannot write tile list: %s', tileList);
end
for i = 1:numel(files)
    fprintf(fid, '%s\n', files{i});
end
fclose(fid);

% -------------------------------------------------------------------------
% 2. Build VRT. 255 is explicitly treated as NoData by default.
% -------------------------------------------------------------------------
overwriteFlag = '';
if opt.overwrite
    overwriteFlag = '-overwrite ';
end

cmd = sprintf([ ...
    'gdalbuildvrt %s', ...
    '-srcnodata %g -vrtnodata %g ', ...
    '-input_file_list %s %s'], ...
    overwriteFlag, ...
    opt.srcNoData, opt.vrtNoData, ...
    qpath(tileList), qpath(outVRT));

fprintf('\n============================================================\n');
fprintf('C002 build Water Probability global VRT\n');
fprintf('Water_Prob dir : %s\n', opt.waterProbDir);
fprintf('Output dir     : %s\n', opt.outDir);
fprintf('Output VRT     : %s\n', outVRT);
fprintf('Tile list      : %s\n', tileList);
fprintf('Number of tiles: %d\n', numel(files));
fprintf('srcNoData      : %g\n', opt.srcNoData);
fprintf('vrtNoData      : %g\n', opt.vrtNoData);
fprintf('============================================================\n\n');

fprintf('Running command:\n%s\n\n', cmd);
[status, msg] = system(cmd);

fid = fopen(logFile, 'w');
if fid > 0
    fprintf(fid, 'C002_Build_WaterProb_GlobalVRT\n');
    fprintf(fid, 'Water_Prob dir : %s\n', opt.waterProbDir);
    fprintf(fid, 'Output VRT     : %s\n', outVRT);
    fprintf(fid, 'Tile list      : %s\n', tileList);
    fprintf(fid, 'Number of tiles: %d\n', numel(files));
    fprintf(fid, 'Command:\n%s\n\n', cmd);
    fprintf(fid, 'Status: %d\n', status);
    fprintf(fid, 'Message:\n%s\n', msg);
    fclose(fid);
end

if status ~= 0
    fprintf('%s\n', msg);
    error('gdalbuildvrt failed. See log: %s', logFile);
end

if ~exist(outVRT, 'file')
    error('gdalbuildvrt finished but output VRT was not found: %s', outVRT);
end

fprintf('C002 done. Water Probability VRT created:\n%s\n', outVRT);
fprintf('Log written to:\n%s\n', logFile);

% -------------------------------------------------------------------------
% 3. Quick audit. Do not fail the pipeline if gdalinfo stats are slow/fail.
% -------------------------------------------------------------------------
if opt.runGdalInfo
    fprintf('\nQuick audit with gdalinfo. Key items to check:\n');
    fprintf('  Type should usually be Byte.\n');
    fprintf('  NoData should be 255.\n');
    fprintf('  Values should usually be 0-100 plus NoData=255.\n\n');

    cmdInfo = sprintf('gdalinfo -mm %s | head -120', qpath(outVRT));
    fprintf('Running command:\n%s\n\n', cmdInfo);
    system(cmdInfo);
end

end

% =========================================================================
% Helper functions
% =========================================================================
function files = collect_tiff_files(rootDir)
% Collect tif/tiff files recursively. This avoids relying only on MATLAB's
% dir('**/*.tif') behavior on older versions.

cmdFind = sprintf('find %s -type f \\( -iname "*.tif" -o -iname "*.tiff" \\) | sort', qpath(rootDir));
[status, out] = system(cmdFind);
if status == 0
    C = regexp(strtrim(out), '\n', 'split');
    if numel(C) == 1 && isempty(C{1})
        files = {};
    else
        files = C(:);
    end
    return;
end

% MATLAB fallback.
d1 = dir(fullfile(rootDir, '**', '*.tif'));
d2 = dir(fullfile(rootDir, '**', '*.tiff'));
d = [d1; d2];
files = cell(numel(d), 1);
for k = 1:numel(d)
    files{k} = fullfile(d(k).folder, d(k).name);
end
files = sort(files);
end

function s = qpath(x)
% Quote a file path for shell command.
s = ['"', x, '"'];
end
