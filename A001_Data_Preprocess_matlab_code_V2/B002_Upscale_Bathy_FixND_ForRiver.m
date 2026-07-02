function Summary = B002_Upscale_Bathy_FixND_ForRiver(riverName, targetResM, varargin)
% B002_Upscale_Bathy_FixND_ForRiver
% 2026-07-01
%
% Purpose
%   Per-river version of B002. Upscale bathymetry-only FixND products from
%   1 m to 3 m / 5 m / 10 m. This function is designed so that different
%   rivers can be run in separate MATLAB sessions.
%
% Input
%   Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt
%
% Output
%   Processed_Results/Bathy_3m_FixND/<river>/Bathy_3m.tif + Bathy_3m.vrt
%   Processed_Results/Bathy_5m_FixND/<river>/Bathy_5m.tif + Bathy_5m.vrt
%   Processed_Results/Bathy_10m_FixND/<river>/Bathy_10m.tif + Bathy_10m.vrt
%
% Usage examples
%   B002_Upscale_Bathy_FixND_ForRiver('MD_PotomacRiver_Bathy_2019');
%   B002_Upscale_Bathy_FixND_ForRiver('CA_KlamathRiver_TopoBathy_2018_D18', [3 5 10]);
%   B002_Upscale_Bathy_FixND_ForRiver('OR_MKRC_Topobathy_2021', [3 5 10], 'numThreads', 4);
%   B002_Upscale_Bathy_FixND_ForRiver({'MD_PotomacRiver_Bathy_2019','CA_KlamathRiver_TopoBathy_2018_D18'});
%   B002_Upscale_Bathy_FixND_ForRiver('ALL');
%   B002_Upscale_Bathy_FixND_ForRiver('LIST');
%
% Notes
%   - This step upscales bathymetry only, before 3DEP fusion.
%   - NoData is explicitly kept as -999999.
%   - Zero is NOT treated as NoData.
%   - OR_MKRC is special: horizontal CRS unit is foot. Target 3/5/10 m is
%     converted to 9.842519685 / 16.404199475 / 32.808398950 ft.
%   - OR_MKRC uses Bathy_2ft_elev_m.vrt as the source when available, so
%     upscaling is directly from the corrected 2 ft source rather than from
%     the already-resampled 1 m VRT.
%
% Parallel safety
%   - It is safe to run different rivers in different MATLAB sessions.
%   - Do NOT run the same river/resolution in two MATLAB sessions at the same time.
%   - When running multiple sessions, avoid numThreads='ALL_CPUS' in every session.
%     A value like 4 or 8 is safer on shared nodes.

if nargin < 1 || isempty(riverName)
    riverName = 'LIST';
end
if nargin < 2 || isempty(targetResM)
    targetResM = [3 5 10];
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
    error('riverName must be a char/string, a cell array of river names, ''ALL'', or ''LIST''.');
end

% Validate resolutions
if ~isnumeric(targetResM) || isempty(targetResM)
    error('targetResM must be a numeric vector, e.g. [3 5 10].');
end
targetResM = targetResM(:)';

% Validate rivers
for i = 1:numel(riversToRun)
    if ~ismember(riversToRun{i}, validRivers)
        fprintf('\nUnknown river: %s\n', riversToRun{i});
        printRiverList(validRivers);
        error('River is not in selected valid river list.');
    end
end

fprintf('\n============================================================\n');
fprintf('B002 per-river bathymetry-only upscaling\n');
fprintf('Input root : %s\n', cfg.bathy1mFixRoot);
fprintf('Output root: %s/Bathy_<res>m_FixND\n', cfg.prRoot);
fprintf('Rivers     : %s\n', strjoin(riversToRun, ', '));
fprintf('Resolutions: %s m\n', mat2str(targetResM));
fprintf('Overwrite  : %d\n', cfg.overwrite);
fprintf('GDAL thread: %s\n', cfg.numThreads);
fprintf('============================================================\n');

summaryRows = {};
rowId = 0;

for r = 1:numel(riversToRun)
    river = riversToRun{r};

    for j = 1:numel(targetResM)
        resM = targetResM(j);

        [srcRaster, targetResX, targetResY, note] = getUpscaleSourceAndTargetRes(cfg, river, resM);

        outRoot = fullfile(cfg.prRoot, sprintf('Bathy_%dm_FixND', resM));
        outSub  = fullfile(outRoot, river);
        ensureDir(outSub);

        outTif = fullfile(outSub, sprintf('Bathy_%dm.tif', resM));
        outVrt = fullfile(outSub, sprintf('Bathy_%dm.vrt', resM));

        fprintf('\n============================================================\n');
        fprintf('[River %d/%d] %s -> Bathy_%dm_FixND\n', r, numel(riversToRun), river, resM);
        fprintf('Source : %s\n', srcRaster);
        fprintf('Output : %s\n', outTif);
        fprintf('Target : %.12g x %.12g source CRS units\n', targetResX, targetResY);
        fprintf('Note   : %s\n', note);
        fprintf('============================================================\n');

        if exist(srcRaster, 'file') ~= 2
            msg = sprintf('Missing source raster for %s: %s', river, srcRaster);
            if cfg.stopOnMissing
                error(msg);
            else
                warning(msg);
                rowId = rowId + 1;
                summaryRows(rowId,:) = {river, resM, srcRaster, outTif, outVrt, 'MISSING_SOURCE', note}; %#ok<AGROW>
                continue;
            end
        end

        status = upscaleOneRaster(srcRaster, outTif, outVrt, targetResX, targetResY, cfg);

        if status ~= 0
            rowId = rowId + 1;
            summaryRows(rowId,:) = {river, resM, srcRaster, outTif, outVrt, 'FAILED', note}; %#ok<AGROW>
            error('Upscaling failed for river=%s, resolution=%dm.', river, resM);
        end

        printRasterInfo(outVrt);

        rowId = rowId + 1;
        summaryRows(rowId,:) = {river, resM, srcRaster, outTif, outVrt, 'OK', note}; %#ok<AGROW>
    end
end

Summary = rowsToTable(summaryRows);

summaryFile = fullfile(cfg.logRoot, sprintf('B002_Upscale_Bathy_FixND_%s_%s.csv', safeName(strjoin(riversToRun, '__')), datestr(now, 'yyyymmdd_HHMMSS')));
writeSummaryCSV(summaryFile, summaryRows);

fprintf('\n============================================================\n');
fprintf('B002 per-river run finished. Summary written:\n%s\n', summaryFile);
fprintf('============================================================\n');

end

%% ============================================================
%  Local functions
% ============================================================

function cfg = defaultConfig()
    cfg.prRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
    cfg.bathy1mFixRoot = fullfile(cfg.prRoot, 'Bathy_1m_FixND');
    cfg.logRoot = fullfile(cfg.prRoot, 'Logs');

    cfg.globalND = -999999;
    cfg.meterToFoot = 1 / 0.3048;
    cfg.resampleAlg = 'average';
    cfg.overwrite = true;
    cfg.stopOnMissing = true;

    % Safer default for multiple MATLAB sessions. Use 'ALL_CPUS' only when
    % running a single river/session on an exclusive node.
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
            case 'overwrite'
                cfg.overwrite = logical(val);
            case 'numthreads'
                if isnumeric(val)
                    cfg.numThreads = num2str(val);
                else
                    cfg.numThreads = char(val);
                end
            case 'warpmemorymb'
                cfg.warpMemoryMB = val;
            case 'resamplealg'
                cfg.resampleAlg = char(val);
            case 'stoponmissing'
                cfg.stopOnMissing = logical(val);
            otherwise
                error('Unknown option: %s', key);
        end
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
    fprintf('\nSelected valid rivers for B002:\n');
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

    try
        addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
    catch
    end
end

function ensureDir(d)
    if exist(d, 'dir') ~= 7
        mkdir(d);
    end
end

function [srcRaster, targetResX, targetResY, note] = getUpscaleSourceAndTargetRes(cfg, river, resM)
    srcRaster = fullfile(cfg.bathy1mFixRoot, river, 'Bathy_1m.vrt');
    targetResX = resM;
    targetResY = resM;
    note = 'standard_meter_crs__source_Bathy_1m_FixND';

    if strcmp(river, 'OR_MKRC_Topobathy_2021')
        orSource2ft = fullfile(cfg.bathy1mFixRoot, river, 'Bathy_2ft_elev_m.vrt');
        if exist(orSource2ft, 'file') == 2
            srcRaster = orSource2ft;
            note = 'OR_MKRC_special__source_Bathy_2ft_elev_m__target_resolution_converted_m_to_ft';
        else
            srcRaster = fullfile(cfg.bathy1mFixRoot, river, 'Bathy_1m.vrt');
            note = 'OR_MKRC_special__fallback_source_Bathy_1m_vrt__target_resolution_converted_m_to_ft';
        end
        targetResX = resM * cfg.meterToFoot;
        targetResY = resM * cfg.meterToFoot;
    end
end

function status = upscaleOneRaster(srcRaster, outTif, outVrt, targetResX, targetResY, cfg)
    if exist(outTif, 'file') == 2
        if cfg.overwrite
            delete(outTif);
        else
            fprintf('[SKIP] Existing output tif: %s\n', outTif);
            status = 0;
            return;
        end
    end

    if exist(outVrt, 'file') == 2
        if cfg.overwrite
            delete(outVrt);
        else
            fprintf('[SKIP] Existing output vrt: %s\n', outVrt);
            status = 0;
            return;
        end
    end

    cmd = sprintf([ ...
        'export PROJ_USE_PROJ4_INIT_RULES=YES; ', ...
        'export PROJ_NETWORK=ON; ', ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-tr %.12g %.12g ', ...
        '-r %s ', ...
        '-ot Float32 ', ...
        '-srcnodata %.17g -dstnodata %.17g ', ...
        '-multi -wo NUM_THREADS=%s -wm %g ', ...
        '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
        '"%s" "%s"' ], ...
        targetResX, targetResY, cfg.resampleAlg, ...
        cfg.globalND, cfg.globalND, cfg.numThreads, cfg.warpMemoryMB, srcRaster, outTif);

    fprintf('CMD:\n%s\n', cmd);
    status = system(cmd);
    if status ~= 0
        return;
    end

    cmd2 = sprintf([ ...
        'gdalbuildvrt -overwrite ', ...
        '-srcnodata %.17g -vrtnodata %.17g ', ...
        '"%s" "%s"' ], ...
        cfg.globalND, cfg.globalND, outVrt, outTif);

    fprintf('Build VRT:\n%s\n', cmd2);
    status = system(cmd2);
end

function printRasterInfo(rasterPath)
    if exist(rasterPath, 'file') ~= 2
        fprintf('[WARN] Cannot print raster info; missing: %s\n', rasterPath);
        return;
    end
    cmd = sprintf('gdalinfo "%s" | grep -E "Size is|Pixel Size|Upper Left|Lower Right|NoData"', rasterPath);
    system(cmd);
end

function T = rowsToTable(rows)
    if isempty(rows)
        T = table();
        return;
    end
    T = cell2table(rows, 'VariableNames', {'river','res_m','srcRaster','outTif','outVrt','status','note'});
end

function writeSummaryCSV(summaryFile, rows)
    fid = fopen(summaryFile, 'w');
    if fid < 0
        warning('Cannot write summary CSV: %s', summaryFile);
        return;
    end

    fprintf(fid, 'river,res_m,srcRaster,outTif,outVrt,status,note\n');
    for i = 1:size(rows,1)
        fprintf(fid, '"%s",%g,"%s","%s","%s","%s","%s"\n', ...
            rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, rows{i,7});
    end
    fclose(fid);
end

function s = safeName(s)
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if numel(s) > 120
        s = s(1:120);
    end
end
