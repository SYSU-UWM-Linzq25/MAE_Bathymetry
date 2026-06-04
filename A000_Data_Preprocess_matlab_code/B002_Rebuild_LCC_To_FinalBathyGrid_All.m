function B002_Rebuild_LCC_To_FinalBathyGrid_All(varargin)
%% ============================================================
%  B002_Rebuild_LCC_To_FinalBathyGrid_All.m
%
%  Purpose:
%    Rebuild final LCC_1m / LCC_3m / LCC_5m / LCC_10m products
%    directly from LCC_Origin/LCC_mix.vrt to the final Bathy_*m_FixND grid.
%
%  Why:
%    The old B002 used:
%       raw Bathy.vrt + xres1*res + -tr + -tap
%    which can create 1-row/1-col or even 2x size mismatch.
%
%  New logic:
%    For each river and each resolution:
%      target grid = Processed_Results/Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%
%    Then:
%      1) gdalwarp LCC_mix.vrt to exact target grid using -te + -ts
%      2) tile-read the warped raw LCC class raster
%      3) convert:
%           class 80       -> 1
%           valid non-80   -> 0
%           LCC NoData     -> lccND
%      4) write tiled GeoTIFFs
%      5) build final VRT
%
%  Important:
%    - LCC NoData is preserved as lccND = -99999.
%    - 0 is a valid LCC value meaning non-water / non-river.
%    - Do NOT use -srcnodata 0 or -dstnodata 0.
%    - Do NOT use -tr or -tap.
%    - Use -te + -ts from final bathy grid.
%
%  Output:
%    LCC_<res>m/<river>/ESA_WorldCover_Resampleandclip_<res>m.vrt
%    LCC_<res>m/<river>/_tiles/tile_rXXXXXX_cXXXXXX.tif
%
%  Example:
%    B002_Rebuild_LCC_To_FinalBathyGrid_All('targetRes',[1])
%
%    B002_Rebuild_LCC_To_FinalBathyGrid_All( ...
%       'selectedRivers', {'MD_PotomacRiver_Bathy_2019'}, ...
%       'targetRes', [1 3 5 10])
%
%    B002_Rebuild_LCC_To_FinalBathyGrid_All( ...
%       'selectedRivers', {'KewaFix2Null'}, ...
%       'targetRes', [1], ...
%       'overwrite', true)
% ============================================================

%% -------------------- Parse inputs --------------------
p = inputParser;

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'LCC_vrt', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_mix.vrt', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'targetRes', [1 3 5 10], @isnumeric);

% selectedRivers = {} means process all rivers under Bathy_1m_FixND
addParameter(p, 'selectedRivers', {}, @(x) iscell(x) || isstring(x));

addParameter(p, 'lccClassWater', 80, @isnumeric);

% LCC final NoData
addParameter(p, 'lccND', -99999, @isnumeric);

% Bathy NoData only used to read target info; not used in LCC classification.
addParameter(p, 'tile', 2048, @isnumeric);

addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'backupOld', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, varargin{:});

rootPR = char(p.Results.rootPR);
LCC_vrt = char(p.Results.LCC_vrt);
targetRes = p.Results.targetRes;
selectedRivers = p.Results.selectedRivers;

lccClassWater = p.Results.lccClassWater;
lccND = p.Results.lccND;
tile = p.Results.tile;

overwrite = p.Results.overwrite;
backupOld = p.Results.backupOld;
doPathSetup = p.Results.doPathSetup;

%% -------------------- Path setup --------------------
if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

if exist(LCC_vrt, 'file') ~= 2
    error('Missing LCC source VRT: %s', LCC_vrt);
end

[~, ~, ~, ~, ~, ~, rawLccND] = RasterInfo(LCC_vrt);

if isempty(rawLccND) || ~isfinite(rawLccND)
    % ESA WorldCover normally does not use class 0 as land class.
    % If metadata has no NoData, we still treat raw <= 0 as invalid later.
    rawLccND = NaN;
end

%% -------------------- River list --------------------
if isempty(selectedRivers)
    d = dir(fullfile(rootPR, 'Bathy_1m_FixND'));
    d = d([d.isdir]);
    d(1:2) = [];

    rivers = cell(numel(d), 1);
    for i = 1:numel(d)
        rivers{i} = d(i).name;
    end
else
    rivers = cellstr(selectedRivers);
end

%% -------------------- Logs --------------------
logDir = fullfile(rootPR, 'Z011_Rebuild_LCC_To_FinalBathyGrid_Log');
if exist(logDir, 'dir') ~= 7
    mkdir(logDir);
end

logCSV = fullfile(logDir, 'B002_Rebuild_LCC_To_FinalBathyGrid_Log.csv');
fid = fopen(logCSV, 'w');
fprintf(fid, ['River,Resolution_m,Status,Rows,Cols,' ...
              'N_total,N_water,N_nonwater,N_nodata,' ...
              'SourceLCC,TargetBathy,OutputVRT,Message\n']);
fclose(fid);

fprintf('\n============================================================\n');
fprintf('B002 rebuild LCC to final bathy grid\n');
fprintf('Source LCC: %s\n', LCC_vrt);
fprintf('Target resolutions: %s\n', mat2str(targetRes));
fprintf('Number of rivers: %d\n', numel(rivers));
fprintf('LCC water class = %g\n', lccClassWater);
fprintf('Final LCC NoData = %g\n', lccND);
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

    for j = 1:numel(targetRes)

        res = targetRes(j);

        %% -------------------- Target bathy grid --------------------
        bathy_vrt = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
            river, sprintf('Bathy_%dm.vrt', res));

        if exist(bathy_vrt, 'file') ~= 2
            % Some folders may only have .tif.
            bathy_tif = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
                river, sprintf('Bathy_%dm.tif', res));

            if exist(bathy_tif, 'file') == 2
                bathy_vrt = bathy_tif;
            else
                msg = sprintf('Missing target bathy grid: %s', bathy_vrt);
                warning('[%s %dm] %s', river, res, msg);
                appendLog(logCSV, river, res, 'SKIP_MISSING_BATHY', ...
                    NaN, NaN, NaN, NaN, NaN, NaN, LCC_vrt, bathy_vrt, '', msg);
                continue;
            end
        end

        [~, rowsB, colsB, geoB, projB, dataTypeBathy, ~] = RasterInfo(bathy_vrt);

        xmin = geoB(1);
        xres = geoB(2);
        ymax = geoB(4);
        yres = geoB(6);

        xmax = xmin + colsB * xres;
        ymin = ymax + rowsB * yres;

        proj_arg = sprintf('''%s''', projB);

        %% -------------------- Output paths --------------------
        outRoot = fullfile(rootPR, sprintf('LCC_%dm', res));
        outSub  = fullfile(outRoot, river);

        if exist(outSub, 'dir') ~= 7
            mkdir(outSub);
        end

        tilesDir = fullfile(outSub, '_tiles');
        if exist(tilesDir, 'dir') ~= 7
            mkdir(tilesDir);
        end

        outVrt  = fullfile(outSub, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));
        listTxt = fullfile(outSub, 'tile_list.txt');

        rawWarpDir = fullfile(outSub, '_tmp_raw_LCC_to_bathy_grid');
        if exist(rawWarpDir, 'dir') ~= 7
            mkdir(rawWarpDir);
        end

        rawWarpVRT = fullfile(rawWarpDir, sprintf('Raw_LCC_to_BathyGrid_%dm.vrt', res));

        if exist(outVrt, 'file') == 2 && ~overwrite
            msg = 'Output exists and overwrite=false';
            fprintf('[%s %dm] Skip: %s\n', river, res, msg);
            appendLog(logCSV, river, res, 'SKIP_EXISTS', ...
                rowsB, colsB, NaN, NaN, NaN, NaN, LCC_vrt, bathy_vrt, outVrt, msg);
            continue;
        end

        %% -------------------- Backup old VRT --------------------
        if backupOld && exist(outVrt, 'file') == 2
            backupDir = fullfile(rootPR, 'Z012_Backup_LCC_Before_Rebuild', ...
                sprintf('LCC_%dm', res), river);

            if exist(backupDir, 'dir') ~= 7
                mkdir(backupDir);
            end

            stamp = datestr(now, 'yyyymmdd_HHMMSS');
            backupFile = fullfile(backupDir, ...
                sprintf('ESA_WorldCover_Resampleandclip_%dm_%s.vrt', res, stamp));

            copyfile(outVrt, backupFile);
            fprintf('Backup old LCC VRT:\n%s\n', backupFile);
        end

        %% -------------------- Clean old products --------------------
        if exist(outVrt, 'file') == 2
            delete(outVrt);
        end
        if exist(listTxt, 'file') == 2
            delete(listTxt);
        end
        system(sprintf('rm -f "%s"/tile_*.tif', tilesDir));

        if exist(rawWarpVRT, 'file') == 2
            delete(rawWarpVRT);
        end

        %% -------------------- Warp raw LCC to exact bathy grid --------------------
        % Use Float32 so lccND=-99999 can be represented.
        %
        % Important:
        %   - Use -te + -ts from final bathy grid.
        %   - Do not use -tr.
        %   - Do not use -tap.
        %   - Do not use -srcnodata 0 or -dstnodata 0.
        %
        % If LCC source has valid NoData metadata, use it.
        % Otherwise, only set dstnodata and classify raw <= 0 as invalid later.
        if isfinite(rawLccND)
            nd_arg = sprintf('-srcnodata %.12g -dstnodata %.12g ', rawLccND, lccND);
        else
            nd_arg = sprintf('-dstnodata %.12g ', lccND);
        end

        cmdWarp = sprintf([ ...
            'gdalwarp -overwrite -of VRT ', ...
            '-ot Float32 ', ...
            '-r near ', ...
            '-multi -wo NUM_THREADS=ALL_CPUS -wm 2048 ', ...
            '-t_srs %s -te_srs %s ', ...
            '-te %.10f %.10f %.10f %.10f ', ...
            '-ts %d %d ', ...
            '%s', ...
            '-wo INIT_DEST=NO_DATA -wo SKIP_NOSOURCE=YES ', ...
            '"%s" "%s"' ], ...
            proj_arg, proj_arg, ...
            xmin, ymin, xmax, ymax, ...
            colsB, rowsB, ...
            nd_arg, ...
            LCC_vrt, rawWarpVRT);

        fprintf('\n[%s %dm] Warp raw LCC to final bathy grid\n', river, res);
        fprintf('Target bathy: %s\n', bathy_vrt);
        fprintf('Output VRT  : %s\n', outVrt);
        fprintf('Rows/Cols   : %d / %d\n', rowsB, colsB);
        fprintf('%s\n', cmdWarp);

        statusWarp = system(cmdWarp);

        if statusWarp ~= 0
            msg = 'gdalwarp raw LCC to bathy grid failed';
            warning('[%s %dm] %s', river, res, msg);
            appendLog(logCSV, river, res, 'FAIL_WARP', ...
                rowsB, colsB, NaN, NaN, NaN, NaN, LCC_vrt, bathy_vrt, outVrt, msg);
            continue;
        end

        %% -------------------- Check raw warped grid --------------------
        [~, rowsR, colsR, geoR, ~, ~, ~] = RasterInfo(rawWarpVRT);

        if rowsR ~= rowsB || colsR ~= colsB
            msg = sprintf('Raw warped LCC size mismatch: raw=%d/%d, bathy=%d/%d', ...
                rowsR, colsR, rowsB, colsB);
            warning('[%s %dm] %s', river, res, msg);
            appendLog(logCSV, river, res, 'FAIL_RAW_SIZE', ...
                rowsB, colsB, NaN, NaN, NaN, NaN, LCC_vrt, bathy_vrt, outVrt, msg);
            continue;
        end

        maxGeoDiffRaw = max(abs(geoB(:) - geoR(:)));

        if maxGeoDiffRaw > 1e-8
            msg = sprintf('Raw warped LCC geotransform mismatch: %.12g', maxGeoDiffRaw);
            warning('[%s %dm] %s', river, res, msg);
        end

        %% -------------------- Convert raw classes to 0/1/NoData tile outputs --------------------
        totalTiles = ceil(rowsB / tile) * ceil(colsB / tile);
        tileCount = 0;

        N_total = rowsB * colsB;
        N_water = 0;
        N_nonwater = 0;
        N_nodata = 0;

        for rLocal = 1:tile:rowsB

            rr = min(tile, rowsB - rLocal + 1);

            for cLocal = 1:tile:colsB

                cc = min(tile, colsB - cLocal + 1);

                rawBlock = double(ReadRaster(rawWarpVRT, rLocal, cLocal, rr, cc));

                invalid = isnan(rawBlock) | ~isfinite(rawBlock) | ...
                          (rawBlock == lccND) | (rawBlock < -1e20);

                if isfinite(rawLccND)
                    invalid = invalid | (rawBlock == rawLccND);
                end

                % ESA WorldCover valid classes are positive class codes.
                % If 0 appears, treat it as NoData/unknown rather than non-water.
                invalid = invalid | (rawBlock <= 0);

                water = (~invalid) & (round(rawBlock) == lccClassWater);
                nonwater = (~invalid) & ~water;

                out = zeros(rr, cc);
                out(water) = 1;
                out(nonwater) = 0;
                out(invalid) = lccND;

                N_water = N_water + nnz(water);
                N_nonwater = N_nonwater + nnz(nonwater);
                N_nodata = N_nodata + nnz(invalid);

                subgeoTrans = subTranscoef(geoB, rLocal, cLocal);
                tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));

                % Use bathy datatype, usually Float32/Float64, so lccND=-99999 is safe.
                WriteRaster(tileTif, out, subgeoTrans, projB, dataTypeBathy, ...
                    'GTiff', lccND);

                tileCount = tileCount + 1;
                fprintf('\r  Convert class to LCC mask: %6.2f%% (%d/%d)', ...
                    100 * tileCount / totalTiles, tileCount, totalTiles);

                clear rawBlock invalid water nonwater out
            end
        end

        fprintf('\nClass conversion done.\n');

        %% -------------------- Build final VRT --------------------
        cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
            tilesDir, listTxt);

        statusList = system(cmdList);
        if statusList ~= 0
            error('Failed to build tile list: %s', cmdList);
        end

        info = dir(listTxt);
        if isempty(info) || info.bytes == 0
            msg = 'tile_list.txt is empty';
            warning('[%s %dm] %s', river, res, msg);
            appendLog(logCSV, river, res, 'FAIL_EMPTY_TILELIST', ...
                rowsB, colsB, N_total, N_water, N_nonwater, N_nodata, ...
                LCC_vrt, bathy_vrt, outVrt, msg);
            continue;
        end

        cmdVRT = sprintf('gdalbuildvrt -overwrite -vrtnodata %.12g -input_file_list "%s" "%s"', ...
            lccND, listTxt, outVrt);

        statusVRT = system(cmdVRT);

        if statusVRT ~= 0
            msg = 'gdalbuildvrt failed';
            warning('[%s %dm] %s', river, res, msg);
            appendLog(logCSV, river, res, 'FAIL_BUILD_VRT', ...
                rowsB, colsB, N_total, N_water, N_nonwater, N_nodata, ...
                LCC_vrt, bathy_vrt, outVrt, msg);
            continue;
        end

        %% -------------------- Final verification --------------------
        [~, rowsO, colsO, geoO, ~, ~, ndO] = RasterInfo(outVrt);

        sameSize = rowsO == rowsB && colsO == colsB;
        maxGeoDiff = max(abs(geoB(:) - geoO(:)));
        sameGeo = maxGeoDiff < 1e-8;

        fprintf('\n[%s %dm] Summary\n', river, res);
        fprintf('  Final bathy rows/cols = %d / %d\n', rowsB, colsB);
        fprintf('  Final LCC   rows/cols = %d / %d\n', rowsO, colsO);
        fprintf('  sameSize = %d\n', sameSize);
        fprintf('  maxGeoDiff = %.12g\n', maxGeoDiff);
        fprintf('  sameGeo = %d\n', sameGeo);
        fprintf('  LCC NoData = %.12g\n', ndO);
        fprintf('  N_total    = %d\n', N_total);
        fprintf('  N_water    = %d\n', N_water);
        fprintf('  N_nonwater = %d\n', N_nonwater);
        fprintf('  N_nodata   = %d\n', N_nodata);
        fprintf('  Output VRT = %s\n', outVrt);

        if sameSize && sameGeo
            statusStr = 'PASS';
            msg = 'OK';
        else
            statusStr = 'FAIL_GRID';
            msg = sprintf('sameSize=%d, sameGeo=%d, maxGeoDiff=%.12g', ...
                sameSize, sameGeo, maxGeoDiff);
        end

        appendLog(logCSV, river, res, statusStr, ...
            rowsB, colsB, N_total, N_water, N_nonwater, N_nodata, ...
            LCC_vrt, bathy_vrt, outVrt, msg);

        clear rawWarpVRT rowsB colsB geoB projB dataTypeBathy
    end
end

fprintf('\n============================================================\n');
fprintf('B002 rebuild LCC done.\n');
fprintf('Log written to:\n%s\n', logCSV);
fprintf('============================================================\n');

end

%% ============================================================
%  Local helper: append log row
% ============================================================
function appendLog(logCSV, river, res, statusStr, rows, cols, ...
                   N_total, N_water, N_nonwater, N_nodata, ...
                   srcLCC, targetBathy, outVRT, msg)

    fid = fopen(logCSV, 'a');

    if fid < 0
        warning('Cannot open log CSV: %s', logCSV);
        return;
    end

    msg = strrep(msg, ',', ';');
    srcLCC = strrep(srcLCC, ',', ';');
    targetBathy = strrep(targetBathy, ',', ';');
    outVRT = strrep(outVRT, ',', ';');

    fprintf(fid, '%s,%g,%s,%g,%g,%g,%g,%g,%g,%s,%s,%s,%s\n', ...
        river, res, statusStr, rows, cols, ...
        N_total, N_water, N_nonwater, N_nodata, ...
        srcLCC, targetBathy, outVRT, msg);

    fclose(fid);
end