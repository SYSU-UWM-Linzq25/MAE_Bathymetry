function B002c_Check_LCC_BathyGrid_All(varargin)
%% ============================================================
%  B002c_Check_LCC_BathyGrid_All.m
%
%  Purpose:
%    Check whether LCC_<res>m grid matches Bathy_<res>m_FixND grid.
%
%  Output:
%    Z009_Check_LCC_BathyGrid/Check_LCC_BathyGrid.csv
% ============================================================

p = inputParser;

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'targetRes', [1 3 5 10], @isnumeric);
addParameter(p, 'selectedRivers', {}, @(x) iscell(x) || isstring(x));
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, varargin{:});

rootPR = char(p.Results.rootPR);
targetRes = p.Results.targetRes;
selectedRivers = p.Results.selectedRivers;
doPathSetup = p.Results.doPathSetup;

if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

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

outDir = fullfile(rootPR, 'Z009_Check_LCC_BathyGrid');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

outCSV = fullfile(outDir, 'Check_LCC_BathyGrid.csv');

rowsOut = {};
header = { ...
    'River', ...
    'Resolution_m', ...
    'Bathy_exists', ...
    'LCC_exists', ...
    'Bathy_rows', ...
    'Bathy_cols', ...
    'LCC_rows', ...
    'LCC_cols', ...
    'SameSize', ...
    'MaxGeoDiff', ...
    'SameGeo', ...
    'Status'};

rowsOut(1,:) = header;

fprintf('\n============================================================\n');
fprintf('Check LCC grid against Bathy grid\n');
fprintf('============================================================\n');

for iRiver = 1:numel(rivers)

    river = rivers{iRiver};

    if contains(river, 'NoNeed')
        continue;
    end

    for j = 1:numel(targetRes)

        res = targetRes(j);

        bathy_vrt = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
            river, sprintf('Bathy_%dm.vrt', res));

        lcc_vrt = fullfile(rootPR, sprintf('LCC_%dm', res), ...
            river, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));

        bathy_exists = exist(bathy_vrt, 'file') == 2;
        lcc_exists   = exist(lcc_vrt, 'file') == 2;

        rowsB = NaN; colsB = NaN;
        rowsL = NaN; colsL = NaN;
        maxGeoDiff = NaN;
        sameSize = false;
        sameGeo = false;
        status = 'MISSING';

        if bathy_exists && lcc_exists

            [~, rowsB, colsB, geoB, ~, ~, ~] = RasterInfo(bathy_vrt);
            [~, rowsL, colsL, geoL, ~, ~, ~] = RasterInfo(lcc_vrt);

            sameSize = (rowsB == rowsL) && (colsB == colsL);
            maxGeoDiff = max(abs(geoB(:) - geoL(:)));
            sameGeo = maxGeoDiff < 1e-8;

            if sameSize && sameGeo
                status = 'PASS';
            elseif sameSize && ~sameGeo
                status = 'SIZE_PASS_GEO_FAIL';
            else
                status = 'SIZE_FAIL';
            end
        end

        fprintf('%-40s %2dm  Bathy=%d/%d  LCC=%d/%d  %s\n', ...
            river, res, rowsB, colsB, rowsL, colsL, status);

        rowsOut(end+1,:) = { ...
            river, ...
            res, ...
            bathy_exists, ...
            lcc_exists, ...
            rowsB, ...
            colsB, ...
            rowsL, ...
            colsL, ...
            sameSize, ...
            maxGeoDiff, ...
            sameGeo, ...
            status};
    end
end

writecell(rowsOut, outCSV);

fprintf('\nCheck table written to:\n%s\n', outCSV);
fprintf('============================================================\n');

end