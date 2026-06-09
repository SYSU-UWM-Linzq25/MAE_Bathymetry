function Summary = B000_run_AllTrainingRivers_CanonicalND(varargin)
%% Run the staged canonical-NoData rebuild for all 12 MAE rivers.
%
% This master driver intentionally starts from the existing
% Bathy_1m_FixND products. Therefore:
%   - OR_MKRC keeps the vertical feet->meters and 2 ft->true 1 m correction;
%   - KewaFix2Null keeps the vertical feet->meters correction;
%   - no unit conversion is applied twice.
%
% New products are written to separate staged folders. Nothing under the
% official MAE Data folder is overwritten by this function.

p = inputParser;
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'selectedRivers', {}, @(x) iscell(x) || isstring(x));
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'continueOnError', false, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);
parse(p, varargin{:});

rootPR = char(p.Results.rootPR);
if p.Results.doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

R = B000_get_MAE_river_registry();
if ~isempty(p.Results.selectedRivers)
    requested = string(p.Results.selectedRivers(:));
    missing = requested(~ismember(requested, R.River));
    if ~isempty(missing)
        error('Unknown river(s): %s', strjoin(cellstr(missing), ', '));
    end
    R = R(ismember(R.River, requested), :);
end

% Validate that the one-time unit-fix outputs exist. We do not recreate or
% overwrite them here.
orSentinel = fullfile(rootPR, 'Bathy_1m_FixND', ...
    'OR_MKRC_Topobathy_2021', 'Bathy_2ft_elev_m.tif');
kewaSentinel = fullfile(rootPR, 'Bathy_1m_FixND', ...
    'KewaFix2Null', 'Bathy_1m.tif');

if any(R.River == "OR_MKRC_Topobathy_2021") && exist(orSentinel, 'file') ~= 2
    error(['Missing OR_MKRC unit-fix sentinel: %s\nRun ' ...
           'B001_02_fix_OR_MKRC_unit_only.m first.'], orSentinel);
end
if any(R.River == "KewaFix2Null") && exist(kewaSentinel, 'file') ~= 2
    error(['Missing Kewa unit-fix sentinel: %s\nRun ' ...
           'B001_05_fix_KewaFix2Null_unit_only.m first.'], kewaSentinel);
end

n = height(R);
Status = strings(n,1);
Message = strings(n,1);
StartTime = strings(n,1);
EndTime = strings(n,1);

for i = 1:n
    river = char(R.River(i));
    StartTime(i) = string(datetime('now'));

    fprintf('\n\n############################################################\n');
    fprintf('[%d/%d] CanonicalND pipeline: %s\n', i, n, river);
    fprintf('Unit policy       : %s\n', R.UnitPolicy(i));
    fprintf('ZeroIsNoData      : %d\n', R.ZeroIsNoData(i));
    fprintf('ForbidZeroOutput  : %d\n', R.ForbidZeroOutput(i));
    fprintf('############################################################\n');

    try
        B001_10_Canonicalize_Bathy_ForRiver(river, ...
            'rootPR', rootPR, 'res', 1, ...
            'zeroIsNoData', R.ZeroIsNoData(i), ...
            'overwrite', p.Results.overwrite, 'doPathSetup', false);

        B001_12_Rebuild_Bathy3DEP_Merge_ForRiver(river, ...
            'rootPR', rootPR, 'res', 1, ...
            'forbidZeroOutput', R.ForbidZeroOutput(i), ...
            'overwrite', p.Results.overwrite, 'doPathSetup', false);

        B003s_10_Build_SimpleFinalMask_FromCanonicalBathy(river, ...
            'rootPR', rootPR, 'res', 1, ...
            'zeroIsNoDataFallback', R.ZeroIsNoData(i), ...
            'overwrite', p.Results.overwrite, 'doPathSetup', false);

        B005b_10_Reextract_SelectedTiles_CanonicalND(river, ...
            'rootPR', rootPR, 'res', 1, ...
            'forbidZeroOutput', R.ForbidZeroOutput(i), ...
            'overwrite', p.Results.overwrite, 'doPathSetup', false);

        Status(i) = "PASS";
        Message(i) = "";
    catch ME
        Status(i) = "FAIL";
        Message(i) = string(ME.message);
        fprintf(2, '\n[FAIL] %s\n%s\n', river, getReport(ME));
        if ~p.Results.continueOnError
            EndTime(i) = string(datetime('now'));
            Summary = [R, table(Status, Message, StartTime, EndTime)];
            outCsv = fullfile(rootPR, 'Z020_CanonicalND_AllRivers_Summary.csv');
            writetable(Summary, outCsv);
            rethrow(ME);
        end
    end
    EndTime(i) = string(datetime('now'));
end

Summary = [R, table(Status, Message, StartTime, EndTime)];
outCsv = fullfile(rootPR, 'Z020_CanonicalND_AllRivers_Summary.csv');
writetable(Summary, outCsv);

fprintf('\n============================================================\n');
fprintf('All-river CanonicalND pipeline finished.\n');
fprintf('PASS=%d FAIL=%d\n', nnz(Status=="PASS"), nnz(Status=="FAIL"));
fprintf('Summary: %s\n', outCsv);
fprintf('============================================================\n');
end
