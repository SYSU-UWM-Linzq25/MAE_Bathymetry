function R = B000_get_MAE_river_registry()
%% Registry for the 12 rivers currently used by the 1 m MAE train/holdout set.
%
% UnitPolicy is informational and is also used by the master driver to
% validate that the two one-time unit corrections already exist.
%
% IMPORTANT:
%   This pipeline starts from Bathy_1m_FixND/<river>/Bathy_1m.vrt.
%   It never reconverts units. Therefore the corrected OR_MKRC and Kewa
%   products remain fully compatible.
%
% ZeroIsNoData is deliberately FALSE for every river at present.
% The Santiam investigation found zero pixels in the old extracted tiles,
% but zero pixels were NOT present in the canonical Bathy input. The actual
% permanent fix is explicit NoData writing and VRT metadata, not a global
% "zero means NoData" rule.
%
% Only set ZeroIsNoData=true after river-specific evidence shows that zero is
% a fill value in the SOURCE bathymetry rather than a legitimate elevation.

River = string({ ...
    'BadgerFinNull'
    'CA_KlamathRiver_TopoBathy_2018_D18'
    'CO_UpperColorado_Topobathy_1_2020'
    'Estabrook_Combined'
    'KewaFix2Null'
    'Kletzch_Combined_UpMax3Null'
    'MD_PotomacRiver_Bathy_2019'
    'NE_Niobrara_Topobathy_2018'
    'OR_MKRC_Topobathy_2021'
    'OR_SantiamRiverTB_Topobathy_1_D23'
    'WA_ChehalisRiverTB_Topobathy_1_D23'
    'WA_Nisqually_Bathymetric_2020'
    });

UnitPolicy = strings(size(River));
UnitPolicy(:) = "already_meter";
UnitPolicy(River == "OR_MKRC_Topobathy_2021") = ...
    "already_fixed_vertical_ft_to_m_and_horizontal_2ft_to_true_1m";
UnitPolicy(River == "KewaFix2Null") = ...
    "already_fixed_vertical_ft_to_m_horizontal_grid_already_1m";

ZeroIsNoData = false(size(River));

% Santiam's valid bathy range is far above zero and the rebuilt product must
% contain no valid zero output. Keep this strict check for the confirmed case.
ForbidZeroOutput = false(size(River));
ForbidZeroOutput(River == "OR_SantiamRiverTB_Topobathy_1_D23") = true;

IsHoldout = false(size(River));
IsHoldout(River == "OR_SantiamRiverTB_Topobathy_1_D23") = true;

R = table(River, UnitPolicy, ZeroIsNoData, ForbidZeroOutput, IsHoldout);
end
