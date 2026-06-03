%% 2025年10月21日

% LCC处理
% 前期的转投影，mosaic和裁剪区域在GIS中完成

clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

LCC_Data   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/LCC/ESA_WorldCover_10m_RangeExtract_proj.tif';
outFile   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/LCC/ESA_WorldCover_10m_RangeExtract_proj_River.tif';

[nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(LCC_Data);

LCC = ReadRaster(LCC_Data, 1, 1, rows, cols); 

% 仅保留80（河道），其它设为 NoData
mask_keep = (LCC == 80);
LCC(~mask_keep) = nodataval;

WriteRaster(outFile, LCC, geoTrans, proj, dataType, 'GTiff', nodataval);

%% 将LCC进行一步Resampleandclip, 后续便能够直接用这个判断bathmatry的栅格是否在河道里面
% 注意resampleandclip应该选最近邻的办法，不能够使用双线性插值，否则会出现一些不明所以的值
% 其实只有河道那里有值

clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

LCC_Data   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/LCC/ESA_WorldCover_10m_RangeExtract_proj_River.tif';
bathy_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/MD_PotomacRiver_Bathy.vrt';
dstFile = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/LCC/ESA_WorldCover_1m_River_Proj.tif';

[~,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(bathy_vrt);

ResampleAndClip(geoTrans, proj, cols,...
    rows, LCC_Data, dstFile, 'GTiff', 1, 1); % use 1 nearest




