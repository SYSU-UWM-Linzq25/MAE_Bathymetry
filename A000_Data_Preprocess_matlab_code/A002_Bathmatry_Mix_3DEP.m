%% 2025年10月21日

% 融合Bathmatry和3DEP
% 以前者为准

% 这一步先将3DEP的栅格重采样到bathmatry的栅格上面

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

bathy_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/MD_PotomacRiver_Bathy.vrt';
dep3_vrt  = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/DEM_1m_raw/MD_PotomacRiver_3DEP.vrt';
dstFile = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/DEM_1m_raw/MD_PotomacRiver_3DEP_Proj.vrt';

[nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(bathy_vrt);

ResampleAndClip(geoTrans, proj, cols,...
    rows, dep3_vrt, dstFile, 'GTiff', 1, 2); % use 2 Bilinear




%% 第二次mosaic 和build vrt
clear;clc;
dep3_vrt_proj = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/DEM_1m_raw/MD_PotomacRiver_3DEP_Proj.vrt';
bathy_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/MD_PotomacRiver_Bathy.vrt';
out_vrt   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/Combined_Bathy_Priority.vrt';

cd('/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/')
lst = tempname; fid = fopen(lst,'w');
fprintf(fid, '%s\n', dep3_vrt_proj);   % 先 3DEP
fprintf(fid, '%s\n', bathy_vrt);  % 后 Bathy（优先）
fclose(fid);

cmd = sprintf(['gdalbuildvrt -overwrite -resolution highest -r bilinear -hidenodata ' ...
               '-input_file_list "%s" "%s"'], lst, out_vrt);
assert(system(cmd)==0, 'gdalbuildvrt failed');
% delete(lst);



%% 裁剪部分生成tiff在本地进行验证
clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

final_mix_vrt   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/Combined_Bathy_Priority.vrt';
outFile   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/Combined_Bathy_Priority_tile.tif';
outFile2   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/Combined_Bathy_Priority_tile2.tif';

[nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(final_mix_vrt);

block = 5000;
nodataval_out = -9999;

tile = ReadRaster(final_mix_vrt, 1, 1, block, block);  % double，但仅这一块
WriteRaster(outFile, tile, geoTrans, proj, dataType, 'GTiff', nodataval_out);

tile2 = ReadRaster(final_mix_vrt, 1, 1, 1000, cols);  % double，但仅这一块
WriteRaster(outFile2, tile2, geoTrans, proj, dataType, 'GTiff', nodataval_out);





