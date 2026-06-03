%% 2025年10月20日
% 根据下载得bathmatry数据进行vrt

%% 2025-10-20  Build VRT from a list of GeoTIFFs
clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Tiff_folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/';
List_text   = fullfile(Tiff_folder, 'Filelist.txt');   % 确保是绝对路径 & Unix 换行
dstfile     = fullfile(Tiff_folder, 'MD_PotomacRiver_Bathy.vrt');

cmd = sprintf(['gdalbuildvrt -overwrite ' ...
               '-input_file_list "%s" ' ...
               '"%s"'], List_text, dstfile);
cd(Tiff_folder)
status = system(cmd);
if status ~= 0
    error('gdalbuildvrt 失败，命令为：\n%s', cmd);
end

% 简单校验一下 VRT
system(sprintf('gdalinfo -stats "%s" | head -n 40', dstfile));
fprintf('VRT 已生成：%s\n', dstfile);




