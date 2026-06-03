%% 生成vrt的范围用于下载3DEP
clear
clc

Tiff_folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/';
vrtfile     = fullfile(Tiff_folder, 'MD_PotomacRiver_Bathy.vrt');

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

[nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(vrtfile);

% 找到坐标的范围
[Lat0,Lon0]=RowCol2Proj(geoTrans,1,1);
[Lat1,Lon1]=RowCol2Proj(geoTrans,rows,cols);

XMIN = min([Lon0 Lon1]);
XMAX = max([Lon0 Lon1]);
YMIN = min([Lat0 Lat1]);
YMAX = max([Lat0 Lat1]);


% 写四角点（NW, NE, SE, SW），一行一个“X Y”
outDir = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/';
fn = fullfile(outDir,'utm_corners.txt');
fid = fopen(fn,'w');
fprintf(fid,'%.3f %.3f\n', XMIN, YMAX);  % NW
fprintf(fid,'%.3f %.3f\n', XMAX, YMAX);  % NE
fprintf(fid,'%.3f %.3f\n', XMAX, YMIN);  % SE
fprintf(fid,'%.3f %.3f\n', XMIN, YMIN);  % SW
fclose(fid);

fprintf('写出角点：%s\n', fn);


%% 建立3DEP的vrt
clear;clc;

Tiff_folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/DEM_1m_raw/';
List_text   = fullfile(Tiff_folder, 'Filelist.txt');   % 确保是绝对路径 & Unix 换行
dstfile     = fullfile(Tiff_folder, 'MD_PotomacRiver_3DEP.vrt');

cmd = sprintf(['gdalbuildvrt -overwrite ' ...
               '-input_file_list "%s" ' ...
               '"%s"'], List_text, dstfile);
cd(Tiff_folder)
status = system(cmd);
if status ~= 0
    error('gdalbuildvrt 失败，命令为：\n%s', cmd);
end

% 简单校验一下 VRT
system(sprintf('gdalinfo -stats "%s" | head -n 10', dstfile));
fprintf('VRT 已生成：%s\n', dstfile);

