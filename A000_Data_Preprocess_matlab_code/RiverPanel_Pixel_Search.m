function RiverPanel_Pixel_Search(bathy_vrt,final_mix_vrt,LCC_vrt,srcLine,SearchStep,winH,winW,OutFolder,BasinName)

%% Nov 4th, 2025
% input four kinds of vrt, bathmatry, bathmatry mix 3DEP, LCC, RiverLine shapefile should in same proj

% % 测试
% clear;clc;
% bathy_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/Kletzch_Combined_UpMax3Null.vrt';
% final_mix_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_Merge_3DEP_1m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority.vrt';
% LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_1m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip.vrt';
% srcLine = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_HR_ArtificialPath_Processed/Kletzch_Combined_UpMax3Null/NHDPlus_ArtificialPath_Resampleandclip.shp';
% SearchStep = 100;
% winH = 336;
% winW = 336;
% 
% OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE/';
% if exist(OutFolder,'dir') ~= 7
%     mkdir(OutFolder);
% end

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

GT = shaperead(srcLine);

% GT: shaperead 的 struct（Line，字段 X/Y，NaN 分段）
% spacing_m: 间距，单位=坐标单位（务必是米）
% include_endpoints: 是否包含段的终点（true/false）
% 返回 table: X,Y, LineID(原记录索引), S(沿线里程)

spacing_m = 100;  % 每 100 米取一个点（你的数据必须是米坐标系）
include_endpoints = true;

Xs=[]; Ys=[]; LID=[]; Spos=[];

disp('Step1: Extract srcline target')
for i = 1:numel(GT)
    x = GT(i).X(:); y = GT(i).Y(:);
    if isempty(x), continue; end
    brk = [0; find(~isfinite(x) | ~isfinite(y)); numel(x)+1];
    for k = 1:numel(brk)-1
        a = brk(k)+1; b = brk(k+1)-1;
        if b<=a, continue; end
        seg = [x(a:b) y(a:b)];
        if size(seg,1)<2, continue; end

        % 弧长累计
        d  = diff(seg,1,1);
        ds = hypot(d(:,1), d(:,2));
        S  = [0; cumsum(ds)];
        L  = S(end);
        if L < eps, continue; end

        % 等距里程桩
        sQ = 0:spacing_m:L;
        if include_endpoints && (sQ(end) ~= L)
            sQ(end+1) = L;
        end

        xi = interp1(S, seg(:,1), sQ, 'linear');
        yi = interp1(S, seg(:,2), sQ, 'linear');

        Xs  = [Xs; xi(:)];
        Ys  = [Ys; yi(:)];
        LID = [LID; repmat(i, numel(xi), 1)];
        Spos= [Spos; sQ(:)];
    end
end

P = table(Xs,Ys,LID,Spos,'VariableNames',{'X','Y','LineID','S'});

% % 根据这个的shp的point进行进一步的筛选
% % 读取FAC进行一个筛选判断
% 
% fac = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/NHDPlusV21_Data_Process/Range_fac_Proj.tif';
% [~,~,~,geoTrans,~,~,nodataval]=RasterInfo(fac);
% 
% % 预分配
% P.fac = nan(height(P), 1);
% 
% for i = 1 : height(P)
%     Points_Lon = P.X(i);
%     Points_Lat = P.Y(i);
%     [row0,col0]=Proj2RowCol(geoTrans,Points_Lat,Points_Lon);
% 
%     % 读取 1x1 像元窗口（按你当前函数的参数顺序）
%     fac_Point = ReadRaster(fac,row0,col0,1,1); % 读取这个点的fac
%     % NoData 处理
%     if ~isempty(nodataval) && isfinite(nodataval) && isequal(fac_Point, double(nodataval))
%         P.fac(i) = nan;
%     else
%         P.fac(i) = fac_Point;
%     end
%     disp([num2str(i),' fac extraction is done!'])
% end

% 进一步跟bathmatry进行一个提取范围
% 计算bathmatry栅格数量和占比情况
% 同时可以结合LCC的河流mask进行有效的bathmatry+3DEP的提取
% bathy_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/MD_PotomacRiver_Bathy.vrt';
% LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/LCC/ESA_WorldCover_1m_River_Proj.tif';
% final_mix_vrt   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Mix_3DEP_and_Bathy/Combined_Bathy_Priority.vrt';

clear rows cols geoTrans proj dataType nodataval
[~,rows,cols,geoTrans,~,~,~]=RasterInfo(final_mix_vrt);

% 预分配
P.bathamtry_Grid_num = nan(height(P), 1);
P.bathamtry_Grid_Ratio = nan(height(P), 1);
P.bathamtry_ValidGrid_num = nan(height(P), 1);
P.bathamtry_ValidGrid_Ratio = nan(height(P), 1);

% winH = 336;   % 行数（高度）
% winW = 336;   % 列数（宽度）
hr = floor(winH/2);
hc = floor(winW/2);

disp('Step2: srcline Points bathmatry extraction')

for i = 1 : height(P)
    Points_Lon = P.X(i);
    Points_Lat = P.Y(i);
    [row0,col0]=Proj2RowCol(geoTrans,Points_Lat,Points_Lon);

    % 以这个点为中心进行相应的提取
    % 3) 初始窗口左上角
    r1 = row0 - hr;
    c1 = col0 - hc;

    % 再计算右下角并反推实际尺寸（靠边时窗口会变小）
    r2 = r1 + winH - 1;
    c2 = c1 + winW - 1;
    h  = r2 - r1 + 1;
    w  = c2 - c1 + 1;

    % 判断是否超出bathmatry的范围
    if r1 < 0 || c1 < 0
        P.bathamtry_Grid_num(i) = nan;
        P.bathamtry_Grid_Ratio(i) = nan;
        P.bathamtry_ValidGrid_num(i) = nan;
        P.bathamtry_ValidGrid_Ratio(i) = nan;
        continue
    end

    if r2 > rows || c2 > cols
        P.bathamtry_Grid_num(i) = nan;
        P.bathamtry_Grid_Ratio(i) = nan;
        P.bathamtry_ValidGrid_num(i) = nan;
        P.bathamtry_ValidGrid_Ratio(i) = nan;
        continue
    end

    % 读取 这个tile的bathmatry
    tile = ReadRaster(bathy_vrt, r1, c1, h, w);
    tile_LCC = ReadRaster(LCC_vrt, r1, c1, h, w);
    tileBathmatry3DEP = ReadRaster(final_mix_vrt, r1, c1, h, w);

    % 检验有效的数据
    k_valid = find(~isnan(tile));
    P.bathamtry_Grid_num(i) = length(k_valid);
    P.bathamtry_Grid_Ratio(i) = length(k_valid)/(h*w);

    tileBathmatry3DEP_outRiver = tileBathmatry3DEP;
    tileBathmatry3DEP_outRiver(tile_LCC==80) = nan;
    k_valid_OutRiver = find(~isnan(tileBathmatry3DEP_outRiver));
    P.bathamtry_ValidGrid_num(i) = length(k_valid_OutRiver);
    P.bathamtry_ValidGrid_Ratio(i) = length(k_valid_OutRiver)/(h*w);

    clear Points_Lon Points_Lat
    clear row0 col0
    clear r1 c1 r2 c2 h w
    clear tile tile_LCC
    clear k_valid k_valid_OutRiver
    if mod(i,100) == 0
        disp([num2str(i),' bathmatry extraction is done!'])
    end
end


% 生成逐100m的点图,并且赋予属性
% 假设 P 是你得到的 table，包含 X,Y, LineID, S
outPts = fullfile([OutFolder,'/Range_ArtificoalPath_Points_',num2str(SearchStep),'m.shp']);
% srcLine = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/NHDPlusV21_Data_Process/Range_ArtificoalPath_2D.shp';   % 你的线shp（坐标系正确）

disp('Step3: srcline Points Output with table')
% 1) 组装点要素 struct 并写出
S = struct('Geometry','Point','X',[],'Y',[],'LineID',[],'S',[],'fac',[],'bathynum',[],'bathyRatio',[],'Validbathynum',[],'ValidbathyRatio',[]);
S = repmat(S, height(P), 1);
for i = 1:height(P)
    S(i).X = P.X(i);
    S(i).Y = P.Y(i);
    S(i).LineID = P.LineID(i);
    S(i).S = P.S(i);
    % S(i).fac = P.fac(i);
    S(i).bathynum = P.bathamtry_Grid_num(i);
    S(i).bathyRatio = P.bathamtry_Grid_Ratio(i);
    S(i).fac       = NaN;   % 或者实际的 P.fac(i)（若已计算）
    S(i).bathynum  = NaN;   % 或 P.bathamtry_Grid_num(i)
    S(i).bathyRatio= NaN;   % 或 P.bathamtry_Grid_Ratio(i)
    S(i).Validbathynum = P.bathamtry_ValidGrid_num(i);
    S(i).ValidbathyRatio = P.bathamtry_ValidGrid_Ratio(i);
end
shapewrite(S, outPts);

% 2) 复制 .prj（保证坐标系一致）
[~,base,~] = fileparts(outPts);
[pthL,baseL,~] = fileparts(srcLine);
srcPrj = fullfile(pthL, [baseL '.prj']);
dstPrj = fullfile(fileparts(outPts), [base '.prj']);
if exist(srcPrj,'file')
    copyfile(srcPrj, dstPrj, 'f');
else
    warning('未找到源 .prj：%s（可用 ogr2ogr -a_srs 补写）', srcPrj);
end

% 进行条件筛选，并生成相应的tile的tif
% 为合适的point赋予一个PointID

% 条件：有超过20% 的bathy数据，保证在主河道上
Bathmatry20_mask = isfinite(P.bathamtry_Grid_Ratio) & ~isnan(P.bathamtry_Grid_Ratio) & P.bathamtry_Grid_Ratio > 0.2;
P_Bathmatry20 = P(Bathmatry20_mask, :);

% 条件：有效的bathmatry栅格的ratio应该超过80%，但是不能到1
mask_Valid = isfinite(P_Bathmatry20.bathamtry_ValidGrid_Ratio) & ~isnan(P_Bathmatry20.bathamtry_ValidGrid_Ratio) & P_Bathmatry20.bathamtry_ValidGrid_Ratio > 0.8 & P_Bathmatry20.bathamtry_ValidGrid_Ratio < 1;
P_select = P_Bathmatry20(mask_Valid, :);

% % 条件：有效的bathmatry栅格的ratio应该超过80%，但是不能到1
% mask_Valid = isfinite(P.bathamtry_ValidGrid_Ratio) & ~isnan(P.bathamtry_ValidGrid_Ratio) & P.bathamtry_ValidGrid_Ratio > 0.8 & P.bathamtry_ValidGrid_Ratio < 1;
% P_select = P(mask_Valid, :);

% 生成符合条件的点
% 假设 P_select 是你得到的 table，包含 X,Y, LineID, S
outPts = fullfile([OutFolder,'/Select_Tile_CenterPoints.shp']);
% srcLine = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/NHDPlusV21_Data_Process/Range_ArtificoalPath_2D.shp';   % 你的线shp（坐标系正确）

disp('Step4: Qualified srcline Points Output with table')
% 1) 组装点要素 struct 并写出
S_select = struct('Geometry','Point','PointID',[],'X',[],'Y',[],'LineID',[],'S',[],'fac',[],'bathynum',[],'bathyRatio',[],'Validbathynum',[],'ValidbathyRatio',[]);
S_select = repmat(S_select, height(P_select), 1);
for i = 1:height(P_select)
    S_select(i).PointID = i;
    S_select(i).X = P_select.X(i);
    S_select(i).Y = P_select.Y(i);
    S_select(i).LineID = P_select.LineID(i);
    S_select(i).S = P_select.S(i);
    % S_select(i).fac = P_select.fac(i);
    S_select(i).bathynum = P_select.bathamtry_Grid_num(i);
    S_select(i).bathyRatio = P_select.bathamtry_Grid_Ratio(i);
    S_select(i).fac       = NaN;   % 或者实际的 P.fac(i)（若已计算）
    S_select(i).bathynum  = NaN;   % 或 P.bathamtry_Grid_num(i)
    S_select(i).bathyRatio= NaN;   % 或 P.bathamtry_Grid_Ratio(i)
    S_select(i).Validbathynum = P_select.bathamtry_ValidGrid_num(i);
    S_select(i).ValidbathyRatio = P_select.bathamtry_ValidGrid_Ratio(i);
end
shapewrite(S_select, outPts);

% 2) 复制 .prj（保证坐标系一致）
[~,base,~] = fileparts(outPts);
[pthL,baseL,~] = fileparts(srcLine);
srcPrj = fullfile(pthL, [baseL '.prj']);
dstPrj = fullfile(fileparts(outPts), [base '.prj']);
if exist(srcPrj,'file')
    copyfile(srcPrj, dstPrj, 'f');
else
    warning('未找到源 .prj：%s（可用 ogr2ogr -a_srs 补写）', srcPrj);
end

% 生成tile的对应情况
% BathmatryTile_outFloder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Z990_Select_1mtile_for_MAE/Only_Bathmatry/';
% mkdir(BathmatryTile_outFloder)
Bathmatry3DEP_Tile_outFloder = fullfile([OutFolder,'/Tiles_Bathmatry_Combined_3DEP/']);
mkdir(Bathmatry3DEP_Tile_outFloder)

clear rows cols geoTrans proj dataType nodataval
[~,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(final_mix_vrt);

disp('Step5: Qualified srcline Points Tile Bathmatry&3DEP Tiff output')

for i = 1 : height(P_select)
    Points_Lon = P_select.X(i);
    Points_Lat = P_select.Y(i);
    PointID = S_select(i).PointID;
    [row0,col0]=Proj2RowCol(geoTrans,Points_Lat,Points_Lon);

    % 以这个点为中心进行相应的提取
    % 3) 初始窗口左上角
    r1 = row0 - hr;
    c1 = col0 - hc;

    % 再计算右下角并反推实际尺寸（靠边时窗口会变小）
    r2 = r1 + winH - 1;
    c2 = c1 + winW - 1;
    h  = r2 - r1 + 1;
    w  = c2 - c1 + 1;

    % 读取 这个tile的bathmatry
    % tile = ReadRaster(bathy_vrt, r1, c1, h, w);
    tile_LCC = ReadRaster(LCC_vrt, r1, c1, h, w);
    tileBathmatry3DEP = ReadRaster(final_mix_vrt, r1, c1, h, w);

    % tile_outRiver = tile;
    tileBathmatry3DEP_outRiver = tileBathmatry3DEP;
    % tile_outRiver(tile_LCC==80) = nan;
    tileBathmatry3DEP_outRiver(tile_LCC==80) = nan;

    % 生成tile的tiff
    outFormat = 'GTiff';
    subgeoTrans = subTranscoef(geoTrans,r1,c1);

    % fileRas = fullfile([BathmatryTile_outFloder,'Select_tileOutRiver_',num2str(PointID),'.tiff']);
    % WriteRaster(fileRas,tile_outRiver,subgeoTrans,proj,dataType,outFormat,nodataval)

    % Mask
    dataType2 = 1;
    nodataval2 = 255;
    fileRas_LCC_Mask = fullfile([Bathmatry3DEP_Tile_outFloder,'Select_tile_',num2str(PointID),'_LCC_Mask.tiff']);
    WriteRaster(fileRas_LCC_Mask,tile_LCC,subgeoTrans,proj,dataType2,outFormat,nodataval2) 
    
    % mask 之后的
    fileRas_Bathmatry3DEP_outRiver = fullfile([Bathmatry3DEP_Tile_outFloder,'Select_tileOutRiver_',BasinName,'_ID',num2str(PointID),'.tiff']);
    WriteRaster(fileRas_Bathmatry3DEP_outRiver,tileBathmatry3DEP_outRiver,subgeoTrans,proj,dataType,outFormat,nodataval)
    % 所有的bathy+3DEP
    fileRas_Bathmatry3DEP = fullfile([Bathmatry3DEP_Tile_outFloder,'Select_tile_Basin_',BasinName,'_ID',num2str(PointID),'.tiff']);
    WriteRaster(fileRas_Bathmatry3DEP,tileBathmatry3DEP,subgeoTrans,proj,dataType,outFormat,nodataval)

    clear subgeoTrans
    clear fileRas fileRas_Bathmatry3DEP fileRas_Bathmatry3DEP_outRiver
    clear Points_Lon Points_Lat PointID
    clear row0 col0
    clear r1 c1 r2 c2 h w
    clear tile tile_LCC tileBathmatry3DEP
    clear tile_outRiver tileBathmatry3DEP_outRiver
    disp([num2str(i),' Tile extraction is done!'])
end

