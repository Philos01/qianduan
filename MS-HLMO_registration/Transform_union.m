function[I1_r,I2_r,I1_rs,I2_rs,fusion,mosaic,theta1x,theta1y,trans_point1]=...
    Transform_union(I1_o,I2_o,loc1,loc2,trans_form)

    tform=fitgeotrans(loc2(:,1:2),loc1(:,1:2),trans_form);
    solution=tform.T;

    [M1,N1,B1]=size(I1_o);
    [M2,N2,B2]=size(I2_o);

    C1=[1,1,1]*solution;C2=[N2,1,1]*solution;
    C3=[1,M2,1]*solution;C4=[N2,M2,1]*solution;
    dX=min([C1(1),C2(1),C3(1),C4(1)]);dX=ceil((dX<=1)*(1-dX));
    dY=min([C1(2),C2(2),C3(2),C4(2)]);dY=ceil((dY<=1)*(1-dY));
    sX=max([C1(1),C2(1),C3(1),C4(1)]);sX=ceil(max(N1,sX)+dX);
    sY=max([C1(2),C2(2),C3(2),C4(2)]);sY=ceil(max(M1,sY)+dY);

    fixed_point = [0,0];
    solution_1=[1,0,dX;0,1,dY;0,0,1];
    tform=maketform('projective',solution_1');
    I1_r=imtransform(I1_o,tform,'XYScale',1,'XData',[1,sX],'YData',[1,sY]);
    % 使用仿射变换矩阵 tform 将固定点进行变换，得到变换后的点
    trans_point1 = tformfwd(tform, fixed_point);
    disp(solution_1); % 显示 solution_1 的内容
    % 读取 X 轴和 Y 轴的缩放因子
    sx1 = tform.tdata.T(1,1); % X 轴的缩放因子
    sy1 = tform.tdata.T(2,2); % Y 轴的缩放因子
    % 读取旋转角度
    theta1x = atan2(tform.tdata.T(1,2), tform.tdata.T(1,1)); % 计算x轴y轴旋转角度
    theta1y = atan2(tform.tdata.T(2,1), tform.tdata.T(2,2))

    tform=maketform('projective',solution*solution_1');
    I2_r=imtransform(I2_o,tform,'XYScale',1,'XData',[1,sX],'YData',[1,sY]);
    trans_point2 = tformfwd(tform, fixed_point);
    % 读取 X 轴和 Y 轴的缩放因子
    sx2 = tform.tdata.T(1,1); % X 轴的复合缩放因子
    sy2 = tform.tdata.T(2,2); % Y 轴的复合缩放因子
    % 读取旋转角度
    theta2 = atan2(tform.tdata.T(2,1) , tform.tdata.T(1,2)); % 计算复合旋转角度

    if B1~=1&&B1~=3
        I1_rs=sum(double(I1_r),3);
    else
        I1_rs=double(I1_r);
    end

    if B2~=1&&B2~=3
        I2_rs=sum(double(I2_r),3);
    else
        I2_rs=double(I2_r);
    end

    if(B1==1&&B2==3)
        temp=I1_rs;
        I1_rs(:,:,1)=temp;I1_rs(:,:,2)=temp;I1_rs(:,:,3)=temp;
    elseif(B1==3&&B2==1)
        temp=I2_rs;
        I2_rs(:,:,1)=temp;I2_rs(:,:,2)=temp;I2_rs(:,:,3)=temp;
    end
    I1_rs=Visual(I1_rs);
    I2_rs=Visual(I2_rs);
    fusion=I1_rs/2+I2_rs/2;

    grid_num=10;
    grid_size=floor(min(size(I1_rs,1),size(I1_rs,2))/grid_num);
    [~,~,mosaic]=Mosaic_Map(I1_rs,I2_rs,grid_size);