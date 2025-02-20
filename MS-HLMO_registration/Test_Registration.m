function [cor1,cor2] = Test_Registration(I1,I2,keypoints_1,keypoints_2,patch_size,NBS,NBO,nOctaves1,nOctaves2,nLayers,G_resize,G_sigma,rotation_flag,Error,K)

%% Keypoints Description
tic,descriptors_1 = Multiscale_Descriptor(I1,keypoints_1,patch_size,NBS,NBO,...
    nOctaves1,nLayers,G_resize,G_sigma,rotation_flag);
    str=['Done: Keypoints description of reference image, time cost: ',num2str(toc),'s\n']; fprintf(str);
tic,descriptors_2 = Multiscale_Descriptor(I2,keypoints_2,patch_size,NBS,NBO,...
    nOctaves2,nLayers,G_resize,G_sigma,rotation_flag);
    str=['Done: Keypoints description of sensed image, time cost: ',num2str(toc),'s\n\n']; fprintf(str);

%% Keypoints Matching
tic
if K==1
    [cor1_o,cor2_o] = Multiscale_Matching(descriptors_1,descriptors_2,...
        nOctaves1,nOctaves2,nLayers);
    [cor1,cor2] = Outlier_Removal(cor1_o,cor2_o,Error);
else
    aaa = []; correspond_1 = cell(K,1); correspond_2 = cell(K,1);
    for k = 1:K
        k
        [cor1_o,cor2_o] = Multiscale_Matching(descriptors_1,descriptors_2,...
            nOctaves1,nOctaves2,nLayers);
        [cor1,cor2] = Outlier_Removal(cor1_o,cor2_o,Error);
        correspond_1{k} = cor1; correspond_2{k} = cor2; aaa = [aaa,size(cor1,1)];
    end
    [~,index] = max(aaa);
    cor1 = correspond_1{index}; cor2 = correspond_2{index};
end
    str = ['Done: Keypoints matching, time cost: ',num2str(toc),'s\n']; fprintf(str);
