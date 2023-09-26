close all
clear all
clc

load bvals.txt
load bvecs.txt

% bvals1 = bvals(bvals<1500);
% bvecs1 = bvecs(:,bvals<1500);

mask = load_nii('nodif_brain_mask.nii.gz');
amask = mask.img;
amask = double(amask(end:-1:1,:,72:73));

se = strel('disk',3);        
amask = imerode(amask,se);

nii = load_nii('data_2slice.nii.gz');

A = nii.img(:,:,:,:);

A = A(end:-1:1,:,:,:);

% Simg = A(:,:,1,bvals<1500);
Simg = A(:,:,1,:);

nii1 = load_nii('grad_dev_2slice.nii.gz');

A1 = nii1.img(:,:,:,:);

A1 = A1(end:-1:1,:,:,:);

g = double(A1(:,:,1,:));

S = zeros(size(Simg,4),size(Simg,1),size(Simg,2));
new_bvals = zeros(1,size(Simg,4),size(Simg,1),size(Simg,2));
new_bvecs = zeros(3,size(Simg,4),size(Simg,1),size(Simg,2));

for i = 1:size(Simg,1)
    for j = 1:size(Simg,2)
        if amask(i,j,1)~=0
            [new_bvals(:,:,i,j),new_bvecs(:,:,i,j)] = bval_bvec_correction(g(i,j,:),bvals,bvecs);
            for k = 1:size(Simg,4)
                S(k,i,j) = double(Simg(i,j,k));
                if S(k,i,j) == 0
                    S(k,i,j) = 1;
                end
            end
        end
    end
end

Dcnls = zeros(3,3,size(Simg,1),size(Simg,2));
S0 = zeros(size(Simg,1),size(Simg,2));
resnorm = zeros(size(Simg,1),size(Simg,2));

for i = 1:size(Simg,1)
    for j = 1:size(Simg,2)
        if amask(i,j,1)~=0
            [Dcnls(:,:,i,j), S0(i,j), resnorm(i,j)] = constra_tensor_est(S(:,i,j),...
                new_bvals(:,:,i,j),new_bvecs(:,:,i,j));
        end
    end
end

big_delta = 43.1*10^-3;
little_delta = 10.6*10^-3;
D0 = 0.003;

alpha = zeros(3,3,size(Simg,1),size(Simg,2));
S0_alpha = zeros(size(Simg,1),size(Simg,2));
resnorm_alpha = zeros(size(Simg,1),size(Simg,2));

for i = 1:size(Simg,1)
    for j = 1:size(Simg,2)
        if amask(i,j,1)~=0
            [alpha(:,:,i,j),S0_alpha(i,j),resnorm_alpha(i,j)] = making_alpha(Dcnls(:,:,i,j), D0, little_delta, big_delta,S(:,i,j),...
                new_bvals(:,:,i,j),new_bvecs(:,:,i,j));
        end
    end
end

trace_L_tensor = zeros(size(Simg,1),size(Simg,2));
FA_tensor = zeros(size(Simg,1),size(Simg,2));
cl_tensor = zeros(size(Simg,1),size(Simg,2));
cp_tensor = zeros(size(Simg,1),size(Simg,2));
cs_tensor = zeros(size(Simg,1),size(Simg,2));
cm_tensor = zeros(size(Simg,1),size(Simg,2),3);

trace_L_alpha = zeros(size(Simg,1),size(Simg,2));
FA_alpha = zeros(size(Simg,1),size(Simg,2));
cl_alpha = zeros(size(Simg,1),size(Simg,2));
cp_alpha = zeros(size(Simg,1),size(Simg,2));
cs_alpha = zeros(size(Simg,1),size(Simg,2));
cm1_alpha = zeros(size(Simg,1),size(Simg,2),3);
cm3_alpha = zeros(size(Simg,1),size(Simg,2),3);

for i = 1:size(Simg,1)
    for j = 1:size(Simg,2)
        if amask(i,j,1)~=0
            [trace_L_tensor(i,j),FA_tensor(i,j),cl_tensor(i,j),cp_tensor(i,j), ...
                cs_tensor(i,j),cm_tensor(i,j,:)] = aniso_est(Dcnls(:,:,i,j));
            
            [trace_L_alpha(i,j),FA_alpha(i,j),cl_alpha(i,j),cp_alpha(i,j), ...
                cs_alpha(i,j),cm1_alpha(i,j,:),cm3_alpha(i,j,:)] = aniso_est_alpha(alpha(:,:,i,j));
        end
    end
end

[X,Y,Z] = sphere(10);
xyz = [X(:),Y(:),Z(:)];
sx = size(X);

figure
for i = 1:round(size(Simg,1))
    for j = 1:round(size(Simg,2))
        if amask(i,j,1)~=0
            xyze = xyz*Dcnls(:,:,i,j);
            m_xyze = sqrt(xyze(:,1).^2 + xyze(:,2).^2 + xyze(:,3).^2);
            max_xyze = max(m_xyze);
            xyze = [xyze(:,1)./max_xyze xyze(:,2)./max_xyze xyze(:,3)./max_xyze];
            Xe = reshape(xyze(:,1),sx);
            Ye = reshape(xyze(:,2),sx);
            Ze = reshape(xyze(:,3),sx);
            surf(Xe+2*i,Ye+2*j,Ze,reshape(m_xyze/max_xyze,11,11));
            hold on
        end
    end
end

axis equal
axis vis3d
camlight right
lighting phong
shading interp
xlabel X
ylabel Y
zlabel Z
view(0,90)
title('diffusion-tensor')

figure
for i = 1:round(size(Simg,1))
    for j = 1:round(size(Simg,2))
        if amask(i,j,1)~=0
        xyze = xyz*alpha(:,:,i,j);
        m_xyze = sqrt(xyze(:,1).^2 + xyze(:,2).^2 + xyze(:,3).^2);
        max_xyze = max(m_xyze);
        xyze = [xyze(:,1)./max_xyze xyze(:,2)./max_xyze xyze(:,3)./max_xyze];
        Xe = reshape(xyze(:,1),sx);
        Ye = reshape(xyze(:,2),sx);
        Ze = reshape(xyze(:,3),sx);
        surf(Xe+2*i,Ye+2*j,Ze,reshape(m_xyze/max_xyze,11,11));
        hold on
        end
    end
end

axis equal
axis vis3d
camlight right
lighting phong
shading interp
xlabel X
ylabel Y
zlabel Z
view(0,90)
title('alpha-tensor')

figure,colormap(gray),imagesc(rot90(trace_L_tensor)),title('trace-diffusion-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(FA_tensor)),title('FA-diffusion-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cl_tensor)),title('cl-diffusion-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cp_tensor)),title('cp-diffusion-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cs_tensor)),title('cs-diffusion-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(S0)),title('S0-tensor'),axis equal
figure,imagesc(rot90(cm_tensor)),title('cm-diffusion-tensor'),axis equal
I2 = resnorm/max(resnorm(:))/288;
K2 = imadjust(I2,[0 0.001],[]);
figure,colormap(gray),imagesc(rot90(K2)),title('resnorm-tensor'),axis equal

I1 = (trace_L_alpha)/max(trace_L_alpha(:));
K = imadjust(I1,[0 0.1],[]);
figure,colormap(gray),imagesc(rot90(K)),title('trace-alpha-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(FA_alpha)),title('FA-alpha-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cl_alpha)),title('cl-alpha-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cp_alpha)),title('cp-alpha-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(cs_alpha)),title('cs-alpha-tensor'),axis equal
figure,colormap(gray),imagesc(rot90(S0_alpha)),title('S0-alpha'),axis equal
figure,imagesc(rot90(cm1_alpha)),title('cm-alpha-tensor-third'),axis equal
figure,imagesc(rot90(cm3_alpha)),title('cm-alpha-tensor-first'),axis equal

I3 = resnorm_alpha/max(resnorm_alpha(:))/288;
K3 = imadjust(I3,[0 0.001],[]);
figure,colormap(gray),imagesc(rot90(K3)),title('resnorm-alpha'),axis equal
