addpath('MatlabAPI')

dataDir='..'; prefix='instances';
dataType='train2014';
dataType_res='train2014resize_256';   % datatype resize
labelfile_path = 'labelfiles';

transtype = 'combine';
gttype = 'gt_combine';

prop_low = 0.01;
prop_up = 0.5;

new_size = 256;
save_resized_ori_img = 1;

% the shift limit
shift_up = 127;
shift_down = -127;


rotate_up = 10;
rotate_down = -10;

lumin_up = 32;
lumin_down = -32;

scale_up = 4;
scale_down = 0.5;

deform_up = 2;
deform_down = 0.5;

max_count = 2000;

%%%%%%%%%%%%%%%%%
% The save path of generated image pairs
%%%%%%%%%%%%%%%%%
save_root = 'dataprepare/DMAC-COCO';

save_dir = sprintf('%s/%s/',save_root, dataType);
save_dir_sub = sprintf('%s%s/',save_dir,transtype);
system(['rm -rf ', save_dir_sub]);
system(['mkdir ', save_dir_sub]);
save_dir_gt = [save_dir,gttype,'/'];
system(['rm -rf ', save_dir_gt]);
system(['mkdir ', save_dir_gt]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if save_resized_ori_img == 1
    resized_img_path = sprintf('%s/%s/%s/',save_root, dataType, dataType_res);
    system(['rm -rf ', resized_img_path]);
    system(['mkdir ', resized_img_path]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% label files create
labelfile_dir = sprintf('%s%s/%s_%s.csv', save_dir, labelfile_path, dataType, transtype);
system(['rm -f',labelfile_dir]);
fid= fopen(labelfile_dir,'wt');
fprintf(fid,'combine,gt\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

annFile=sprintf('%s/annotations/%s_%s.json',dataDir,prefix,dataType);
cocoGt=CocoApi(annFile);

imgIds=sort(cocoGt.getImgIds());
[m_i n] = size(imgIds);
for idx = 1 : m_i
    imgId1 = imgIds(idx);
    img1 = cocoGt.loadImgs(imgId1);
    I = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img1.file_name));
    [h1, w1, c1] = size(I);
    if c1~=3
        continue;
    end
    annIds = cocoGt.getAnnIds('imgIds',imgId1);
    [annIdsa,annIdsb]=size(annIds);
    if annIdsa == 0
        disp(['Image --- ',num2str(idx),' no segmentation!']);
        continue;
    end
    % resize image
    I_r = imresize(I,[new_size,new_size]);
    %%%%%%%%%%%%%%%%%%%%%%%
    % save resized original image
    if save_resized_ori_img == 1
        imwrite(I_r, [resized_img_path, img1.file_name]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%
    anns = cocoGt.loadAnns(annIds);
    bimasks = masksgeneration(I, anns);
    instance_num = length(bimasks);
    instance_idx = randi(instance_num, 1);
    mask_ori_img = bimasks{instance_idx,1};
    mask_ori_img_r = imresize(mask_ori_img,[new_size,new_size]);
    prop_rate = sum(sum(mask_ori_img_r))/(new_size*new_size);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get the instance larger than prop_low
    if prop_rate < prop_low || prop_rate > prop_up
        if instance_num == 1
            continue
        else
            itr_count = 0;
            while prop_rate < prop_low || prop_rate > prop_up
                instance_idx = randi(instance_num, 1);
                mask_ori_img = bimasks{instance_idx,1};
                mask_ori_img_r = imresize(mask_ori_img,[new_size,new_size]);
                prop_rate = sum(sum(mask_ori_img_r))/(new_size*new_size);
                
                itr_count=itr_count + 1;
                
                if itr_count > 100
                    break;
                end
            end
            if itr_count > 100
                disp([num2str(idx),' No suitable instance!']);
                continue;
            end
        end 
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    tmp_mask = cat(3,mask_ori_img_r,mask_ori_img_r,mask_ori_img_r);
    seg_ori_img_r=I_r.*tmp_mask;
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % any operation on the mask
    %% Shift
    
    seg_paste_ = seg_ori_img_r;         % seg image resize
    mask_paste_ = mask_ori_img_r;
    
    prop_rate = 0;
    count = 0;
    while prop_rate < prop_low || prop_rate > prop_up
        
        R = randi([shift_down,shift_up],2,1);        
        seg_paste = imtranslate(seg_paste_,[R(1),R(2)]);
        mask_paste = imtranslate(mask_paste_,[R(1),R(2)]);
        
        prop_rate = sum(sum(mask_paste))/(new_size*new_size);
        
        count = count + 1;
        if count > max_count
            break;
        end
    end
    if count > max_count
        disp(['Shift Error -- ',num2str(idx)]);
        continue;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% Rotation
    count = 0;
    Trans_flag = rand(1);
    % 50% chance to operate rotation
    if Trans_flag > 0.5
        prop_rate = 0;
        while prop_rate < prop_low || prop_rate > prop_up
            R = randi([rotate_down,rotate_up],1);
            
            seg_paste = imrotate(seg_paste_, R(1), 'bilinear', 'crop');
            mask_paste = imrotate(mask_paste_, R(1), 'bilinear', 'crop');
            prop_rate = sum(sum(mask_paste))/(new_size*new_size);
            
            count = count + 1;
            if count > max_count
                break;
            end
        end
    end
    if count > max_count
        disp(['Rotation Error -- ',num2str(idx)]);
        continue;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% luminance
    
    Trans_flag = rand(1);
    % 50% chance to operate luminance change
    if Trans_flag > 0.5
        R = randi([lumin_down,lumin_up],1,1);
        seg_paste = seg_paste_+R(1);
        mask_paste = mask_paste_;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% scale
    Trans_flag = rand(1);
    % 50% chance to operate scale
    count = 0;
    if Trans_flag > 0.5
        stop_flag = 1;
        prop_rate = 0;
        while prop_rate < prop_low || prop_rate > prop_up || stop_flag > 0
            
            R_seed = rand(1);
            R = (scale_up - scale_down) * R_seed + scale_down;
            
            seg_paste = imresize(seg_paste_, R);
            mask_paste = imresize(mask_paste_, R);
            [m,n,c]=size(seg_paste);
            
            if m < new_size
                new_seg_paste = zeros(new_size,new_size,3,'uint8');
                new_mask_paste = zeros(new_size,new_size,'uint8');
                i_idx = floor((new_size - m)/2);
                new_seg_paste((i_idx+1):(i_idx+m),(i_idx+1):(i_idx+n),:)=seg_paste;
                new_mask_paste((i_idx+1):(i_idx+m),(i_idx+1):(i_idx+n))=mask_paste;
            
            elseif m > new_size
                i_idx = floor((m - new_size)/2);
                new_seg_paste = seg_paste((i_idx+1):(i_idx+new_size),(i_idx+1):(i_idx+new_size),:);
                new_mask_paste = mask_paste((i_idx+1):(i_idx+new_size),(i_idx+1):(i_idx+new_size));
            end
            
            stop_flag = sum(sum(mask_paste)) - sum(sum(new_mask_paste));
            
            seg_paste = new_seg_paste;
            mask_paste = new_mask_paste;
            prop_rate = sum(sum(mask_paste))/(new_size*new_size);

            count = count + 1;
            if count > max_count
                break;
            end
        end
    end
    
    if count > max_count
        disp(['Scale Error -- ',num2str(idx)]);
        continue;
    end
    
    seg_paste_ = seg_paste;
    mask_paste_ = mask_paste;
    
    %% deformation
    %Trans_flag = rand(1);
    % 50% chance to operate deformation
    
    Trans_flag = 0;
    count = 0;
    
    if Trans_flag > 0.5
        stop_flag = 1;
        prop_rate = 0;
        while prop_rate < prop_low || prop_rate > prop_up || stop_flag > 0
            
            R_seed = rand(1);
            R = (deform_up - deform_down) * R_seed + deform_down;
            deform_size = floor(R * new_size);
            dim_flag = rand(1);
            
            if dim_flag < 0.5
                seg_paste = imresize(seg_paste_,[deform_size,new_size]);
                mask_paste = imresize(mask_paste_,[deform_size,new_size]);
                
                if deform_size < new_size
                    new_seg_paste = zeros(new_size,new_size,3,'uint8');
                    new_mask_paste = zeros(new_size,new_size,'uint8');
                    i_idx = floor((new_size - deform_size)/2);
                    new_seg_paste((i_idx+1):(i_idx+deform_size), :,:)=seg_paste;
                    new_mask_paste((i_idx+1):(i_idx+deform_size),:)=mask_paste;
                elseif deform_size > new_size
                    i_idx = floor((deform_size - new_size)/2);
                    new_seg_paste = seg_paste((i_idx+1):(i_idx+new_size), :, :);
                    new_mask_paste = mask_paste((i_idx+1):(i_idx+new_size),:);
                end
            
            else
                seg_paste = imresize(seg_paste_,[new_size, deform_size]);
                mask_paste = imresize(mask_paste_,[new_size, deform_size]);
                
                if deform_size < new_size
                    new_seg_paste = zeros(new_size,new_size,3,'uint8');
                    new_mask_paste = zeros(new_size,new_size,'uint8');
                    i_idx = floor((new_size - deform_size)/2);
                    new_seg_paste(:, (i_idx+1):(i_idx+deform_size),:)=seg_paste;
                    new_mask_paste(:, (i_idx+1):(i_idx+deform_size))=mask_paste;
            
                elseif deform_size > new_size
                    i_idx = floor((deform_size - new_size)/2);
                    new_seg_paste = seg_paste(:, (i_idx+1):(i_idx+new_size), :);
                    new_mask_paste = mask_paste(:, (i_idx+1):(i_idx+new_size));
                end
                
            end
            
            stop_flag = sum(sum(mask_paste)) - sum(sum(new_mask_paste));
            
            seg_paste = new_seg_paste;
            mask_paste = new_mask_paste;
            prop_rate = sum(sum(mask_paste))/(new_size*new_size);

            count = count + 1;
            if count > max_count
                break;
            end
        end
    end
    
    if count > max_count
        disp(['deformable Error -- ',num2str(idx)]);
        continue;
    end
   
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    % get the second image
    c2 = 0;
    while c2 ~= 3
        id2 = randi(m_i, 1);
        while id2 == idx
            id2 = randi(m_i, 1);
        end
        imgId2 = imgIds(id2);
        img2 = cocoGt.loadImgs(imgId2);
        I2 = imread(sprintf('%s/images/%s/%s',dataDir,dataType,img2.file_name));
        [h2, w2, c2] = size(I2);
    end
    I2_r = imresize(I2,[new_size,new_size]);
    
    mask_paste_opp = 1 - mask_paste;
    mask_paste_opp = cat(3,mask_paste_opp,mask_paste_opp,mask_paste_opp);
    composite_img = I2_r.*mask_paste_opp + seg_paste;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% The foreground label file 
    %% save label file
    %% the instance generation
    image1_len = length(img1.file_name);
    image2_len = length(img2.file_name);
    
    composed_img_name = [img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'.jpg'];
    gt_name = [img1.file_name(1:(image1_len-4)),...
        '_', img1.file_name(1:(image1_len-4)),... 
        '_', num2str(instance_idx), '_',...
        img2.file_name(1:(image2_len-4)),...
        '_',transtype,'.png'];
    
    %% the label file generation
    labelfile_image = [transtype, '/',...
        composed_img_name];   
    labelfile_gt = [gttype,'/',gt_name];

    fprintf(fid,[labelfile_image, ',',labelfile_gt,'\n']);
    
    
    %% the composited image and groundtruths
    mask_ori_img_r(mask_ori_img_r==1)=255;
    mask_paste(mask_paste==1)=255;
    imwrite(composite_img, [save_dir_sub, composed_img_name]);
    imwrite(mask_paste, [save_dir_gt, gt_name]);
    
    disp(['Proccessed image -- ',num2str(idx)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
