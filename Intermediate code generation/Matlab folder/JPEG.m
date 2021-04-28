
% CS443-01 
% Project: JPEG Implementation
% Team 1 - Jordan Biffle, Keyara Coleman, Tyler Goodwyn
% Leonie Nutz, Nicholas Zwolinski




function o=JPEG(image)
    close all;
    %clear;
    clc;

    %% Read Image
    %I = imread('tulips.png');
    I = imread(image); % will be padded if necessary
    
    ogIMG = I; % will be used for error calculation

    %% set up the globals for tracking changes in the image
    global tracked_img; % will be used in tracking changing
    global stage_num;
    global img_string;
    tracked_img = I;
    f=split(image,".");
    img_string = f(1); 
    stage_num = 0;

    % figure(1);
    % subplot(3,3,1),imshow(I),title("Original image");
    
    %% Output and show first 8x8 block before step 1
    tracked_img = I;
    intermediate_output(); % stage 0
    
    %% pad the image to make pixels in multiples of 16
    [m,n,~]=size(I);
    pad_x = rem(m,16);
    pad_y = rem(n,16);
    if pad_x ~= 0
        pad_x = 16 - pad_x;
    end
    if pad_y ~= 0
        pad_y = 16 - pad_y;
    end
    I = padarray(I,[pad_x/2 pad_y/2 0],'replicate','both');

    %% convert image to YCbCr format
    I2 = rgb2ycbcr(I);

    % intermediate output
    tracked_img = I2;
    intermediate_output(); % end of stage 1

    %% Perform chroma subsampling 4:2:0 on color components Cb and Cr individually
    % Downsample from [m,n] to [m/2, n/2]
    % nY = I2(:,:,1);
    % nCb=downSample420(I2(:,:,2));
    % nCr=downSample420(I2(:,:,3));

    nY = I2(:,:,1);
    nCb=simpleSample420(I2(:,:,2));
    nCr=simpleSample420(I2(:,:,3));

    % intermediate output
    tracked_img = cat(3, nY, nCb, nCr);
    intermediate_output(); % end of stage 2


    %% Performing dct in blocks of 8x8
    % 8 is the size of the 8x8 block being DCTed.
    C = create_c_matrix(8);

    nnY = perform_dct(nY,C);
    nnCb = perform_dct(nCb,C);
    nnCr = perform_dct(nCr,C);
    
    % intermediate output
    tracked_img = cat(3, nnY, nnCb, nnCr);
    intermediate_output(); % end of stage 3

    %% Quantization matrices

    Y_Table=[16 11  10  16 24  40  51 61
            12  12  14 19  26  58 60  55
            14  13  16 24  40  57 69  56
            14  17  22 29  51  87 80  62
            18  22  37 56  68 109 103 77
            24  35  55 64  81 104 113 92
            49  64  78  87 103 121 120 101
            72  92  95  98 112 100 103  99];%Luminance quantization table

    CbCr_Table=[17, 18, 24, 47, 99, 99, 99, 99;
                18, 21, 26, 66, 99, 99, 99, 99;
                24, 26, 56, 99, 99, 99, 99, 99;
                47, 66, 99 ,99, 99, 99, 99, 99;
                99, 99, 99, 99, 99, 99, 99, 99;
                99, 99, 99, 99, 99, 99, 99, 99;
                99, 99, 99, 99, 99, 99, 99, 99;
                99, 99, 99, 99, 99, 99, 99, 99];%Color difference quantization table

    %% Performing quantization
    nnnY = perform_quantization(nnY,Y_Table);
    nnnCb = perform_quantization(nnCb,CbCr_Table);
    nnnCr = perform_quantization(nnCr,CbCr_Table);
    
    % intermediate output
    tracked_img = cat(3, nnnY, nnnCb, nnnCr);
    intermediate_output(); % end of stage 4

    %% Now performing all steps in reverse
    % Performing Inverse quantization

    new_nnnY = perform_inverse_quantization(nnnY,Y_Table);
    new_nnnCb = perform_inverse_quantization(nnnCb,CbCr_Table);
    new_nnnCr = perform_inverse_quantization(nnnCr,CbCr_Table);

     % intermediate output
     tracked_img = cat(3, new_nnnY, new_nnnCb, new_nnnCr);
     intermediate_output(); % end of stage 5

    %% Performing Inverse DCT

    new_nnY = perform_inverse_dct(new_nnnY,C);
    new_nnCb = perform_inverse_dct(new_nnnCb,C);
    new_nnCr = perform_inverse_dct(new_nnnCr,C);
    
     % intermediate output
     tracked_img = cat(3, new_nnY, new_nnCb, new_nnCr);
     intermediate_output(); % end of stage 6

    %% Upsample from [m/2,n/2] to [m, n]

    new_nY = new_nnY;
    new_nCb = new_nnCb;
    new_nCr = new_nnCr;


    % intermediate output
    tracked_img = cat(3,new_nY,new_nCb,new_nCr);
    intermediate_output(); % end of stage 7
    
    %% Concatenating, and reconverting to RGB
    
    final_im = cat(3,new_nY,new_nCb,new_nCr);
    final_im = ycbcr2rgb(final_im); 
    

    
    %% Removing padding

    final_im = final_im((pad_x/2)+1:(pad_x/2)+m , (pad_y/2)+1:(pad_y/2)+n , :);

    
    %imwrite(final_im,'tulips_new.png');
    imwrite(final_im,append("3_ Outputs/",img_string," result.png"));
    
    o=final_im;
    %% Error calculations
    % this function returns the PSNR, displays the error map
    psnr = calculate_errors(ogIMG, final_im);
    % disp("PSNR = "+psnr)


    %% FUNCTIONS
    function intermediate_output() 
        eight_block=tracked_img(1:8,1:8,1);
        stage_num = int2str(stage_num);
        writematrix(eight_block,append("2_ Intermediate results/",img_string," stage ",stage_num," 8x8 values",".txt"));
        imwrite(eight_block,append("2_ Intermediate results/",img_string," stage ",stage_num," 8x8 image",".png"));
        imwrite(tracked_img,append("2_ Intermediate results/",img_string," stage ",stage_num," full image",".png"));
        stage_num = uint8(str2double(stage_num));
        stage_num = stage_num +1;
    end
    function C = create_c_matrix(N)
        % N is the size of the NxN block being DCTed.
        % Create C
        C = zeros(N,N);
        for mm = 0:1:N-1
            for nn = 0:1:N-1
                if nn == 0
                k = sqrt(1/N);
                else
                k = sqrt(2/N);
                end
            C(mm+1,nn+1) = k*cos( ((2*mm+1)*nn*pi) / (2*N));
            end
        end
    end

    function [new_Im]=simpleSample420(I)
        I = double(I);
        [row,col] = size(I);
        % new_Im = zeros(row,col);
        new_Im = I;
        for i = 1:row
            for j = 1:col
                if mod(i,2) == 1 % odd rows
                    % even cols
                    if mod(j,2) == 0
                        new_Im(i,j) = I(i,j-1);
                    end
                else % if an odd row, grab value from element above
                    new_Im(i,j) = I(i-1,j);
                end 
            end
        end 
    end

    function [new_Im]=perform_dct(I,C)

        % function for performing dct
        dct = @(block_struct) C * block_struct.data * C';

        I=double(I) - 128;  %Level shift128Gray levels

        % performing dct in blocks of 8x8,
        new_Im = blockproc(I,[8 8],dct);

    end
    function [new_Im]=perform_inverse_dct(I,C)

        % function for performing inverse dct
        invdct = @(block_struct) C' * block_struct.data * C;

        % performing inverse dct in blocks of 8x8,
        % then rounding result, then converting to uint8
        new_Im = round(blockproc(I,[8 8],invdct));

        % reversing gray level shift
        new_Im = uint8(new_Im + 128);

        % limiting out of range pixel values
        [row,column]=size(new_Im);
        for i=1:row
            for j=1:column
                if new_Im(i,j)>255
                    new_Im(i,j)=255;
                elseif new_Im(i,j)<0
                    new_Im(i,j)=0;
                end
            end
        end

    end
    function [new_Im]=perform_inverse_quantization(I,quan_mat)

        % function for performing quantization
        qua_func = @(block_struct) block_struct.data .* quan_mat;

        new_Im = blockproc(I,[8 8],qua_func);

    end

    function [new_Im]=perform_quantization(I,quan_mat)

        % function for performing quantization
        qua_func = @(block_struct) round(block_struct.data ./ quan_mat);

        new_Im = blockproc(I,[8 8],qua_func);

    end

    function [psnr] = calculate_errors(ogIMG, finalIMG)
        ogIMG = rgb2gray(ogIMG);
        finalIMG = rgb2gray(finalIMG);
        diffIMG = imabsdiff(ogIMG,finalIMG);
        mse = immse(finalIMG, ogIMG);
        psnr = 20 * log10(255/sqrt(mse));
        imwrite(diffIMG,append("4_ Error results/",img_string," pixelwise_error_map.png"));
        fid = fopen(append("4_ Error results/",img_string," PSNR value.txt"), 'w+');
        fprintf(fid, 'PSNR = %.3f',psnr);
        fclose(fid);
    end
end

