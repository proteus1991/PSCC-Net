function masks = masksgeneration(I, anns)
[h, w, ~] = size(I);
len = length(anns);
masks = cell(len, 1);
    for n=1:len
        poly = anns(n).segmentation;
        
        if isa(poly, 'cell')
            R = MaskApi.frPoly(poly, h, w);
            mask = MaskApi.decode(R);
        else
            poly = {poly.counts};
            R = MaskApi.frPoly(poly, h, w);
            mask  = MaskApi.decode(R);
        end
        masks{n, 1} = mask;
    end
end