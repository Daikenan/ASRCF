if frame == 1
    bias = min(w(:));
end
if (frame==1||mod(frame,update_interval)==0)&&params.show_regularization
    figure(2);
    I = im;
    init_rect = rect_position(loop_frame,:);
    I = insertShape(I, 'Rectangle', init_rect, 'LineWidth', 2, 'Color', 'red');
    padding = 2.0;
    rect_paddding = [init_rect(1)-init_rect(3)*padding/2,...
    init_rect(2)-init_rect(4)*padding/2,...
    init_rect(3)*(1+padding),init_rect(4)*(1+padding)];
    crop = imresize(imcrop(I, rect_paddding), [125, 125]);
    [h, width, c] = size(crop);

    startw=round(size(w,1)*0.4/2)+1;
    starth=round(size(w,2)*0.4/2)+1;
    w_=w(startw:startw+round(size(w,1)*0.6)-1,starth:starth+round(size(w,2)*0.6)-1);
    w_=imresize(w_,[h,width]);
   
    w_=(w_-bias+0.1);
    surf(w_);
    %surf(X,Y,(Z-min(Z(:))),'FaceAlpha',0.4);colormap(hsv);hold on

    g = hgtransform('Matrix',makehgtform('translate',[0 0 0]));
    image(g, crop)

    axis off
    view(205, 30)
    set(gcf, 'position', [100 100 900 900], 'Color',[0,0,0]);  
end   