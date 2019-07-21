function overlap=compute_overlap(results,ground_truth)
gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
 results.gt = gt_boxes;
    %   compute the OP
    pd_boxes = results.res;
    pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
    lenALL=size(ground_truth,1);
     OP = zeros(size(gt_boxes,1),1);
    for i=1:size(gt_boxes,1)
        b_gt = gt_boxes(i,:);
        b_pd = pd_boxes(i,:);
        OP(i) = computePascalScore(b_gt,b_pd);
    end
    thresholdSetOverlap = 0:0.05:1;
    successNumOverlap = zeros(length(thresholdSetOverlap));
     for tIdx=1:length(thresholdSetOverlap)
         successNumOverlap(idx,tIdx) = sum(OP >thresholdSetOverlap(tIdx));
     end
     aveSuccessRatePlot = successNumOverlap/(lenALL+eps);
     overlap=mean(aveSuccessRatePlot);
end