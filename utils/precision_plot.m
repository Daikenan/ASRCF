function precisions = precision_plot(positions, ground_truth, video_name, show)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	
	max_threshold = 50;  %used for graphs in the paper
	
	
	precisions = zeros(max_threshold, 1);
	
	if size(positions,1) ~= size(ground_truth,1)
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
		%just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames
    gt_center = [ground_truth(:,1)+(ground_truth(:,3)-1)/2 ground_truth(:,2)+(ground_truth(:,4)-1)/2];
    positions_center = [positions(:,1)+(positions(:,3)-1)/2 positions(:,2)+(positions(:,4)-1)/2];

    distances = sqrt(sum((positions_center-gt_center).^2,2));

    index = ground_truth>0;
    ind = (sum(index,2)==4);

    distances(~ind) = -1;   

    %compute precisions
    for p = 1:max_threshold
        precisions(p) = nnz(distances <= p) / numel(distances);
    end
    
    %get annotation of plot
    score = precisions(20);
    tmp = sprintf('%.3f', score);
    tmpName = ['TrackerName' ' [' tmp ']'];
    titleName = 'Precision plots of error';
	
	%plot the precisions
	if show == 1
		figure('NumberTitle','off', 'Name',['Precisions - ' video_name])
		plot(precisions, 'r-', 'LineWidth',2)
        legend1 = legend(tmpName);
		xlabel('Threshold'), ylabel('Precision')
        title(titleName) 
        saveas(gca,'PrecisionPlot','jpg')
	end
	
end

