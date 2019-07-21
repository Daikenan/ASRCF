function featuremap=extract_features(patch,use_sz,hogfeatures,hogparams,frame,ifcompress,pe)

   x_vgg16= get_vggfeatures(patch,use_sz,23);
   x_vggm=get_vggmfeatures(patch,use_sz,4);
   x_hc=get_features(patch,hogfeatures,hogparams);
   x_hc=gpuArray(x_hc);

   if ifcompress
       if frame==1
        x_vgg16=featPCA_init(x_vgg16,64,'vgg16');
        x_vggm=featPCA_init(x_vggm,16,'vggm');
       else
           x_vgg16=featPCA(x_vgg16,64,'vgg16');
           x_vggm=featPCA(x_vggm,16,'vggm');
       end
   end     
   x_vgg16=normalization(x_vgg16);  
   x_vggm=normalization(x_vggm);        

   featuremap={x_hc*pe(1),x_vgg16*pe(2),x_vggm*pe(3)};
end

function feat=normalization(x)
%          feat=bsxfun(@times,x,(size(x,1)*size(x,2)*size(x,3)./...
%             (sum(abs(reshape(x, [], 1, 1, size(x,4))).^2, 1) + eps)).^(1/2));
        feat=(x-min(x(:)))/(max(x(:))-min(x(:)));
end

function feat=featPCA_init(feat,num_channels,mode)

        [hf,wf,cf]=size(feat);
matrix=reshape(feat,hf*wf,cf);
global vgg16coeff
global vggmcoeff
global hccoeff
switch mode
    case 'vgg16'
        vgg16coeff = pca2(matrix);
        vgg16coeff=vgg16coeff(:,1:num_channels);
        feat_=reshape(feat,hf*wf,cf);
        feat_=feat_*vgg16coeff;
    case 'vggm'
        vggmcoeff = pca2(matrix);
        vggmcoeff=vggmcoeff(:,1:num_channels);
        feat_=reshape(feat,hf*wf,cf);
        feat_=feat_*vggmcoeff;
    case 'hc'
        hccoeff = pca2(matrix);
        hccoeff=hccoeff(:,1:num_channels);
        feat_=reshape(feat,hf*wf,cf);
        feat_=feat_*hccoeff;
end       
        feat=reshape(feat_,hf,wf,num_channels);

end
function feat=featPCA(feat,num_channels,mode)
    global vgg16coeff
    global vggmcoeff
    global hccoeff
    [hf,wf,cf]=size(feat);
    feat_=reshape(feat,hf*wf,cf);
    switch mode
        case 'vgg16'
            feat_=feat_*vgg16coeff;
        case 'vggm'
            feat_=feat_*vggmcoeff;
        case 'hc'
            feat_=feat_*hccoeff;
    end
    feat=reshape(feat_,hf,wf,num_channels);
end
function [coeff, score, latent, tsquared, explained, mu] = pca2(x,varargin)


[n, p] = size(x);

% Parse arguments and check if parameter/value pairs are valid 
paramNames = {'Algorithm','Centered','Economy','NumComponents','Rows',...
    'Weights','VariableWeights','Coeff0','Score0','Options'};
defaults   = {'svd',       true,      true,    p,           'complete',...
    ones(1,n) ,ones(1,p),        [],      [], statset('pca')};

[vAlgorithm, vCentered, vEconomy, vNumComponents, vRows,vWeights,...
    vVariableWeights, c0, s0, opts, setFlag]...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});
% Validate String value for  Algorithm value
AlgorithmNames = {'svd','eig','als'};
vAlgorithm = internal.stats.getParamVal(vAlgorithm,AlgorithmNames,...
    '''Algorithm''');
% Validate boolean value for 'Centered' option
vCentered = internal.stats.parseOnOff(vCentered,'''Centered''');
% Validate boolean value for 'Economy' option
vEconomy = internal.stats.parseOnOff(vEconomy,'''Economy''');
% Validate the number of components option 'NumComponents'
if ~isempty(x) && ~internal.stats.isScalarInt(vNumComponents,1,p)
    error(message('stats:pca:WrongNumComponents',p));
end
% Validate value for 'Rows' option
RowsNames = {'complete','pairwise','all'};
vRows = internal.stats.getParamVal(vRows,RowsNames,'''Rows''');

switch vAlgorithm
    case 'svd'
        % Switch 'Algorithm' to 'eig' if 'Rows' set to 'pairwise'
        if strcmp(vRows,'pairwise')
            if setFlag.Algorithm
                warning(message('stats:pca:NoPairwiseSVD'));
            end
            vAlgorithm = 'eig';
        end    
        % Switch algorithm to 'als' if user specify 'Coeff0' and 'Score0'.
        if setFlag.Coeff0 || setFlag.Score0
            vAlgorithm = 'als';
        end
    case 'als'
        % If 'als' is chosen, force PCA to use ALS and to ignore the
        % Rows' option
        if setFlag.Rows
            warning(message('stats:pca:NoALSRows'));
        end
end

% Validate Weights Vectors
if isvector(vWeights) && isequal(numel(vWeights),n)
    vWeights = reshape(vWeights,1,n); % make sure it is a row vector
else
    error(message('stats:pca:WrongObsWeights', n));
end

if internal.stats.isString(vVariableWeights)
    WeightsNames = {'variance'};
    internal.stats.getParamVal(vVariableWeights,WeightsNames,...
        '''VariableWeights''');
    vVariableWeights = 1./classreg.learning.internal.wnanvar(x,vWeights,1);
elseif isnumeric(vVariableWeights) && isvector(vVariableWeights)...
        && (isequal(numel(vVariableWeights), p))
    vVariableWeights = reshape(vVariableWeights,1,p);
else
    error(message('stats:pca:WrongVarWeights', p));
end

if any(vWeights <= 0) || any(vVariableWeights <= 0)
    error(message('stats:pca:NonPositiveWeights'));
end
% end of checking input arguments


% Handling special empty case
if isempty(x)
    pOrZero = ~vEconomy * p;
    coeff = zeros(p, pOrZero); 
    coeff(1:p+1:end) = 1;
    score = zeros(n, pOrZero);
    latent = zeros(pOrZero, 1);
    tsquared = zeros(n, 1);
    explained = [];
    mu = [];
    return;
end

nanIdx = isnan(x);
numNaN = sum(nanIdx, 2); % number of NaNs in each row
wasNaN = any(numNaN,2); % Rows that contain NaN

% Handling special cases where X is all NaNs:
if all(nanIdx(:))
    coeff = NaN;
    score = NaN;
    latent = NaN;
    tsquared = NaN;
    explained = NaN;
    mu = NaN;
    return;
end
% Handling special scalar case;
if isscalar(x)
    coeff = 1;
    score = (~vCentered)*x;
    latent = (~vCentered)*x^2;
    tsquared = ~vCentered;
    explained = 100;
    mu = vCentered*x;
    return;
end

if strcmp(vRows,'all') && (~strcmp(vAlgorithm,'als'))
    if any(wasNaN)
        error(message('stats:pca:RowsAll'));
    else
        vRows = 'complete';
    end
end

if strcmp(vRows,'complete')
    % Degrees of freedom (DOF) is n-1 if centered and n if not centered,
    % where n is the numer of rows without any NaN element.
    DOF = max(0,n-vCentered-sum(wasNaN));
elseif strcmp(vRows,'pairwise') 
    % DOF is the maximum number of element pairs without NaNs
    notNaN = double(~nanIdx);
    nanC = notNaN'*notNaN;
    nanC = nanC.*(~eye(p));
    DOF = max(nanC(:));
    DOF = DOF-vCentered;
else
    DOF = max(0,n-vCentered);
end

% Weighted sample mean:
mu = classreg.learning.internal.wnanmean(x, vWeights);

% Compute by EIG if no weights are given
switch vAlgorithm
case 'eig'
    % Center the data if 'Centered' is true.
    if vCentered
        x = bsxfun(@minus,x,mu);
    end
    
    % Use EIG to compute.
    [coeff, eigValues] = localEIG(x, vCentered, vRows, vWeights,...
        vVariableWeights);
    
    % When 'Economy' value is true, nothing corresponding to zero
    % eigenvalues should be returned.
    if (DOF<p)
        if vEconomy
            coeff(:, DOF+1:p) = [];
            eigValues(DOF+1:p, :) = [];
        else % make sure they are zeros.
            eigValues(DOF+1:p, :) = 0;
        end
    end
    
    % Check if eigvalues are all postive
    if any(eigValues<0)
        error(message('stats:pca:CovNotPositiveSemiDefinite'));
    end
    
    if nargout > 1
        score = x/coeff';
        latent = eigValues; % Output Eigenvalues
        if nargout > 3
            tsquared = localTSquared(score, latent, n, p);
        end
    end
    
case 'svd' % Use SVD to compute
    % Center the data if 'Centered' is true.
    if vCentered
        x = bsxfun(@minus,x,mu);
    end
    
    [U,sigma, coeff, wasNaN] = localSVD(x, n,...
        vEconomy, vWeights, vVariableWeights);
    if nargout > 1
        score =  bsxfun(@times,U,sigma');
        latent = sigma.^2./DOF;
        if nargout > 3
            tsquared = localTSquared(score,latent,DOF,p);
        end
        %Insert NaNs back
        if any(wasNaN)
            score = internal.stats.insertnan(wasNaN, score);
            if nargout >3
                tsquared = internal.stats.insertnan(wasNaN,tsquared);
            end
        end
    end
    
    if DOF < p
        % When 'Economy' value is true, nothing corresponding to zero
        % eigenvalues should be returned.
        if vEconomy
            coeff(:, DOF+1:end) = [];
            if nargout > 1
                score(:, DOF+1:end)=[];
                latent(DOF+1:end, :)=[];
            end
        elseif nargout > 1
        % otherwise, eigenvalues and corresponding outputs need to pad
        % zeros because svd(x,0) does not return columns of U corresponding
        % to components of (DOF+1):p.
            score(:, DOF+1:p) = 0;
            latent(DOF+1:p, 1) = 0;
        end
    end
    
case 'als' % Alternating Least Square Algorithm
    
    vNumComponents = min([vNumComponents,n-vCentered,p]);  % ALS always return economy sized outputs    
    
    if isempty(s0);
        s0 = randn(n,vNumComponents);
    elseif ~isequal(size(s0),[n,vNumComponents])|| any(isnan(s0(:)))
        error(message('stats:pca:BadInitialValues','Score0',n,vNumComponents));
    end
    if isempty(c0);
        c0 = randn(p,vNumComponents);
    elseif ~isequal(size(c0),[p,vNumComponents]) || any(isnan(c0(:)))
        error(message('stats:pca:BadInitialValues','Coeff0',p,vNumComponents));
    end
     
    [score,coeff,mu,latent]=alsmf(x,vNumComponents,'L0',s0,'R0',c0,...
        'Weights',vWeights,'VariableWeights',vVariableWeights,...
        'Orthonormal',true,'Centered',vCentered,'Options',opts);
    
    if nargout > 3   
        % T-squared values are in reduced space.
        tsquared = localTSquared(score, latent,n-vCentered, vNumComponents); 
    end    
end % end of switch vAlgorithm

% Calcuate the percentage of the total variance explained by each principal
% component.
if nargout > 4
    explained = 100*latent/sum(latent);
end

% Output only the first k principal components
if (vNumComponents<DOF)
    coeff(:, vNumComponents+1:end) = [];
    if nargout > 1
    score(:, vNumComponents+1:end) = [];
    end
end


% Enforce a sign convention on the coefficients -- the largest element in
% each column will have a positive sign.
[~,maxind] = max(abs(coeff), [], 1);
[d1, d2] = size(coeff);
colsign = sign(coeff(maxind + (0:d1:(d2-1)*d1)));
coeff = bsxfun(@times, coeff, colsign);
if nargout > 1
    score = bsxfun(@times, score, colsign); % scores = score
end

end % End of main function


%----------------Subfucntions--------------------------------------------

function [coeff, eigValues]=localEIG(x,vCentered, vRows,vWeights,...
        vVariableWeights)
% Compute by EIG. vRows are the options of handing NaN when compute
% covariance matrix

% Apply observation and variable weights
OmegaSqrt = sqrt(vWeights);
PhiSqrt = sqrt(vVariableWeights);
x = bsxfun(@times, x, OmegaSqrt');
x = bsxfun(@times, x, PhiSqrt);

xCov = ncnancov(x, vRows, vCentered);

[coeff, eigValueDiag] = eig(xCov);
[eigValues, idx] = sort(diag(eigValueDiag), 'descend');
coeff = coeff(:, idx);

coeff = bsxfun(@times, coeff,1./PhiSqrt');
end


function [U,sigma, coeff, wasNaN] = localSVD(x, n,...,
    vEconomy, vWeights, vVariableWeights)
% Compute by SVD. Weights are supplied by vWeights and vVariableWeights.

% Remove NaNs missing data and record location
[~,wasNaN,x] = internal.stats.removenan(x);
if n==1  % special case because internal.stats.removenan treats all vectors as columns
    wasNaN = wasNaN';
    x = x';
end

% Apply observation and variable weights
vWeights(wasNaN) = [];
OmegaSqrt = sqrt(vWeights);
PhiSqrt = sqrt(vVariableWeights);
x = bsxfun(@times, x, OmegaSqrt');
x = bsxfun(@times, x, PhiSqrt);
    
if vEconomy
    [U,sigma,coeff] = svd(x,'econ');
else
    [U,sigma, coeff] = svd(x, 0);
end

U = bsxfun(@times, U, 1./OmegaSqrt');
coeff = bsxfun(@times, coeff,1./PhiSqrt');

if n == 1     % sigma might have only 1 row
    sigma = sigma(1);
else
    sigma = diag(sigma);
end
end

function tsquared = localTSquared(score, latent,DOF, p)
% Subfunction to calulate the Hotelling's T-squared statistic. It is the
% sum of squares of the standardized scores, i.e., Mahalanobis distances.
% When X appears to have column rank < r, ignore components that are
% orthogonal to the data.

if isempty(score)
    tsquared = score;
    return;
end

r = min(DOF,p); % Max possible rank of x;
if DOF > 1
    q = sum(latent > max(DOF,p)*eps(latent(1)));
    if q < r
        warning(message('stats:pca:ColRankDefX', q)); 
    end
else
    q = 0;
end
standScores = bsxfun(@times, score(:,1:q), 1./sqrt(latent(1:q,:))');
tsquared = sum(standScores.^2, 2);
end

function c = ncnancov(x,Rows,centered)
%   C = NCNANCOV(X) returns X'*X/N, where N is the number of observations
%   after removing rows missing values. 
%
%   C = NCNANCOV(...,'pairwise') computes C(I,J) using rows with no NaN
%   values in columns I or J.  The result may not be a positive definite
%   matrix. C = NCNANCOV(...,'complete') is the default, and it omits rows
%   with any NaN values, even if they are not in column I or J.
%
%   C = NCNANCOV(...,true), C is normalized by N-1 if data X is already
%   centered. The default is false.

if nargin <2
    Rows = 'complete';
end

d = 0;
if nargin>2
    d =  d + centered;
end

idxnan = isnan(x);

[n, p] = size(x);


if ~any(any(idxnan))
    c = x'*x/(n-d);
elseif strcmp(Rows,'complete')
    nanrows = any(idxnan,2);
    xNotNaN = x((~nanrows),:);
    denom = max(0, (size(xNotNaN,1)-d) );
    c = xNotNaN'*xNotNaN/denom;
elseif strcmp(Rows,'pairwise')
    c = zeros(p,class(x));
    for i = 1:p
        for j = 1:i
            NonNaNRows = ~any(idxnan(:,[i, j]),2);
            denom = max(0,(sum(NonNaNRows)-d));
            c(i,j) = x(NonNaNRows,i)'*x(NonNaNRows,j)/denom;
        end
    end
    c = c + tril(c,-1)';
end
end
