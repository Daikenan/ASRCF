function [w]=ADMM_solve_w(params,use_sz,model_w,h_f)
    
    w = gpuArray(params.w_init*single(ones(use_sz)));
    q = w;
    m = w;
    
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    params.admm_iterations=2;
    T = prod(use_sz);
    h=T*real(ifft2(h_f));
%     hw=bsxfun(@times,h,model_w);
    hw=h;
    Hh=sum(hw.^2,3);
%     Hh=sum(real(bsxfun(@times,h_f(:),conj(h_f(:)))));
    
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for w- please refer to the paper for more details
      %  w = (q-m)/(1+(params.admm_lambda1/mu)*Hh);
        w = bsxfun(@rdivide,(q-m),(1+(params.admm_lambda1/mu)*Hh));
        %   solve for q
        q=(params.admm_lambda2*model_w + mu*(w+m))/(params.admm_lambda2 + mu);
        
        %   update m
        m = m + (w - q);
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
               
    end
    


end