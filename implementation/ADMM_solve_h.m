function [g_f,h_f]=ADMM_solve_h(params,use_sz,model_xf,yf,small_filter_sz,w,model_w,frame)
   
     g_f = gpuArray(single(zeros(size(model_xf))));
     h_f = g_f;
     l_f = g_f;
    
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
%     ws=bsxfun(@times,w,model_w);
    ws=w;
    T = prod(use_sz);
    S_xx = sum(conj(model_xf) .* model_xf, 3);
     if frame <=params.admm_3frame
         params.admm_iterations=3;end
    %   ADMM
   
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = S_xx + (T * mu);
        S_lx = sum(conj(model_xf) .* l_f, 3);
        S_hx = sum(conj(model_xf) .* h_f, 3);
        g_f = 1/(T*mu) * bsxfun(@times, yf, model_xf) - (1/mu) * l_f + h_f - ...
            (bsxfun(@rdivide,(1/(T*mu) * bsxfun(@times, model_xf, (S_xx .* yf)) - (1/mu) * bsxfun(@times, model_xf, S_lx) + bsxfun(@times, model_xf, S_hx)), B));
  
        %   solve for H
  %      h = (T/(mu*T+ params.admm_lambda))* ifft2(mu*g_f + l_f);
        h=argmin_h(T,mu,params.admm_lambda1,g_f,l_f,ws);

        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = gpuArray(single(zeros(use_sz(1), use_sz(2), size(h,3))));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
               
    end
   