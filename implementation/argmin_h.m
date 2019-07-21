function h=argmin_h(T,mu,lambda,g_f,l_f,w0)

     lhd= T ./  (lambda*w0 .^2 + mu*T); % left hand
  
     X=ifft2(mu*g_f + l_f);
%      h=gpuArray(zeros(size(X)));
     % compute T for each channel

     h=bsxfun(@times,lhd,X);

end