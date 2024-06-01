function data = analytical_solution(N, a, f, init)

    L = 1; % length
    xgrid = linspace(0, L, N);
    dx = xgrid(2); % spatial sampling

    dt = f*(dx^2/(2*a)); % =>  1/2 > (dt*a)/dx^2 (see Colaco, p.180, eq. 5.138b) 5.139 b
    Tf = 1;

    [tt, xx] = meshgrid((0:dt:Tf), xgrid);

    %% 
    n_coeffs = 10000;
    sample = linspace(0,1,n_coeffs);
    init_val = init(sample);
    symm_init = [init_val flip(init_val(2:end-1))];     % symmetric extension
    L_ = 2*L;                                           
    f_coeffs = 1/length(symm_init)*real(fft(symm_init));
    f_coeffs(2:end) = 2*f_coeffs(2:end);

    data = zeros(size(xx));

    for s = 1:n_coeffs
        data = data + f_coeffs(s).*exp(-4*((s-1)/L_*pi)^2*tt*a).*cos(2*pi*(s-1)*xx/L_);
    end
end