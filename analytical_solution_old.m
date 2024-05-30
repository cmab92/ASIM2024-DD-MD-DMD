function data = analytical_solution_old(n_samples_x, k, r, c, f)
    a = k / (r*c); % diffusivity
    
    L = 1; % length
    N = n_samples_x; % spatial sampling points
    dx = L/(N-1); % spatial sampling
    xgrid = 0 : dx : L; % Position in x-direction
    
    dt = f*(dx^2/(2*a)); % => dt < dx^2/(2*a)
    Tf = 1;
    t_samp = dt;
    tgrid = 0 : t_samp : Tf;
    
    data = zeros(length(xgrid), length(tgrid));
    for i = 1:length(tgrid)
        td = tgrid(i);
        for j = 1:length(xgrid)
            xd = xgrid(j);
            c0 = L^2 / 6;
            c1 = L^2 / pi^2 ;
            s = 0;
            kmax = 5000;
            for k = 1:kmax
                s = s + exp(-4*a*(k*pi/L)^2*td)*cos(2*k*pi*xd/L) / k^2;
            end
            data(j, i) = c0 - c1*s;
        end
    end
end