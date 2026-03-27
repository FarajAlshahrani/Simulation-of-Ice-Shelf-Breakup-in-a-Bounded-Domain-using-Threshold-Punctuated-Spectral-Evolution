%% ================================================================================================================
%  Master Script: Simulation of Ice-Shelf Breakup in a Bounded Domain using Threshold-Punctuated Spectral Evolution
%  Author: Faraj Alshahrani  |  The University of Newcastle, Australia
%  ================================================================================================================
clc;
close all;
base_path = '/Users/f16/Library/CloudStorage/OneDrive-TheUniversityOfNewcastle/JFM Paper2/';
movies_dir = fullfile(base_path,'Movies');
paths = {movies_dir};
cellfun(@(p) exist(p,'dir') || mkdir(p), paths);

flag_values = [1 2];
N_crack = 50;
h_value = [30,50,80,100,130,150,200,250];   
run_id = 0;

% ============================
% SNAPSHOT SETTINGS
% ============================
letters = 'abcdefghijklmnopqrstuvwxyz';
snapshots_dir = fullfile(base_path, 'Figures');
if ~exist(snapshots_dir,'dir')
    mkdir(snapshots_dir);
end

%fixed figure-number map for requested paper order =====
% ic_idx = 1, flag = 1  -> fig2  to fig9
% ic_idx = 1, flag = 2  -> fig10 to fig17
% ic_idx = 2, flag = 1  -> fig20 to fig27  
% ic_idx = 2, flag = 2  -> fig28 to fig35   
fig_map_main = [ ...
     2,  3,  4,  5,  6,  7,  8,  9;    % row 1: ic_idx = 1, flag = 1
    10, 11, 12, 13, 14, 15, 16, 17;    % row 2: ic_idx = 1, flag = 2
    20, 21, 22, 23, 24, 25, 26, 27;    % row 3: ic_idx = 2, flag = 1
    28, 29, 30, 31, 32, 33, 34, 35];   % row 4: ic_idx = 2, flag = 2

% helper to select correct row =====
get_map_row = @(ic_idx, flag_initial) ...
    (ic_idx == 1 && flag_initial == 1) * 1 + ...
    (ic_idx == 1 && flag_initial == 2) * 2 + ...
    (ic_idx == 2 && flag_initial == 1) * 3 + ...
    (ic_idx == 2 && flag_initial == 2) * 4;

% store first breakup time for each h, flag, ic_idx =====
breakup_times_all = nan(2,2,length(h_value),6);   
for flag_idx = 1:length(flag_values)
    flag_initial = flag_values(flag_idx);

    for ic_idx = 1:2  
        for idx = 1:length(h_value)

            flag = flag_initial;
            h = h_value(idx);

            H_1 = 300;
            L = [0,1e4];
            E = 11e9;
            p_i = 922.5;
            p_w = 1025;
            H_2 = H_1 - ((p_i/p_w) * h);
            omega = linspace(0,1,5e3);

            N = 10001;
            x = linspace(-L(end), L(end), N);
            dx = x(2)-x(1);
            x = x(1:end-1) + dx/2;
            eps_cr = 1e-5;
            a = 2*eps_cr/h;  
            nu = 0.33;
            g = 9.81;
            D = (E * h^3) / (12 * (1 - nu^2));

            % ============================
            % INITIAL MODES
            % ============================
            if (flag == 1) || (flag == 3)
                Eigenvalues = compute_eigenvalues_values_CC_F3(H_1,H_2,h,L,E,omega);
                NM = length(Eigenvalues);
                Eigenvalues = Eigenvalues(1:NM);
                Eigenvalues(1) = 0;
                [phi, w, w2, w3, w4] = All_fun_CC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_CC(L,x,p_w,g,D);

            elseif flag == 2
                Eigenvalues = compute_eigenvalues_values_SSC_F3(H_1,H_2,h,L,E,omega);
                NM = length(Eigenvalues);
                Eigenvalues(2:NM+1) = Eigenvalues(1:NM);
                Eigenvalues(1) = 0;
                Eigenvalues = Eigenvalues(1:NM);
                [phi, w, w2, w3, w4] = All_fun_SSC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_SSC(L,x,p_w,g,D);

            else
                Eigenvalues = compute_eigenvalues_values_FEC_F3(H_1,H_2,h,L,E,omega);
                NM = length(Eigenvalues);
                Eigenvalues = Eigenvalues(1:NM);
                Eigenvalues(1) = 0;
                [phi, w, w2, w3, w4] = All_fun_FEC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_FEC(L,x,p_w,g,D);
            end

            phi(:,1) = phi_0;
            w(:,1)   = w_0;
            w2(:,1)  = w2_0;
            w3(:,1)  = w3_0;
            w4(:,1)  = w4_0;

            for k = 1:NM
                Inner_product = sum(p_w*g*w(:,k).*conj(w(:,k))*dx) + ...
                                sum(D*w(x>=0,k).*conj(w4(x>=0,k))*dx);
                scale = 1/sqrt(Inner_product);
                w(:,k)  = w(:,k)*scale;
                w2(:,k) = w2(:,k)*scale;
                w3(:,k) = w3(:,k)*scale;
                w4(:,k) = w4(:,k)*scale;
            end

            A = zeros(NM,NM);
            for iA = 1:NM
                for jA = 1:NM
                    A(iA,jA) = sum(p_w * g * w(:,iA).*conj(w(:,jA)) * dx) + ...
                               sum(D * w(x>=0,iA).*conj(w4(x>=0,jA)) * dx);
                end
            end

            % ============================
            % INITIAL CONDITION
            % ============================
            % ic_idx=1 and ic_idx=2 now use the SAME time vector =====
            if ic_idx == 1
                f_x = exp(-1e-6*(x + L(end)/2).^2).';
                c = sqrt(g*H_1);
                dfdx = -2e-6*(x + L(end)/2).*exp(-1e-6*(x + L(end)/2).^2);
                dfdx = dfdx.';
                t = linspace(0.00001,500,5001);
                g_x = -c * dfdx;
            else
                f_x = exp(-1e-6*(x - L(end)/2).^2).';
                dfdx = -2e-6*(x - L(end)/2).*exp(-1e-6*(x - L(end)/2).^2);
                dfdx = dfdx.';
                t = linspace(0.00001,500,5001);  
                g_x = zeros(length(x),1);
            end

            C_n = zeros(NM,1);
            D_n = zeros(NM,1);
            for j = 1:NM
                C_n(j) = sum(p_w*g*f_x.*conj(w(:,j))*dx) + ...
                         sum(D*f_x(x>=0).*conj(w4(x>=0,j))*dx);
                D_n(j) = sum(p_w*g*g_x.*conj(w(:,j))*dx) + ...
                         sum(D*g_x(x>=0).*conj(w4(x>=0,j))*dx);
            end

            fig = figure('Units','normalized','Position',[0.1 0.1 0.8 0.6]);
            set(fig,'PaperPositionMode','auto');
            run_id = run_id + 1;

            movieObj = VideoWriter(fullfile(movies_dir, sprintf('Movie%d.mp4', run_id)), 'MPEG-4');
            movieObj.FrameRate = 15;
            open(movieObj);

            % use fixed requested figure numbering =====
            map_row = get_map_row(ic_idx, flag_initial);
            figNum = fig_map_main(map_row, idx);

            xi = nan(1,N_crack+1);
            ti = nan(1,N_crack+1);
            xi(1) = 0;
            ti(1) = 0;

            % fig2-fig19 all use (a-f) with t1=0 =====
            nPanelsToSave = 6;
            targetTimes = nan(1,nPanelsToSave);
            targetTimes(1) = 0;         
            savedTargets = false(1,nPanelsToSave);

            x_local = x(x>=0);

            if run_id == 1
                En = cell(N_crack,1);
                E0 = cell(N_crack,1);
                En_dyn = cell(N_crack,1);
            end

            stop_movie = false;

            for n_crack = 1:N_crack
                for ii = 1:length(t)

                    if n_crack > N_crack || isnan(ti(n_crack))
                        break
                    end

                    if t(ii) <= ti(n_crack)
                        continue
                    end

                    tau = t(ii) - ti(n_crack);
                    if tau < 0
                        tau = 0;
                    end

                    % =====================================================
                    % PRE-BREAK STATE at current global time t(ii)
                    % =====================================================
                    W_xx_old = (C_n(1)+(D_n(1)*tau))*w2(:,1) + ...
                               w2(:,2:NM) * (diag(cos(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                               diag(sin(Eigenvalues(2:NM)*tau)./Eigenvalues(2:NM))*D_n(2:NM));

                    W_x_t_old = (C_n(1)+(D_n(1)*tau))*w(:,1) + ...
                                w(:,2:NM) * (diag(cos(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                                diag(sin(Eigenvalues(2:NM)*tau)./Eigenvalues(2:NM))*D_n(2:NM));

                    W_tt_x_old = (D_n(1)*w(:,1)) + ...
                                 w(:,2:NM)*(diag(-Eigenvalues(2:NM).*sin(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                                 diag(cos(Eigenvalues(2:NM)*tau))*D_n(2:NM));

                    if run_id == 1
                        E0 = abs(C_n(1)).^2 + abs(D_n(1)).^2;
                        En_dyn = abs(C_n(2:NM)).^2 + (abs(D_n(2:NM)).^2 ./ (abs(Eigenvalues(2:NM)).^2));
                        En{n_crack} = [E0 ; En_dyn];
                    end

                    % =====================================================
                    % DETECT THRESHOLD ON THE RIGHT PANEL FIRST
                    % =====================================================
                    [val, br] = max(abs(real(W_xx_old(x>=0))));
                    breakup_now = false;
                    snapIdx = NaN;

                    W_x_t_plot = W_x_t_old;
                    W_xx_plot  = W_xx_old;

                    if (flag == 1) && x_local(br) > (L(end) - dx)
                        xi(n_crack+1) = L(end);
                        flag = 3;
                        disp('Shifted from Clamped boundary conditions to Free Edge Conditions')
                    end

                    % =====================================================
                    % SAME LOOP LOGIC
                    % Right panel = pre-break threshold exceedance
                    % Left panel  = post-break state in same loop
                    % =====================================================
                    if (abs(val) > a) && (n_crack < N_crack) && (x_local(br) < L(end) - dx)

                        breakup_now = true;

                        xi(n_crack+1) = x_local(br);
                        ti(n_crack+1) = t(ii);

                        % store breakup times t_1,...,t_6 =====
                        if (n_crack >= 1) && (n_crack <= 6)
                            breakup_times_all(ic_idx,flag_idx,idx,n_crack) = t(ii);
                        end

                        % save a-f as t1,...,t6 with t1=0 and first real breakup = t2 =====
                        if (n_crack+1) <= nPanelsToSave
                            targetTimes(n_crack+1) = t(ii);
                            snapIdx = n_crack + 1;
                        end

                        f_x = W_x_t_old;
                        g_x = W_tt_x_old;

                        L = [sort(xi(1:n_crack+1)), L(end)];
                        L = uniquetol(L);

                        if (flag == 1) || (flag == 3)
                            if flag == 1
                                Eigenvalues = compute_eigenvalues_values_CC_F3(H_1,H_2,h,L,E,omega);
                                NM = length(Eigenvalues);
                                Eigenvalues = Eigenvalues(1:NM);
                                Eigenvalues(1) = 0;
                                [phi, w, w2, w3, w4] = All_fun_CC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_CC(L,x,p_w,g,D);
                            else
                                Eigenvalues = compute_eigenvalues_values_FEC_F3(H_1,H_2,h,L,E,omega);
                                NM = length(Eigenvalues);
                                Eigenvalues = Eigenvalues(1:NM);
                                Eigenvalues(1) = 0;
                                [phi, w, w2, w3, w4] = All_fun_FEC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_FEC(L,x,p_w,g,D);
                            end

                        elseif flag == 2
                            Eigenvalues = compute_eigenvalues_values_SSC_F3(H_1,H_2,h,L,E,omega);
                            NM = length(Eigenvalues);
                            Eigenvalues(2:NM+1) = Eigenvalues(1:NM);
                            Eigenvalues(1) = 0;
                            Eigenvalues = Eigenvalues(1:NM);
                            [phi, w, w2, w3, w4] = All_fun_SSC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                            [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_SSC(L,x,p_w,g,D);

                        else
                            Eigenvalues = compute_eigenvalues_values_FEC_F3(H_1,H_2,h,L,E,omega);
                            NM = length(Eigenvalues);
                            Eigenvalues = Eigenvalues(1:NM);
                            Eigenvalues(1) = 0;
                            [phi, w, w2, w3, w4] = All_fun_FEC_F3(Eigenvalues, H_1, H_2, h, E, L, x, NM);
                            [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_FEC(L,x,p_w,g,D);
                        end

                        phi(:,1) = phi_0;
                        w(:,1)   = w_0;
                        w2(:,1)  = w2_0;
                        w3(:,1)  = w3_0;
                        w4(:,1)  = w4_0;

                        for k = 1:NM
                            Inner_product = sum(p_w*g*w(:,k).*conj(w(:,k))*dx) + ...
                                            sum(D*w(x>=0,k).*conj(w4(x>=0,k))*dx);
                            scale = 1/sqrt(Inner_product);
                            w(:,k)  = w(:,k)*scale;
                            w2(:,k) = w2(:,k)*scale;
                            w3(:,k) = w3(:,k)*scale;
                            w4(:,k) = w4(:,k)*scale;
                        end

                        C_n = zeros(NM,1);
                        D_n = zeros(NM,1);
                        for j = 1:NM
                            C_n(j) = sum(p_w*g*f_x.*conj(w(:,j))*dx) + ...
                                     sum(D*f_x(x>=0).*conj(w4(x>=0,j))*dx);
                            D_n(j) = sum(p_w*g*g_x.*conj(w(:,j))*dx) + ...
                                     sum(D*g_x(x>=0).*conj(w4(x>=0,j))*dx);
                        end

                        tau_new = 0;
                        W_x_t_plot = (C_n(1)+(D_n(1)*tau_new))*w(:,1) + ...
                                     w(:,2:NM) * (diag(cos(Eigenvalues(2:NM)*tau_new))*C_n(2:NM) + ...
                                     diag(sin(Eigenvalues(2:NM)*tau_new)./Eigenvalues(2:NM))*D_n(2:NM));

                        W_xx_plot = W_xx_old;
                    end

                    % =====================================================
                    % PLOT
                    % =====================================================

                    clf(fig);
                    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

                    currentLegendIdx = find(~isnan(targetTimes), 1, 'last');
                    if isempty(currentLegendIdx)
                        currentLegendIdx = 1;
                    end
                    legendText = sprintf('$t_{%d}=%.3f\\,\\mathrm{s}$', currentLegendIdx, targetTimes(currentLegendIdx));

                    % ---------------- LEFT PANEL ----------------
                    ax1 = nexttile;
                    plot(x(x<0)/L(end), real(W_x_t_plot(x<0)), 'LineWidth', 4, 'HandleVisibility','off');
                    box on;
                    hold on;

                    for jj = 1:length(L)-1
                        idx2 = (x > L(jj)) & (x < L(jj+1));
                        if mod(jj,2) == 1
                            col = [0.9 0 0];
                        else
                            col = [0 0.6 0];
                        end
                        plot(x(idx2)/L(end), real(W_x_t_plot(idx2)), 'LineWidth', 4, 'HandleVisibility','off', 'Color', col);
                        [valTmp, brTmp] = max(abs(real(W_xx_plot(x>=0)))); 
                    end
                    ylim('auto');
                    yl = ylim;
                    Ymin0 = -2.2;
                    Ymax0 = 2.2;
                    ylim([min(yl(1),Ymin0), max(yl(2),Ymax0)]);
                    xlim([-1, 1]);
                    xlabel('$$x/L\;$$','FontSize',35,'Interpreter','latex');
                    ylabel('$$W(x,t)\;$$','FontSize',35,'Interpreter','latex');
                    set(gca,'FontSize',35);
                    set(gca, 'LineWidth', 2.0);
                    hTime1 = plot(ax1, nan, nan, 'LineStyle', 'none', 'Color', 'none', 'HandleVisibility', 'on');
                    legend(ax1, hTime1, legendText,'Interpreter', 'latex', 'FontSize', 35, 'Location', 'southwest');
                    legend boxoff

                    % ---------------- RIGHT PANEL ----------------
                    ax2 = nexttile;
                    plot(x(x>=0)/L(end), real(W_xx_plot(x>=0)), 'k', 'LineWidth', 4, 'DisplayName', 'Second derivative of the displacement');
                    box on;
                    hold on;
                    ymax = max(1.1*max(abs(real(W_xx_plot(x>=0)))), 1.5*a);
                    ylim(ax2,[-ymax ymax]);
                    xlim([-1, 1]);
                    yline(a,'--k','LineWidth',4,'DisplayName','±a Breakup threshold');
                    yline(-a,'--k','LineWidth',4,'HandleVisibility','off');
                    xlabel('$$x/L\;$$','FontSize',35,'Interpreter','latex');
                    ylabel('$$\frac{\partial^2 W}{\partial x^2}\,(x,t)$$','FontSize',35,'Interpreter','latex');
                    set(gca,'FontSize',35);
                    hTime2 = plot(ax2, nan, nan, 'LineStyle', 'none', 'Color', 'none', 'HandleVisibility', 'off');
                    set(gca, 'LineWidth', 2.0);
                    drawnow;

                    % =====================================================
                    % SNAPSHOT SAVE
                    % =====================================================
                    if ~savedTargets(1)
                        panelLetter = letters(1);   % a = t1 = 0
                        fname = sprintf('fig%d%c.png', figNum, panelLetter);
                        exportgraphics(fig, fullfile(snapshots_dir, fname), 'Resolution', 300);
                        savedTargets(1) = true;
                    end

                    if breakup_now && ~isnan(snapIdx) && snapIdx <= nPanelsToSave && ~savedTargets(snapIdx)
                        panelLetter = letters(snapIdx);   % b..f = t2..t6
                        fname = sprintf('fig%d%c.png', figNum, panelLetter);
                        exportgraphics(fig, fullfile(snapshots_dir, fname), 'Resolution', 300);
                        savedTargets(snapIdx) = true;
                    end

                    fr = getframe(fig);
                    if ~isempty(fr.cdata)
                        writeVideo(movieObj, fr);
                    end

                    if ii == length(t)
                        stop_movie = true;
                    end

                    if breakup_now
                        break
                    end
                end

                if stop_movie
                    break
                end
            end

            close(movieObj);
            close(fig);
        end
    end
end

%% ============================================================
%  DELTA t STUDY ONLY (flag=1, ic=1, h=50) -> fig18a..fig18f
%% ============================================================

flag_dt = 1; 
ic_dt   = 1; 
h_dt    = 50;

dt_list = [1,0.5,0.25,0.125,0.0625,0.0312,0.0156,0.0078,0.0039,0.0019];

N_crack_dt = 6;
Tend_dt    = 500;
eps_cr_dt  = 1e-5;
a_dt       = 2*eps_cr_dt/h_dt;   

X_dt = nan(length(dt_list), 6);

H_1 = 300;
L0  = [0,1e4];
E   = 11e9;
p_i = 922.5;
p_w = 1025;
g   = 9.81;
v   = 0.33;

H_2 = H_1 - ((p_i/p_w) * h_dt);
omega = linspace(0,1, 5e3);
N     = 10001;
x     = linspace(-L0(end), L0(end), N);
dx    = x(2)-x(1);
x     = x(1:end-1) + dx/2;

D     = (E * h_dt^3) / (12 * (1 - v^2));
x_local = x(x>=0);

f_x0 = exp(-1e-6*(x + L0(end)/2).^2).';
c    = sqrt(g*H_1);
dfdx0 = -2e-6*(x + L0(end)/2).*exp(-1e-6*(x + L0(end)/2).^2);
dfdx0 = dfdx0.';
g_x0  = -c * dfdx0;

for idt = 1:length(dt_list)

    dt = dt_list(idt);
    t  = 0.00001:dt:Tend_dt;

    L  = L0;
    xi = zeros(N_crack_dt+1,1);
    ti = zeros(N_crack_dt+1,1);
    xi(1) = 0;
    ti(1) = 0;

    f_x = f_x0;
    g_x = g_x0;

    Eigenvalues = compute_eigenvalues_values_CC_F3(H_1,H_2,h_dt,L,E,omega);
    NM = length(Eigenvalues);
    Eigenvalues = Eigenvalues(1:NM);
    Eigenvalues(1) = 0;

    [phi, w, w2, w3, w4] = All_fun_CC_F3(Eigenvalues, H_1, H_2, h_dt, E, L, x, NM);
    [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_CC(L,x,p_w,g,D);
    phi(:,1)=phi_0; w(:,1)=w_0; w2(:,1)=w2_0; w3(:,1)=w3_0; w4(:,1)=w4_0;

    for k = 1:NM
        Inner_product = sum(p_w*g*w(:,k).*conj(w(:,k))*dx) + ...
                        sum(D*w(x>=0,k).*conj(w4(x>=0,k))*dx);
        scale = 1/sqrt(Inner_product);
        w(:,k)  = w(:,k)*scale;
        w2(:,k) = w2(:,k)*scale;
        w3(:,k) = w3(:,k)*scale;
        w4(:,k) = w4(:,k)*scale;
    end

    C_n = zeros(NM,1);
    D_n = zeros(NM,1);
    for j = 1:NM
        C_n(j) = sum(p_w*g*f_x.*conj(w(:,j))*dx) + ...
                 sum(D*f_x(x >= 0).*conj(w4(x >= 0,j))*dx);
        D_n(j) = sum(p_w*g*g_x.*conj(w(:,j))*dx) + ...
                 sum(D*g_x(x >= 0).*conj(w4(x >= 0,j))*dx);
    end

    n_crack = 1;
    while n_crack <= N_crack_dt

        found_crack = false;

        for ii = 1:length(t)

            if t(ii) <= ti(n_crack), continue, end
            tau = t(ii) - ti(n_crack);

            W_xx   = (C_n(1)+(D_n(1)*tau))*w2(:,1) + ...
                     w2(:,2:NM) *(diag(cos(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                     diag(sin(Eigenvalues(2:NM)*tau)./Eigenvalues(2:NM))*D_n(2:NM));

            W_x_t  = (C_n(1)+(D_n(1)*tau))*w(:,1) + ...
                     w(:,2:NM) *(diag(cos(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                     diag(sin(Eigenvalues(2:NM)*tau)./Eigenvalues(2:NM))*D_n(2:NM));

            W_tt_x = (D_n(1)*w(:,1)) + ...
                     w(:,2:NM)*(diag(-Eigenvalues(2:NM).*sin(Eigenvalues(2:NM)*tau))*C_n(2:NM) + ...
                     diag(cos(Eigenvalues(2:NM)*tau))*D_n(2:NM));

            [val, br] = max(abs(real(W_xx(x>=0))));

            if (abs(val) > a_dt) && x_local(br) < L(end) - dx

                xi(n_crack+1) = x_local(br);
                ti(n_crack+1) = t(ii);

                f_x = W_x_t;
                g_x = W_tt_x;

                L = [sort(xi(1:n_crack+1)).', L(end)];
                L = uniquetol(L);

                Eigenvalues = compute_eigenvalues_values_CC_F3(H_1,H_2,h_dt,L,E,omega);
                NM = length(Eigenvalues);
                Eigenvalues = Eigenvalues(1:NM);
                Eigenvalues(1) = 0;

                [phi, w, w2, w3, w4] = All_fun_CC_F3(Eigenvalues, H_1, H_2, h_dt, E, L, x, NM);
                [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_CC(L,x,p_w,g,D);
                phi(:,1)=phi_0; w(:,1)=w_0; w2(:,1)=w2_0; w3(:,1)=w3_0; w4(:,1)=w4_0;

                for k = 1:NM
                    Inner_product = sum(p_w*g*w(:,k).*conj(w(:,k))*dx) + ...
                                    sum(D*w(x>=0,k).*conj(w4(x>=0,k))*dx);
                    scale = 1/sqrt(Inner_product);
                    w(:,k)  = w(:,k)*scale;
                    w2(:,k) = w2(:,k)*scale;
                    w3(:,k) = w3(:,k)*scale;
                    w4(:,k) = w4(:,k)*scale;
                end

                C_n = zeros(NM,1);
                D_n = zeros(NM,1);
                for j = 1:NM
                    C_n(j) = sum(p_w*g*f_x.*conj(w(:,j))*dx) + ...
                             sum(D*f_x(x >= 0).*conj(w4(x >= 0,j))*dx);
                    D_n(j) = sum(p_w*g*g_x.*conj(w(:,j))*dx) + ...
                             sum(D*g_x(x >= 0).*conj(w4(x >= 0,j))*dx);
                end

                found_crack = true;
                break
            end
        end

        if ~found_crack
            break
        end

        n_crack = n_crack + 1;
    end

    for kk = 1:6
        X_dt(idt,kk) = xi(kk+1);
    end

end

figures_dir = fullfile(base_path,'Figures');
letters6 = 'abcdef';
xfreq = 1./dt_list;

for kk = 1:6
    fdt = figure;

    yk = X_dt(:,kk);

    plot(xfreq, yk, 'o', ...
        'MarkerFaceColor','k', ...
        'MarkerEdgeColor','k', ...
        'LineStyle','none', ...
        'MarkerSize', 9, ...
        'LineWidth', 2);

    xlabel('$$1/\Delta t$$', 'Interpreter','latex', 'FontSize', 35);
    ylabel(sprintf('$$x_{%d}$$', kk), 'Interpreter','latex', 'FontSize', 35);

    set(gca,'FontSize',24, 'LineWidth',2);
    box on;
    xlim([min(xfreq) max(xfreq)]);

    exportgraphics(fdt, fullfile(figures_dir, sprintf('fig18%c.png', letters6(kk))), 'Resolution', 300);
    close(fdt);
end

%% =======================
% ENERGY FIGURES
%% =======================
figures_dir = fullfile(base_path,'Figures');

f = figure;
colors = lines(6);
for k = 1:6
    Ek   = En{k};
    nm_k = numel(Ek);
    semilogy(1:nm_k, Ek, '*', ...
        'LineWidth', 2, ...
        'MarkerSize', 4, ...
        'Color', colors(k,:));
    hold on
end
xlabel('Number of modes');
ylabel('Energy','Interpreter','latex');
set(gca,'FontSize',15);
legendStrings = arrayfun(@(k) sprintf('Crack %d', k), 1:6, 'UniformOutput', false);
legend(legendStrings, 'Location', 'best');

out_png = fullfile(figures_dir, 'fig19a.png');
print(f, out_png, '-dpng', '-r300');

f1 = figure;
colors = lines(6);
for k = 1:6
    Ek   = En{k};
    nm_k = numel(Ek);
    plot(1:nm_k, Ek, '*', ...
        'LineWidth', 2, ...
        'MarkerSize', 4, ...
        'Color', colors(k,:));
    hold on
end
xlabel('Number of modes');
ylabel('Energy','Interpreter','latex');
set(gca,'FontSize',15);
legendStrings = arrayfun(@(k) sprintf('Crack %d', k), 1:6, 'UniformOutput', false);
legend(legendStrings, 'Location', 'best');

out_png = fullfile(figures_dir, 'fig19b.png');
print(f1, out_png, '-dpng', '-r300');

%% =======================
% % Graphical_Abstract
fig = figure('Units', 'inches','Position', [0 0 6 5],'Color', 'k');
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0 0 2.36 1.97]);
set(fig, 'PaperSize', [2.36 1.97]);

plot(x(x < 0)/1e4, real(W_x_t(x < 0)), 'Color', [0.7 0.85 1], 'LineWidth', 2);
hold on;

plot(x(x >= 0    & x <= 759)/1e4,    real(W_x_t(x >= 0    & x <= 759)),    'Color', 'w', 'LineWidth', 5);
plot(x(x > 759   & x <= 1529)/1e4,   real(W_x_t(x > 759   & x <= 1529)),   'Color', 'w', 'LineWidth', 5);
plot(x(x > 1529  & x <= 2317)/1e4,   real(W_x_t(x > 1529  & x <= 2317)),   'Color', 'w', 'LineWidth', 5);
plot(x(x > 2317  & x <= 3109)/1e4,   real(W_x_t(x > 2317  & x <= 3109)),   'Color', 'w', 'LineWidth', 5);
plot(x(x > 3109  & x <= 3555)/1e4,   real(W_x_t(x > 3109  & x <= 3555)),   'Color', 'w', 'LineWidth', 5);
plot(x(x > 3555  & x <= 4037)/1e4,   real(W_x_t(x > 3555  & x <= 4037)),   'Color', 'w', 'LineWidth', 5);
plot(x(x > 4037  & x <= 10000)/1e4,  real(W_x_t(x > 4037  & x <= 10000)),  'Color', 'w', 'LineWidth', 5);

hold off;
axis off;
set(gcf, 'InvertHardcopy', 'off');
print(fullfile(figures_dir,'Graphical_Abstract'), '-djpeg', '-r300');

%% =============================== FUNCTIONS ===============================
%% All_fun_SSC_F3
function [phi,w,w2,w3,w4] = All_fun_SSC_F3(Eigenvalues,H_1,H_2,h,E,L,x,NM)
N   = length(L)-1;
phi = zeros(length(x),NM); 
w = zeros(length(x),NM); 
w2 = zeros(length(x),NM);
w3 = zeros(length(x),NM); 
w4 = zeros(length(x),NM);
for j = 2:NM
    M  = compute_matrix_M_SSC_F3(Eigenvalues(j),H_1,h,H_2,E,L);
    V  = null(M);
    R  = V(1:2);
    C  = V(3:end);
    k  = Kroots_F3(Eigenvalues(j),H_1);
    r  = Rroots_F3(Eigenvalues(j),h,H_2,E);

    for i = 1:length(x)
        if x(i) <= L(1)
            phi(i,j) = R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i));
            w(i,j)   = (H_1/(-1i)/Eigenvalues(j))*(-k^2)*(R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i)));

        else
            for s = 1:N
                if L(s) <= x(i) && x(i) <= L(s+1)
                    idx = 6*(s-1) + 1;           
                    A   = C(idx:idx+2);         
                    B   = C(idx+3:idx+5);      
                    phi(i,j) = A(1)*exp(r(1)*(x(i) - L(s))) + A(2)*exp(r(2)*(x(i) - L(s))) + A(3)*exp(r(3)*(x(i) - L(s))) + ...
                    B(1)*exp(-r(1)*(x(i) - L(s+1))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) + B(3)*exp(-r(3)*(x(i) - L(s+1)));

                   w (i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^2*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^2*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^2*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w2(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^4*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^4*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^4*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w3(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^5*( A(1)*exp(r(1)*(x(i) - L(s))) - B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^5*( A(2)*exp(r(2)*(x(i) - L(s))) - B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^5*( A(3)*exp(r(3)*(x(i) - L(s))) - B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w4(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^6*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^6*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^6*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );
                 break
                end
            end
        end
    end
end
end
%% All_fun_FEC_F3
function [phi,w,w2,w3,w4] = All_fun_FEC_F3(Eigenvalues,H_1,H_2,h,E,L,x,NM)
N   = length(L)-1;
phi = zeros(length(x),NM); 
w = zeros(length(x),NM); 
w2 = zeros(length(x),NM);
w3 = zeros(length(x),NM); 
w4 = zeros(length(x),NM);
for j = 2:NM
    M  = compute_matrix_M_FEC_F3(Eigenvalues(j),H_1,h,H_2,E,L);
    V  = null(M);
    R = V(1:2);
    C = V(3:end);
    k  = Kroots_F3(Eigenvalues(j),H_1);
    r  = Rroots_F3(Eigenvalues(j),h,H_2,E);

    for i = 1:length(x)
        if x(i) <= L(1)
            phi(i,j) = R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i));
            w(i,j)   = (H_1/(-1i)/Eigenvalues(j))*(-k^2)*(R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i)));

        else
            for s = 1:N
                if L(s) <= x(i) && x(i) <= L(s+1)
                    idx = 6*(s-1) + 1;           
                    A   = C(idx:idx+2);         
                    B   = C(idx+3:idx+5);      
                    phi(i,j) = A(1)*exp(r(1)*(x(i) - L(s))) + A(2)*exp(r(2)*(x(i) - L(s))) + A(3)*exp(r(3)*(x(i) - L(s))) + ...
                    B(1)*exp(-r(1)*(x(i) - L(s+1))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) + B(3)*exp(-r(3)*(x(i) - L(s+1)));

                   w (i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^2*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^2*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^2*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w2(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^4*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^4*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^4*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w3(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^5*( A(1)*exp(r(1)*(x(i) - L(s))) - B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^5*( A(2)*exp(r(2)*(x(i) - L(s))) - B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^5*( A(3)*exp(r(3)*(x(i) - L(s))) - B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w4(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^6*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^6*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^6*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );
                
                break
                end
            end
        end
    end
    
end
end
%% All_fun_CC_F3
function [phi,w,w2,w3,w4] = All_fun_CC_F3(Eigenvalues,H_1,H_2,h,E,L,x,NM)
N   = length(L)-1;
phi = zeros(length(x),NM); 
w = zeros(length(x),NM); 
w2 = zeros(length(x),NM);
w3 = zeros(length(x),NM); 
w4 = zeros(length(x),NM);
for j = 2:NM
    M  = compute_matrix_M_CC_F3(Eigenvalues(j),H_1,h,H_2,E,L);
    V  = null(M);
    R = V(1:2);
    C = V(3:end);
    k  = Kroots_F3(Eigenvalues(j),H_1);
    r  = Rroots_F3(Eigenvalues(j),h,H_2,E);

    for i = 1:length(x)
        if x(i) <= L(1)
            phi(i,j) = R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i));
            w(i,j)   = (H_1/(-1i)/Eigenvalues(j))*(-k^2)*(R(1)*exp( 1i*k*x(i)) + R(2)*exp(-1i*k*x(i)));

        else
            for s = 1:N
                if L(s) <= x(i) && x(i) <= L(s+1)
                    idx = 6*(s-1) + 1;           
                    A   = C(idx:idx+2);         
                    B   = C(idx+3:idx+5);      
                    phi(i,j) = A(1)*exp(r(1)*(x(i) - L(s))) + A(2)*exp(r(2)*(x(i) - L(s))) + A(3)*exp(r(3)*(x(i) - L(s))) + ...
                    B(1)*exp(-r(1)*(x(i) - L(s+1))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) + B(3)*exp(-r(3)*(x(i) - L(s+1)));

                   w (i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^2*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^2*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^2*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w2(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                     (r(1)^4*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                     r(2)^4*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                     r(3)^4*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w3(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^5*( A(1)*exp(r(1)*(x(i) - L(s))) - B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^5*( A(2)*exp(r(2)*(x(i) - L(s))) - B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^5*( A(3)*exp(r(3)*(x(i) - L(s))) - B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );

                   w4(i,j) = (H_2/(-1i)/Eigenvalues(j))* ...
                  (r(1)^6*( A(1)*exp(r(1)*(x(i) - L(s))) + B(1)*exp(-r(1)*(x(i) - L(s+1))) ) + ...
                  r(2)^6*( A(2)*exp(r(2)*(x(i) - L(s))) + B(2)*exp(-r(2)*(x(i) - L(s+1))) ) + ...
                  r(3)^6*( A(3)*exp(r(3)*(x(i) - L(s))) + B(3)*exp(-r(3)*(x(i) - L(s+1))) ) );
                break
                end
            end
        end
    end
end
end

%% Static_Mode_N_FEC
function [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_FEC(L,x,p_w,g,D)

s  = exp(1i*pi/4)*(p_w*g/D)^(1/4);
N  = length(L)-1;            % number of segments
LN = L(end);                 % right end

% Outputs
phi_0 = zeros(size(x));
w_0   = zeros(size(x));
w2_0  = zeros(size(x));
w3_0  = zeros(size(x));
w4_0  = zeros(size(x));

% Ocean side x<=0
idx_ocean = (x <= 0);
phi_0(idx_ocean) = x(idx_ocean);
w_0(idx_ocean)   = 1;

% Left boundary (free): incoming interface values
w2_in = 0;    % w''(0)
w3_in = 0;    % w'''(0)

for seg = 1:N
    xl = L(seg);          % left of segment
    xr = L(seg+1);        % right of segment

    % ---- Build 4x4 for this segment: C = [C1..C4]^T ----
    % Right end at x=LN (free edge): w''(LN)=0, w'''(LN)=0
    dxR = LN - xr;
    csR = cos(s*dxR);  snR = sin(s*dxR);
    esR = exp(s*dxR);  emR = exp(-s*LN);

    vpp_LN  = [-s^2*csR,   -s^2*snR,    s^2*esR,    s^2*emR];     % w''(LN)
    vppp_LN = [ s^3*snR,   -s^3*csR,    s^3*esR,   -s^3*emR];     % w'''(LN)

    % Left of this segment x=xl: match incoming w'' and w'''
    dxL = xl - xr;
    csL = cos(s*dxL); snL = sin(s*dxL);
    esL = exp(s*dxL); emL = exp(-s*xl);

    vpp_xl  = [-s^2*csL,   -s^2*snL,    s^2*esL,   s^2*emL];      % w''(xl)
    vppp_xl = [ s^3*snL,   -s^3*csL,    s^3*esL,  -s^3*emL];      % w'''(xl)

    M = [ vpp_LN;
          vppp_LN;
          vpp_xl;
          vppp_xl ];

    % RHS: w''(LN)=0, w'''(LN)=0, w''(xl)=w2_in, w'''(xl)=w3_in
    f = [0; 0; w2_in; w3_in];

    C = M \ f;

    % ---- Accumulate this segment on (xl, xr] ----
    idx_seg = (x > xl) & (x <= xr);
    if any(idx_seg)
        dx = x(idx_seg) - xr;
        cs = cos(s*dx);  sn = sin(s*dx);
        es = exp(s*dx);  em = exp(-s*x(idx_seg));

        % phi, w (C-dependent only)
        phi_0(idx_seg) = phi_0(idx_seg) ...
            + (C(1)/s).*sin(s*dx) ...
            - (C(2)/s).*cos(s*dx) ...
            + C(3).*es ...
            - C(4).*em;

        w_0(idx_seg)   = w_0(idx_seg) ...
            + C(1).*cs + C(2).*sn + C(3).*es + C(4).*em;

        % derivatives
        w2_0(idx_seg)  = w2_0(idx_seg) ...
            - C(1)*s^2.*cs - C(2)*s^2.*sn + C(3)*s^2.*es + C(4)*s^2.*em;

        w3_0(idx_seg)  = w3_0(idx_seg) ...
            + C(1)*s^3.*sn - C(2)*s^3.*cs + C(3)*s^3.*es - C(4)*s^3.*em;

        w4_0(idx_seg)  = w4_0(idx_seg) ...
            + C(1)*s^4.*cs + C(2)*s^4.*sn + C(3)*s^4.*es + C(4)*s^4.*em;
    end

    % ---- Pass w'' and w''' at xr to next segment ----
    w2_in = -C(1)*s^2 + C(3)*s^2 + C(4)*s^2*exp(-s*xr);
    w3_in = -C(2)*s^3 + C(3)*s^3 - C(4)*s^3*exp(-s*xr);
    
% fprintf('seg=%d: C1=% .6e  C2=% .6e  C3=% .6e  C4=% .6e\n', ...
%         seg, C(1), C(2), C(3), C(4))

end
% ---- Add base terms ONCE for x>0 ----
idx_pos = (x > 0);
phi_0(idx_pos) = phi_0(idx_pos) + x(idx_pos);
w_0(idx_pos)   = w_0(idx_pos)   + 1;
end

%% Static_Mode_N_SSC
function [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_SSC(L,x,p_w,g,D)

s  = exp(1i*pi/4)*(p_w*g/D)^(1/4);
N  = length(L)-1;            % number of segments
LN = L(end);                 % right end

% Outputs
phi_0 = zeros(size(x));
w_0   = zeros(size(x));
w2_0  = zeros(size(x));
w3_0  = zeros(size(x));
w4_0  = zeros(size(x));

% Ocean side x<=0
idx_ocean = (x <= 0);
phi_0(idx_ocean) = x(idx_ocean);
w_0(idx_ocean)   = 1;

% Left boundary (free): incoming interface values
w2_in = 0;    % w''(0)
w3_in = 0;    % w'''(0)

for seg = 1:N
    xl = L(seg);          % left of segment
    xr = L(seg+1);        % right of segment

    % ---- Build 4x4 for this segment: C = [C1..C4]^T ----
    % Right end at x=LN (simply supported): w(LN)=0, w''(LN)=0
    dxR = LN - xr;
    csR = cos(s*dxR);  snR = sin(s*dxR);
    esR = exp(s*dxR);  emR = exp(-s*LN);

    v_LN   = [ csR,        snR,        esR,        emR   ];          % w(LN)
    vpp_LN = [-s^2*csR,   -s^2*snR,    s^2*esR,    s^2*emR];          % w''(LN)

    % Left of this segment x=xl: match incoming w'' and w'''
    dxL = xl - xr;
    csL = cos(s*dxL); snL = sin(s*dxL);
    esL = exp(s*dxL); emL = exp(-s*xl);

    vpp_xl  = [-s^2*csL,   -s^2*snL,    s^2*esL,   s^2*emL];          % w''(xl)
    vppp_xl = [ s^3*snL,   -s^3*csL,    s^3*esL,  -s^3*emL];          % w'''(xl)

    M = [ v_LN;
          vpp_LN;
          vpp_xl;
          vppp_xl ];

    % RHS: w(LN)=0 (base +1 added later), w''(LN)=0, w''(xl)=w2_in, w'''(xl)=w3_in
    f = [-1; 0; w2_in; w3_in];

    C = M \ f;

    % ---- Accumulate this segment on (xl, xr] ----
    idx_seg = (x > xl) & (x <= xr);
    if any(idx_seg)
        dx = x(idx_seg) - xr;
        cs = cos(s*dx);  sn = sin(s*dx);
        es = exp(s*dx);  em = exp(-s*x(idx_seg));

        % phi, w (C-dependent only)
        phi_0(idx_seg) = phi_0(idx_seg) ...
            + (C(1)/s).*sin(s*dx) ...
            - (C(2)/s).*cos(s*dx) ...
            + C(3).*es ...
            - C(4).*em;

        w_0(idx_seg)   = w_0(idx_seg) ...
            + C(1).*cs + C(2).*sn + C(3).*es + C(4).*em;

        % derivatives
        w2_0(idx_seg)  = w2_0(idx_seg) ...
            - C(1)*s^2.*cs - C(2)*s^2.*sn + C(3)*s^2.*es + C(4)*s^2.*em;

        w3_0(idx_seg)  = w3_0(idx_seg) ...
            + C(1)*s^3.*sn - C(2)*s^3.*cs + C(3)*s^3.*es - C(4)*s^3.*em;

        w4_0(idx_seg)  = w4_0(idx_seg) ...
            + C(1)*s^4.*cs + C(2)*s^4.*sn + C(3)*s^4.*es + C(4)*s^4.*em;
    end

    % ---- Pass w'' and w''' at xr to next segment ----
    % at dx=0: cos=1, sin=0, exp=1
    w2_in = -C(1)*s^2 + C(3)*s^2 + C(4)*s^2*exp(-s*xr);
    w3_in = -C(2)*s^3 + C(3)*s^3 - C(4)*s^3*exp(-s*xr);
end

% ---- Add base terms ONCE for x>0 ----
idx_pos = (x > 0);
phi_0(idx_pos) = phi_0(idx_pos) + x(idx_pos);
w_0(idx_pos)   = w_0(idx_pos)   + 1;
end

%% Static_Mode_N_CC
function [phi_0,w_0,w2_0,w3_0,w4_0] = Static_Mode_N_CC(L,x,p_w,g,D)

s  = exp(1i*pi/4)*(p_w*g/D)^(1/4);
N  = length(L)-1;          % number of segments
LN = L(end);               % right end

% Preallocate outputs
phi_0 = zeros(size(x));
w_0   = zeros(size(x));
w2_0  = zeros(size(x));
w3_0  = zeros(size(x));
w4_0  = zeros(size(x));

% ---- Ocean side: x <= 0  →  phi = x,  w = 1,  w2=w3=w4=0 ----
phi_0(x<= 0) = x( x<= 0);
w_0(x<= 0)   = 1;   

% ---- For x > 0 we build piecewise on cracks and then add base terms ----
% Start interface data at x=0 (free left edge)
w2_in = 0;
w3_in = 0;

for seg = 1:N
    xl = L(seg);          % left of segment
    xr = L(seg+1);        % right of segment

    % --- 4x4 system for this segment's coefficients C = [C1..C4]^T ---
    % Right boundary at x = LN (clamped): w(LN)=0, w'(LN)=0
    dxR  = LN - xr;                         % evaluate at LN, anchor at xr
    csR  = cos(s*dxR);  snR = sin(s*dxR);
    esR  = exp(s*dxR);  emR = exp(-s*LN);

    v_LN  = [ csR,        snR,        esR,        emR   ];    % w(LN)
    vp_LN = [-s*snR,      s*csR,      s*esR,     -s*emR ];    % w'(LN)

    % Left boundary of this segment at x = xl: match incoming w'' and w'''
    dxL  = xl - xr;                         % evaluate at xl, anchor at xr
    csL  = cos(s*dxL);  snL = sin(s*dxL);
    esL  = exp(s*dxL);  emL = exp(-s*xl);

    vpp_xl  = [-s^2*csL,   -s^2*snL,    s^2*esL,   s^2*emL];  % w''(xl)
    vppp_xl = [ s^3*snL,   -s^3*csL,    s^3*esL,  -s^3*emL];  % w'''(xl)

    M = [ v_LN;
          vp_LN;
          vpp_xl;
          vppp_xl ];

    % RHS: w(LN)=0 (we add +1 later), w'(LN)=0, w''(xl)=w2_in, w'''(xl)=w3_in
    f = [-1; 0; w2_in; w3_in];

    % Solve
    C = M \ f;

    % --- Accumulate this segment only on (xl, xr] ---
    idx_seg = (x > xl) & (x <= xr);
    if any(idx_seg)
        dx = x(idx_seg) - xr;
        cs = cos(s*dx);  sn = sin(s*dx);
        es = exp(s*dx);  em = exp(-s*x(idx_seg));

        % C-dependent parts
        phi_0(idx_seg) = phi_0(idx_seg) ...
            + (C(1)/s).*sin(s*dx) ...
            - (C(2)/s).*cos(s*dx) ...
            + C(3).*es ...
            - C(4).*em;

        w_0(idx_seg)   = w_0(idx_seg) ...
            + C(1).*cs + C(2).*sn + C(3).*es + C(4).*em;

        w2_0(idx_seg)  = w2_0(idx_seg) ...
            - C(1)*s^2.*cs - C(2)*s^2.*sn + C(3)*s^2.*es + C(4)*s^2.*em;

        w3_0(idx_seg)  = w3_0(idx_seg) ...
            + C(1)*s^3.*sn - C(2)*s^3.*cs + C(3)*s^3.*es - C(4)*s^3.*em;

        w4_0(idx_seg)  = w4_0(idx_seg) ...
            + C(1)*s^4.*cs + C(2)*s^4.*sn + C(3)*s^4.*es + C(4)*s^4.*em;
    end

    % --- Pass w'' and w''' at xr to the next segment (continuity) ---
    % at dx=0: cos=1, sin=0, exp=1
    w2_in = -C(1)*s^2 + C(3)*s^2 + C(4)*s^2*exp(-s*xr);
    w3_in = -C(2)*s^3 + C(3)*s^3 - C(4)*s^3*exp(-s*xr);
end

% ---- Add base terms ONCE for x>0 (same as your original) ----
phi_0(x > 0) = phi_0(x > 0) + x(x > 0);
w_0(x > 0)   = w_0(x > 0)   + 1;

end
%% compute_eigenvalues_values_FEC_F3
function Eigenvalues = compute_eigenvalues_values_FEC_F3(H_1,H_2, h,L, E, omega) 

omega_values_candidates= zeros(size(omega));
det_M=zeros(size(omega));

% Loop over omega to compute det(M)
for i = 1:length(omega)
M=compute_matrix_M_FEC_F3(omega(i), H_1, h, H_2, E,L);
det_M(i) = det(M);
end
%%% Find candidates for Zeros of the determinant
for i = 1:length(omega)-1
    if (real(det_M(i)) >= 0 && real(det_M(i+1)) <= 0) || (real(det_M(i)) <= 0 && real(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    elseif (imag(det_M(i)) >= 0 && imag(det_M(i+1)) <= 0) || (imag(det_M(i)) <= 0 && imag(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    end
end
omega_values_candidates = nonzeros(omega_values_candidates);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply the iterative method from the paper to refine omega_values
tol=1e-10;
h_step=1e-10;
omega_values = 0*omega_values_candidates;
for i = 1:length(omega_values_candidates)
    w_0 = omega_values_candidates(i);
    accuracy=1;
    while abs(accuracy) > tol 
        %%%Compute M(w_0) and M'(w_0) using finite difference
        %% Free Edge Conditions
        M_w0 = compute_matrix_M_FEC_F3(w_0, H_1, h, H_2, E,L);
        M_w0_plus_h = compute_matrix_M_FEC_F3(w_0 + h_step, H_1, h, H_2, E,L);
        M_w0_minus_h = compute_matrix_M_FEC_F3(w_0 - h_step, H_1, h, H_2, E,L);
        M_prime_w0 = (M_w0_plus_h - M_w0_minus_h) / (2 * h_step);

        %Solve the generalized eigenvalue problem M(w_0)v = λM'(w_0)v
        [~, D] = eig(M_w0, M_prime_w0);
        [~, I] = min(abs(diag(D)));
        lambda = D(I, I);
        [~, D] = eig(M_w0);
        [~, I] = min(abs(diag(D)));
        accuracy = D(I,I); % eigenvalue that should be zero
        % Update w_0
        w_0 = w_0 - real(lambda);

    end
    omega_values(i) = w_0;
end
Eigenvalues = uniquetol(omega_values, 5e-3);
end

%% compute_eigenvalues_values_SSC_F3
function Eigenvalues = compute_eigenvalues_values_SSC_F3(H_1,H_2, h,L, E, omega) 

omega_values_candidates= zeros(size(omega));
det_M=zeros(size(omega));

%% Loop over omega to compute det(M)
for i = 1:length(omega)
M=compute_matrix_M_SSC_F3(omega(i), H_1, h, H_2, E,L);
det_M(i) = det(M);
end
%% Find candidates for Zeros of the determinant
for i = 1:length(omega)-1
    if (real(det_M(i)) >= 0 && real(det_M(i+1)) <= 0) || (real(det_M(i)) <= 0 && real(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    elseif (imag(det_M(i)) >= 0 && imag(det_M(i+1)) <= 0) || (imag(det_M(i)) <= 0 && imag(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    end
end
omega_values_candidates = nonzeros(omega_values_candidates);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Apply the iterative method from the paper to refine omega_values
tol=1e-10;
h_step=1e-10;
omega_values = 0*omega_values_candidates;
for i = 1:length(omega_values_candidates)
    w_0 = omega_values_candidates(i);
    accuracy=1;
    while abs(accuracy) > tol 
        %% Compute M(w_0) and M'(w_0) using finite difference
        %% Simply Supported Conditions
        M_w0 = compute_matrix_M_SSC_F3(w_0, H_1, h, H_2, E,L);
        M_w0_plus_h = compute_matrix_M_SSC_F3(w_0 + h_step, H_1, h, H_2, E,L);
        M_w0_minus_h = compute_matrix_M_SSC_F3(w_0 - h_step, H_1, h, H_2, E,L);
        M_prime_w0 = (M_w0_plus_h - M_w0_minus_h) / (2 * h_step);

        %% Solve the generalized eigenvalue problem M(w_0)v = λM'(w_0)v
        [~, D] = eig(M_w0, M_prime_w0);
        [~, I] = min(abs(diag(D)));
        lambda = D(I, I);
        [~, D] = eig(M_w0);
        [~, I] = min(abs(diag(D)));
        accuracy = D(I,I); % eigenvalue that should be zero
        % Update w_0
        w_0 = w_0 - real(lambda);

    end
    omega_values(i) = w_0;
end
Eigenvalues = uniquetol(omega_values, 5e-3);
end

%% compute_eigenvalues_values_CC_F3 
function Eigenvalues = compute_eigenvalues_values_CC_F3(H_1,H_2, h,L, E, omega) 

omega_values_candidates= zeros(size(omega));
det_M=zeros(size(omega));

% Loop over omega to compute det(M)
for i = 1:length(omega)
M=compute_matrix_M_CC_F3(omega(i), H_1, h, H_2, E,L);
det_M(i) = det(M);
end
%%% Find candidates for Zeros of the determinant
for i = 1:length(omega)-1
    if (real(det_M(i)) >= 0 && real(det_M(i+1)) <= 0) || (real(det_M(i)) <= 0 && real(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    elseif (imag(det_M(i)) >= 0 && imag(det_M(i+1)) <= 0) || (imag(det_M(i)) <= 0 && imag(det_M(i+1)) >= 0)
        omega_values_candidates(i) = (omega(i) + omega(i+1)) / 2;
    end
end
omega_values_candidates = nonzeros(omega_values_candidates);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply the iterative method from the paper to refine omega_values
tol=1e-10;
h_step=1e-10;
omega_values = 0*omega_values_candidates;
for i = 1:length(omega_values_candidates)
    w_0 = omega_values_candidates(i);
    accuracy=1;
    while abs(accuracy) > tol 
        %%%Compute M(w_0) and M'(w_0) using finite difference
        %% Clamped Conditions
        M_w0 = compute_matrix_M_CC_F3(w_0, H_1, h, H_2, E,L);
        M_w0_plus_h = compute_matrix_M_CC_F3(w_0 + h_step, H_1, h, H_2, E,L);
        M_w0_minus_h = compute_matrix_M_CC_F3(w_0 - h_step, H_1, h, H_2, E,L);
        M_prime_w0 = (M_w0_plus_h - M_w0_minus_h) / (2 * h_step);
        %Solve the generalized eigenvalue problem M(w_0)v = λM'(w_0)v
        [~, D] = eig(M_w0, M_prime_w0);
        [~, I] = min(abs(diag(D)));
        lambda = D(I, I);
        [~, D] = eig(M_w0);
        [~, I] = min(abs(diag(D)));
        accuracy = D(I,I); % eigenvalue that should be zero
        % Update w_0
        w_0 = w_0 - real(lambda);

    end
    omega_values(i) = w_0;
end
Eigenvalues = uniquetol(omega_values, 5e-3);
end


%% compute_matrix_M_FEC_F3

function M = compute_matrix_M_FEC_F3(omega, H_1, h, H_2, E, L)
N   = length(L)-1;
dim = 6*N + 2;
M   = zeros(dim, 2 + 6*N);
k = Kroots_F3(omega, H_1);
r = Rroots_F3(omega, h, H_2, E); r = r(:);
if N==1
M = [1i*k*exp(-1i*k*L(end)), -1i*k*exp(1i*k*L(end)), 0, 0, 0, 0, 0, 0;
         1, 1, -1, -1, -1, -exp(r(1)*L(end)), -exp(r(2)*L(end)), -exp(r(3)*L(end));
         H_1*1i*k, -H_1*1i*k, -r(1)*H_2, -r(2)*H_2, -r(3)*H_2, r(1)*exp(r(1)*L(end))*H_2, r(2)*exp(r(2)*L(end))*H_2, r(3)*exp(r(3)*L(end))*H_2;
         0, 0, r(1)^4, r(2)^4, r(3)^4, r(1)^4*exp(r(1)*L(end)), r(2)^4*exp(r(2)*L(end)), r(3)^4*exp(r(3)*L(end));
         0, 0, r(1)^5, r(2)^5, r(3)^5, -r(1)^5*exp(r(1)*L(end)), -r(2)^5*exp(r(2)*L(end)), -r(3)^5*exp(r(3)*L(end));
         0, 0, r(1)*exp(r(1)*L(end)), r(2)*exp(r(2)*L(end)), r(3)*exp(r(3)*L(end)), -r(1), -r(2), -r(3);
         0, 0, r(1)^4*exp(r(1)*L(end)), r(2)^4*exp(r(2)*L(end)), r(3)^4*exp(r(3)*L(end)), r(1)^4, r(2)^4, r(3)^4;
         0, 0, r(1)^5*exp(r(1)*L(end)), r(2)^5*exp(r(2)*L(end)), r(3)^5*exp(r(3)*L(end)), -r(1)^5, -r(2)^5, -r(3)^5];
    M = diag([1; 1; 1/H_1/k; r(1)^(-4); r(1)^(-5); r(1)^(-1); r(1)^(-4); r(1)^(-5)])*M;
else
M(1,1:2) = [1i*k*exp(-1i*k*L(N+1)) , -1i*k*exp(1i*k*L(N+1))];
M(2,1:8) = [1 1 -1 -1 -1 -exp(r(1)*L(2)) -exp(r(2)*L(2)) -exp(r(3)*L(2))];
M(3,1:8) = [H_1*1i*k -H_1*1i*k -H_2*r(1) -H_2*r(2) -H_2*r(3) H_2*r(1)*exp(r(1)*L(2)) H_2*r(2)*exp(r(2)*L(2)) H_2*r(3)*exp(r(3)*L(2))];
M(4,3:8) = [r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(r(1)*L(2)) r(2)^4*exp(r(2)*L(2)) r(3)^4*exp(r(3)*L(2))];
M(5,3:8) = [r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(r(1)*L(2)) -r(2)^5*exp(r(2)*L(2)) -r(3)^5*exp(r(3)*L(2))];

for m = 1:N-1
    col1=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col2=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col3=3+6*(m-1):3+6*(m-1)+5;
    col4=3+6*(m-1):3+6*(m-1)+5;
    col5=3+6*(m-1)+6:3+6*(m-1)+11;
    col6=3+6*(m-1)+6:3+6*(m-1)+11;

    M(6*m,col1 ) =[exp(r(1)*(L(m+1)-L(m))) exp(r(2)*(L(m+1)-L(m))) exp(r(3)*(L(m+1)-L(m))) 1 1 1 -1 -1 -1 -exp(-r(1)*(L(m+1)-L(m+2))) -exp(-r(2)*(L(m+1)-L(m+2))) -exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+1, col2) =[r(1)*exp(r(1)*(L(m+1)-L(m))) r(2)*exp(r(2)*(L(m+1)-L(m))) r(3)*exp(r(3)*(L(m+1)-L(m))) -r(1) -r(2) -r(3) -r(1) -r(2) -r(3) r(1)*exp(-r(1)*(L(m+1)-L(m+2))) r(2)*exp(-r(2)*(L(m+1)-L(m+2))) r(3)*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+2,col3 ) = [r(1)^4*exp(r(1)*(L(m+1)-L(m))) r(2)^4*exp(r(2)*(L(m+1)-L(m))) r(3)^4*exp(r(3)*(L(m+1)-L(m))) r(1)^4 r(2)^4 r(3)^4];

    M(6*m+3, col4) =[r(1)^5*exp(r(1)*(L(m+1)-L(m))) r(2)^5*exp(r(2)*(L(m+1)-L(m))) r(3)^5*exp(r(3)*(L(m+1)-L(m))) -r(1)^5 -r(2)^5 -r(3)^5];

    M(6*m+4,col5 ) =[r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(-r(1)*(L(m+1)-L(m+2))) r(2)^4*exp(-r(2)*(L(m+1)-L(m+2))) r(3)^4*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+5,col6 ) =[r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(-r(1)*(L(m+1)-L(m+2))) -r(2)^5*exp(-r(2)*(L(m+1)-L(m+2))) -r(3)^5*exp(-r(3)*(L(m+1)-L(m+2)))];
end

M(dim-2, 3+6*(N-1):3+6*(N-1)+5) = [r(1)*exp(r(1)*(L(N+1)-L(N))) r(2)*exp(r(2)*(L(N+1)-L(N))) r(3)*exp(r(3)*(L(N+1)-L(N))) -r(1) -r(2) -r(3)];
M(dim-1, 3+6*(N-1):3+6*(N-1)+5) = [r(1)^4*exp(r(1)*(L(N+1)-L(N))) r(2)^4*exp(r(2)*(L(N+1)-L(N))) r(3)^4*exp(r(3)*(L(N+1)-L(N))) r(1)^4 r(2)^4 r(3)^4];
M(dim,   3+6*(N-1):3+6*(N-1)+5) = [r(1)^5*exp(r(1)*(L(N+1)-L(N))) r(2)^5*exp(r(2)*(L(N+1)-L(N))) r(3)^5*exp(r(3)*(L(N+1)-L(N))) -r(1)^5 -r(2)^5 -r(3)^5];

scale       = ones(dim,1);      
scale(3)    = 1/(H_1*k);          
for idx = 4 : dim-3              
    p = mod(idx-4,6);            
    switch p
        case 0                    
            scale(idx) = 1/(r(1)^4);
        case 1                    
            scale(idx) = 1/(r(1)^5);
        case {2,3}                
            scale(idx) = 1/(r(1));
        case 4                    
            scale(idx) = 1/(r(1)^4);
        case 5                    
            scale(idx) = 1/(r(1)^5);
    end
end
scale(dim-2) = 1/r(1);            
scale(dim-1) = 1/(r(1)^4);          
scale(dim) = 1/(r(1)^5);   
S=diag(scale);
M=S*M;
end
end

%% compute_matrix_M_SSC_F3

function M = compute_matrix_M_SSC_F3(omega, H_1, h, H_2, E, L)
N   = length(L)-1;
dim = 6*N + 2;
M   = zeros(dim, 2 + 6*N);
k = Kroots_F3(omega, H_1);
r = Rroots_F3(omega, h, H_2, E); r = r(:);
if N==1
M = [1i*k*exp(-1i*k*L(end)), -1i*k*exp(1i*k*L(end)), 0, 0, 0, 0, 0, 0;
         1, 1, -1, -1, -1, -exp(r(1)*L(end)), -exp(r(2)*L(end)), -exp(r(3)*L(end));
         H_1*1i*k, -H_1*1i*k, -r(1)*H_2, -r(2)*H_2, -r(3)*H_2, r(1)*exp(r(1)*L(end))*H_2, r(2)*exp(r(2)*L(end))*H_2, r(3)*exp(r(3)*L(end))*H_2;
         0, 0, r(1)^4, r(2)^4, r(3)^4, r(1)^4*exp(r(1)*L(end)), r(2)^4*exp(r(2)*L(end)), r(3)^4*exp(r(3)*L(end));
         0, 0, r(1)^5, r(2)^5, r(3)^5, -r(1)^5*exp(r(1)*L(end)), -r(2)^5*exp(r(2)*L(end)), -r(3)^5*exp(r(3)*L(end));
         0, 0, r(1)*exp(r(1)*L(end)), r(2)*exp(r(2)*L(end)), r(3)*exp(r(3)*L(end)), -r(1), -r(2), -r(3);
         0, 0, r(1)^2*exp(r(1)*L(end)), r(2)^2*exp(r(2)*L(end)), r(3)^2*exp(r(3)*L(end)), r(1)^2, r(2)^2, r(3)^2;
         0, 0, r(1)^4*exp(r(1)*L(end)), r(2)^4*exp(r(2)*L(end)), r(3)^4*exp(r(3)*L(end)), r(1)^4, r(2)^4, r(3)^4];
    M = diag([1; 1; 1/H_1/k; r(1)^(-4); r(1)^(-5); r(1)^(-1); r(1)^(-2); r(1)^(-4)])*M;
else
M(1,1:2) = [1i*k*exp(-1i*k*L(N+1)) , -1i*k*exp(1i*k*L(N+1))];
M(2,1:8) = [1 1 -1 -1 -1 -exp(r(1)*L(2)) -exp(r(2)*L(2)) -exp(r(3)*L(2))];
M(3,1:8) = [H_1*1i*k -H_1*1i*k -H_2*r(1) -H_2*r(2) -H_2*r(3) H_2*r(1)*exp(r(1)*L(2)) H_2*r(2)*exp(r(2)*L(2)) H_2*r(3)*exp(r(3)*L(2))];
M(4,3:8) = [r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(r(1)*L(2)) r(2)^4*exp(r(2)*L(2)) r(3)^4*exp(r(3)*L(2))];
M(5,3:8) = [r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(r(1)*L(2)) -r(2)^5*exp(r(2)*L(2)) -r(3)^5*exp(r(3)*L(2))];

for m = 1:N-1
    col1=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col2=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col3=3+6*(m-1):3+6*(m-1)+5;
    col4=3+6*(m-1):3+6*(m-1)+5;
    col5=3+6*(m-1)+6:3+6*(m-1)+11;
    col6=3+6*(m-1)+6:3+6*(m-1)+11;

    M(6*m,col1 ) =[exp(r(1)*(L(m+1)-L(m))) exp(r(2)*(L(m+1)-L(m))) exp(r(3)*(L(m+1)-L(m))) 1 1 1 -1 -1 -1 -exp(-r(1)*(L(m+1)-L(m+2))) -exp(-r(2)*(L(m+1)-L(m+2))) -exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+1, col2) =[r(1)*exp(r(1)*(L(m+1)-L(m))) r(2)*exp(r(2)*(L(m+1)-L(m))) r(3)*exp(r(3)*(L(m+1)-L(m))) -r(1) -r(2) -r(3) -r(1) -r(2) -r(3) r(1)*exp(-r(1)*(L(m+1)-L(m+2))) r(2)*exp(-r(2)*(L(m+1)-L(m+2))) r(3)*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+2,col3 ) = [r(1)^4*exp(r(1)*(L(m+1)-L(m))) r(2)^4*exp(r(2)*(L(m+1)-L(m))) r(3)^4*exp(r(3)*(L(m+1)-L(m))) r(1)^4 r(2)^4 r(3)^4];

    M(6*m+3, col4) =[r(1)^5*exp(r(1)*(L(m+1)-L(m))) r(2)^5*exp(r(2)*(L(m+1)-L(m))) r(3)^5*exp(r(3)*(L(m+1)-L(m))) -r(1)^5 -r(2)^5 -r(3)^5];

    M(6*m+4,col5 ) =[r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(-r(1)*(L(m+1)-L(m+2))) r(2)^4*exp(-r(2)*(L(m+1)-L(m+2))) r(3)^4*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+5,col6 ) =[r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(-r(1)*(L(m+1)-L(m+2))) -r(2)^5*exp(-r(2)*(L(m+1)-L(m+2))) -r(3)^5*exp(-r(3)*(L(m+1)-L(m+2)))];
end

M(dim-2, 3+6*(N-1):3+6*(N-1)+5) = [r(1)*exp(r(1)*(L(N+1)-L(N))) r(2)*exp(r(2)*(L(N+1)-L(N))) r(3)*exp(r(3)*(L(N+1)-L(N))) -r(1) -r(2) -r(3)];
M(dim-1, 3+6*(N-1):3+6*(N-1)+5) = [r(1)^2*exp(r(1)*(L(N+1)-L(N))) r(2)^2*exp(r(2)*(L(N+1)-L(N))) r(3)^2*exp(r(3)*(L(N+1)-L(N))) r(1)^2 r(2)^2 r(3)^2];
M(dim,   3+6*(N-1):3+6*(N-1)+5) = [r(1)^4*exp(r(1)*(L(N+1)-L(N))) r(2)^4*exp(r(2)*(L(N+1)-L(N))) r(3)^4*exp(r(3)*(L(N+1)-L(N))) r(1)^4 r(2)^4 r(3)^4];

scale       = ones(dim,1);      
scale(3)    = 1/(H_1*k);          
for idx = 4 : dim-3              
    p = mod(idx-4,6);            
    switch p
        case 0                    
            scale(idx) = 1/(r(1)^4);
        case 1                    
            scale(idx) = 1/(r(1)^5);
        case {2,3}                
            scale(idx) = 1/(r(1));
        case 4                    
            scale(idx) = 1/(r(1)^4);
        case 5                    
            scale(idx) = 1/(r(1)^5);
    end
end
scale(dim-2) = 1/r(1);            
scale(dim-1) = 1/(r(1)^2);          
scale(dim) = 1/(r(1)^4);   
S=diag(scale);
M=S*M;
end
end

%% compute_matrix_M_CC_F3

function M = compute_matrix_M_CC_F3(omega, H_1, h, H_2, E, L)
N   = length(L)-1;
dim = 6*N + 2;
M   = zeros(dim, 2 + 6*N);
k = Kroots_F3(omega, H_1);
r = Rroots_F3(omega, h, H_2, E); r = r(:);
if N==1
M = [1i*k*exp(-1i*k*L(end)), -1i*k*exp(1i*k*L(end)), 0, 0, 0, 0, 0, 0;
         1, 1, -1, -1, -1, -exp(r(1)*L(end)), -exp(r(2)*L(end)), -exp(r(3)*L(end));
         H_1*1i*k, -H_1*1i*k, -r(1)*H_2, -r(2)*H_2, -r(3)*H_2, r(1)*exp(r(1)*L(end))*H_2, r(2)*exp(r(2)*L(end))*H_2, r(3)*exp(r(3)*L(end))*H_2;
         0, 0, r(1)^4, r(2)^4, r(3)^4, r(1)^4*exp(r(1)*L(end)), r(2)^4*exp(r(2)*L(end)), r(3)^4*exp(r(3)*L(end));
         0, 0, r(1)^5, r(2)^5, r(3)^5, -r(1)^5*exp(r(1)*L(end)), -r(2)^5*exp(r(2)*L(end)), -r(3)^5*exp(r(3)*L(end));
         0, 0, r(1)*exp(r(1)*L(end)), r(2)*exp(r(2)*L(end)), r(3)*exp(r(3)*L(end)), -r(1), -r(2), -r(3);
         0, 0, r(1)^2*exp(r(1)*L(end)), r(2)^2*exp(r(2)*L(end)), r(3)^2*exp(r(3)*L(end)), r(1)^2, r(2)^2, r(3)^2;
         0, 0, r(1)^3*exp(r(1)*L(end)), r(2)^3*exp(r(2)*L(end)), r(3)^3*exp(r(3)*L(end)), -r(1)^3, -r(2)^3, -r(3)^3];
    M = diag([1; 1; 1/H_1/k; r(1)^(-4); r(1)^(-5); r(1)^(-1); r(1)^(-2); r(1)^(-3)])*M;
else
M(1,1:2) = [1i*k*exp(-1i*k*L(N+1)) , -1i*k*exp(1i*k*L(N+1))];
M(2,1:8) = [1 1 -1 -1 -1 -exp(r(1)*L(2)) -exp(r(2)*L(2)) -exp(r(3)*L(2))];
M(3,1:8) = [H_1*1i*k -H_1*1i*k -H_2*r(1) -H_2*r(2) -H_2*r(3) H_2*r(1)*exp(r(1)*L(2)) H_2*r(2)*exp(r(2)*L(2)) H_2*r(3)*exp(r(3)*L(2))];
M(4,3:8) = [r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(r(1)*L(2)) r(2)^4*exp(r(2)*L(2)) r(3)^4*exp(r(3)*L(2))];
M(5,3:8) = [r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(r(1)*L(2)) -r(2)^5*exp(r(2)*L(2)) -r(3)^5*exp(r(3)*L(2))];

for m = 1:N-1
    col1=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col2=[3+6*(m-1):3+6*(m-1)+5  3+6*(m-1)+6:3+6*(m-1)+11];
    col3=3+6*(m-1):3+6*(m-1)+5;
    col4=3+6*(m-1):3+6*(m-1)+5;
    col5=3+6*(m-1)+6:3+6*(m-1)+11;
    col6=3+6*(m-1)+6:3+6*(m-1)+11;

    M(6*m,col1 ) =[exp(r(1)*(L(m+1)-L(m))) exp(r(2)*(L(m+1)-L(m))) exp(r(3)*(L(m+1)-L(m))) 1 1 1 -1 -1 -1 -exp(-r(1)*(L(m+1)-L(m+2))) -exp(-r(2)*(L(m+1)-L(m+2))) -exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+1, col2) =[r(1)*exp(r(1)*(L(m+1)-L(m))) r(2)*exp(r(2)*(L(m+1)-L(m))) r(3)*exp(r(3)*(L(m+1)-L(m))) -r(1) -r(2) -r(3) -r(1) -r(2) -r(3) r(1)*exp(-r(1)*(L(m+1)-L(m+2))) r(2)*exp(-r(2)*(L(m+1)-L(m+2))) r(3)*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+2,col3 ) = [r(1)^4*exp(r(1)*(L(m+1)-L(m))) r(2)^4*exp(r(2)*(L(m+1)-L(m))) r(3)^4*exp(r(3)*(L(m+1)-L(m))) r(1)^4 r(2)^4 r(3)^4];

    M(6*m+3, col4) =[r(1)^5*exp(r(1)*(L(m+1)-L(m))) r(2)^5*exp(r(2)*(L(m+1)-L(m))) r(3)^5*exp(r(3)*(L(m+1)-L(m))) -r(1)^5 -r(2)^5 -r(3)^5];

    M(6*m+4,col5 ) =[r(1)^4 r(2)^4 r(3)^4 r(1)^4*exp(-r(1)*(L(m+1)-L(m+2))) r(2)^4*exp(-r(2)*(L(m+1)-L(m+2))) r(3)^4*exp(-r(3)*(L(m+1)-L(m+2)))];

    M(6*m+5,col6 ) =[r(1)^5 r(2)^5 r(3)^5 -r(1)^5*exp(-r(1)*(L(m+1)-L(m+2))) -r(2)^5*exp(-r(2)*(L(m+1)-L(m+2))) -r(3)^5*exp(-r(3)*(L(m+1)-L(m+2)))];
end

M(dim-2, 3+6*(N-1):3+6*(N-1)+5) = [r(1)*exp(r(1)*(L(N+1)-L(N))) r(2)*exp(r(2)*(L(N+1)-L(N))) r(3)*exp(r(3)*(L(N+1)-L(N))) -r(1) -r(2) -r(3)];
M(dim-1, 3+6*(N-1):3+6*(N-1)+5) = [r(1)^2*exp(r(1)*(L(N+1)-L(N))) r(2)^2*exp(r(2)*(L(N+1)-L(N))) r(3)^2*exp(r(3)*(L(N+1)-L(N))) r(1)^2 r(2)^2 r(3)^2];
M(dim,   3+6*(N-1):3+6*(N-1)+5) = [r(1)^3*exp(r(1)*(L(N+1)-L(N))) r(2)^3*exp(r(2)*(L(N+1)-L(N))) r(3)^3*exp(r(3)*(L(N+1)-L(N))) -r(1)^3 -r(2)^3 -r(3)^3];

scale       = ones(dim,1);      
scale(3)    = 1/(H_1*k);          
for idx = 4 : dim-3              
    p = mod(idx-4,6);            
    switch p
        case 0                    
            scale(idx) = 1/(r(1)^4);
        case 1                    
            scale(idx) = 1/(r(1)^5);
        case {2,3}                
            scale(idx) = 1/(r(1));
        case 4                    
            scale(idx) = 1/(r(1)^4);
        case 5                    
            scale(idx) = 1/(r(1)^5);
    end
end
scale(dim-2) = 1/r(1);            
scale(dim-1) = 1/(r(1)^2);          
scale(dim) = 1/(r(1)^3);   
S=diag(scale);
M=S*M;
end
end

%% Rroots_F3
function r_roots = Rroots_F3(omega, h, H_2,E)
% Constants and parameters
v = 0.33; % Poisson's ratio
g = 9.81; % Acceleration due to gravity (m/s^2)
p_w = 1025; % Density of water (kg/m^3)
p_i = 922.5; % Density of ice (kg/m^3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find r roots
FR=(E * h^3) / 12 / (1 - v^2);
A= (H_2*FR)/p_w;
B= (H_2 * g) - ((H_2 * omega.^2 * p_i * h)/ p_w);
C1= omega.^2;
p = [A, 0, 0, 0, B, 0, C1];
r = roots(p);
pr=10;
% Select specific roots based on the conditions
for count = 1:length(r)
    if real(round(r(count), pr)) == 0 && imag(round(r(count), pr)) > 0
        r(3) = r(count);
    elseif real(round(r(count), pr)) < 0 && imag(round(r(count), pr)) > 0
        r(1) = r(count);
    elseif real(round(r(count), pr)) < 0 && imag(round(r(count), pr)) < 0
        r(2) = r(count);
    end
end

% Delete the rest of the roots
r_roots=[r(1);r(2);r(3)];
end
%% Kroots_F3
function k_roots = Kroots_F3(omega,H_1)
% Constants and parameters
g = 9.81; % Acceleration due to gravity (m/s^2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find k roots
R = (omega.^2) / g /H_1;
q = [1, 0, R];
k = roots(q);
k_roots= abs(k(1));
end