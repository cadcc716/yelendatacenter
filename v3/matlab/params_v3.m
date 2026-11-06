function P = params_v3(varargin)
% PARAMS_V3  表A–表D 全部参数（与 Python 版 v3/code/params.py 逐项一致）。
%   P = params_v3()                    基准参数
%   P = params_v3('C_HEAT',0.25,...)   覆盖任意字段；安装上限用 'CAP_PV',15000 这类写法
% 单位：功率 kW，电量 kWh，冷/热量 kW_th、kWh_th，温度 ℃，费用 元。
%
% 调参记录（与正文参数表同步）：
%   采暖季 11-01–03-31；c_R 30→150；自然冷却上限 12→20 MW_th；式(12) N^max=6000；
%   禁止反送电（删售电）；删风电；C_w 归入热耦合（S3–S5 均含）；
%   蓄冷单价 150→80→60 元/kWh_th（C5 未达标，按调整顺序第一项）。

P = struct();
P.T = 24; P.DT = 1.0; P.DAYS = 365;
% ---------------- 表A GPU 频率档位 ----------------
P.P_BASE = 0.45; P.P_LK0 = 0.14; P.T_REF_LK = 40.0;
%          频率比  吞吐比v  动态功率  芯片发热h
P.GEARS = [1.0    1.00    0.560    0.700;
           0.9    0.94    0.448    0.588;
           0.8    0.87    0.350    0.490;
           0.7    0.78    0.266    0.406;
           0.6    0.68    0.196    0.336];
P.K_R = [1 2 3];            % 实时任务可用档位
P.K_B = [1 2 3 4 5];        % 可延期任务可用档位
% ---------------- 表B 热管理 ----------------
P.T_CHIP_MAX = 85.0; P.R_TH = 60.0;               % K/kW
P.ZONES = [30 35; 35 40; 40 45; 45 50; 50 55];
P.ALPHA_LK = 0.02; P.BETA = 0.75; P.C_W = 1150.0;
P.DT_APP = 5.0; P.DT_DES = 15.0; P.A_FC = 0.04;
P.COP0 = 6.0; P.C1 = 0.12; P.C2 = 0.10; P.T_AMB_REF = 25.0; P.T_W_REF = 35.0; P.COP_MIN = 3.0; P.COP_MAX = 9.0;
P.T_SUP = 70.0; P.ETA_C = 0.45; P.COP_HP_MAX = 7.0;
P.TES_LOSS = 0.005; P.TES_EFF = 0.97; P.TES_TAU = 4.0; P.TES_SOC0 = 0.5;
P.DELTA_E = 0.03; P.A_CIRC = 0.02; P.A_AIR = 0.25; P.P0 = 300.0;
% 无热耦合（S1、S2）：45 ℃ 温水液冷设计，C_w=0 瞬时平衡，自然冷却按 45–50 ℃ 温区逐时气温计算，
% 冷机 COP 取设计日值（全年 0.4% 最高小时气温），漏电取参考值，不施加门控
P.T_W_DES = 45.0; P.AMB_DES_Q = 0.996;
% ---------------- 电池 ----------------
P.B_ETA_C = 0.95; P.B_ETA_D = 0.95; P.B_LOSS = 0.0002; P.B_SOC_MIN = 0.1; P.B_SOC_MAX = 0.9; P.B_SOC0 = 0.5; P.C_B_MAX = 0.5;
% ---------------- 表C 设备经济参数：单价, 寿命, 年运维率, 安装上限 ----------------
P.DISC = 0.08;
P.CAPS = {'N','PV','PB','EB','FC','MC','TES','HP'};
P.COST = struct( ...
    'N',   [250000, 5,  0.02,  8000], ...
    'PV',  [3000,   25, 0.01,  15000], ...
    'PB',  [300,    12, 0.015, 20000], ...
    'EB',  [600,    12, 0.015, 80000], ...
    'FC',  [800,    20, 0.02,  20000], ...
    'MC',  [1200,   20, 0.02,  10000], ...
    'TES', [60,     25, 0.01,  60000], ...
    'HP',  [2000,   20, 0.02,  6000]);
% ---------------- 表D 价格、碳交易与惩罚 ----------------
P.PRICE_PEAK = 1.05; P.PRICE_FLAT = 0.62; P.PRICE_VALLEY = 0.28;
P.PEAK_H = [8:10, 17:21];          % 8–11 时、17–22 时（时段序号 0–23）
P.VALLEY_H = [23, 0:6];            % 23–7 时
P.C_HEAT = 0.20; P.P_GRID_MAX = 12000.0;
P.E_GRID = 0.57e-3; P.E_HEAT = 0.22e-3;
P.QUOTA = 0.6e-3; P.CO2_PRICE = 80.0; P.CO2_LEN = 20.0; P.CO2_GROW = 0.25; P.CO2_V = 5;
P.C_R = 150.0; P.C_O = 3.0; P.C_CUR = 0.1; P.C_DEG = 0.02;
P.D_DEADLINE = 6; P.B_PERIODIC = true;
% ---------------- 模糊集 ----------------
P.K = 12; P.M = 365; P.ALPHA1 = 0.5; P.ALPHAINF = 0.99;
% ---------------- 求解 ----------------
P.N_BIGM = 6000; P.GATE_NEXT = true; P.RELAX_COMPL = true;

for i = 1:2:numel(varargin)
    k = varargin{i}; v = varargin{i+1};
    if strncmp(k, 'CAP_', 4)
        c = P.COST.(k(5:end)); c(4) = v; P.COST.(k(5:end)) = c;
    elseif isfield(P, k)
        P.(k) = v;
    else
        error('params_v3: 未知参数 %s', k);
    end
end
end
