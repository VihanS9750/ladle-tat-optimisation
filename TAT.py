"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        AMALGAM STEEL × IITM — LADLE TAT OPTIMISATION SUITE  v5-final       ║
║                    5-Component Industrial Energy Model                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Optimises Ladle Turnaround Time (TAT) for an IF → LF → Caster route.
Cases: 8 IF + 2 LF  |  4 IF + 1 LF  |  General M+N scaling table

ENERGY MODEL — 5 COMPONENTS
────────────────────────────
  [1] E_steel       : sensible heat of liquid steel
  [2] E_slag        : slag heating requirement
  [3] E_refractory  : refractory wall heat absorption
  [4] E_loss_dyn    : radiation + conduction losses
  [5] E_reaction    : dephosphorisation exotherm (credit)

  Total raw = [1]+[2]+[3]+[4]+[5]  (÷ arc efficiency η)

  Note: Standard ladle pre-heating practice is assumed under normal LF
  operation; its thermal benefit is implicitly included in the refractory
  absorption term (E_ref_nominal = 8.0 kWh/T).

COMPANY DATA (as of 2026-05-20)
────────────────────────────────
  T_tap = 1 660 °C  (CONFIRMED)  |  Heat tonnage: 13–17 T, 15 T nominal
  Zone A loss: 50 °C total       |  Zone B: 2–3 °C/min
  T_cast: ~1 620–1 630 °C (target 1 625 °C)
"""

import math
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SIGMA  = 5.67e-8
Cp_STL = 800
Cp_kWh = Cp_STL / 3.6e6
R_GAS  = 8.314

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = {
    "m_heat_nominal"  : 15.0,
    "m_heat_min"      : 13.0,
    "m_heat_max"      : 17.0,
    "m_heat_design"   : 13.5,

    "T_tap"           : 1660,

    "dT_zone_A"       : 50.0,
    "dT_zone_A_rate"  : 2.5,

    "dT_zone_B_rate"  : 2.5,
    "dT_zone_B_lo"    : 2.0,
    "dT_zone_B_hi"    : 3.0,

    "T_cast_req"      : 1625,
    "T_cast_lo"       : 1620,
    "T_cast_hi"       : 1630,

    "T_tap_op"        : 10,
    "T_tr1"           : 10,
    "T_tr2"           : 5,
    "T_IF"            : 165,
    "TAT_max"         : 40,
    "W_hold_max"      : 15,

    "ladle_dia"       : 1.6,
    "ladle_h"         : 1.1,
    "ref_L"           : 0.15,
    "ref_k"           : 2.0,
    "h_ext"           : 10.0,
    "eps_open"        : 0.85,
    "eps_slag"        : 0.40,
    "T_amb_K"         : 303,

    "P_init"          : 0.090,
    "P_init_lo"       : 0.080,
    "P_init_hi"       : 0.100,
    "P_target"        : 0.040,
    "k_IF"            : 0.010,
    "k_LF_base"       : 0.0116,
    "Ea_deP"          : 50_000,
    "T_ref_deP_K"     : 1843,

    # Energy model components
    "slag_frac"       : 0.04,
    "Cp_slag"         : 1.2,
    "dT_slag"         : 200,
    # [3] Refractory absorption — preheating assumed nominal; implicitly included
    "E_ref_nominal"   : 8.0,    # kWh/T
    "dH_deP_kJ_mol"   : 2100,
    "MW_P"            : 30.97,

    "eta_foamy"       : 0.82,
    "eta_normal"      : 0.75,
    "eta_poor"        : 0.62,

    "P_LF_MVA"        : None,
    "pf_LF"           : 0.85,
    "E_LF_limit"      : 100,

    "rho_max"         : 0.40,
    "arrival_erlang_k": 3,
    "service_cv"      : 0.20,

    "grades": {
        "E250": {"T_liquidus": 1518, "superheat": 25},
        "E350": {"T_liquidus": 1520, "superheat": 30},
        "V55" : {"T_liquidus": 1525, "superheat": 35},
    },
    "grade_default"   : "E350",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE T — THERMAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _ladle_areas():
    d = PARAMS["ladle_dia"]
    h = PARAMS["ladle_h"]
    return math.pi * (d / 2) ** 2, math.pi * d * h


def heat_loss_rate_physics(T_steel_C, slag_cover=True, m_heat=None):
    """
    Instantaneous heat loss from ladle via T⁴ radiation (top) and series-
    resistance conduction (wall). alpha=0.70 — normal preheated operation.
    Returns dict of fluxes and °C/min cooling rate.
    """
    p  = PARAMS
    m  = (m_heat or p["m_heat_nominal"]) * 1000
    Ts = T_steel_C + 273.15
    Ta = p["T_amb_K"]
    A_top, A_wall = _ladle_areas()

    eps   = p["eps_slag"] if slag_cover else p["eps_open"]
    q_rad = eps * SIGMA * A_top * (Ts**4 - Ta**4) / 1000   # kW

    # alpha = 0.70: preheated refractory reduces wall gradient
    alpha   = 0.70
    R_brick = p["ref_L"] / p["ref_k"]
    R_conv  = 1.0 / p["h_ext"]
    q_cond  = alpha * A_wall * (Ts - Ta) / (R_brick + R_conv) / 1000  # kW

    q_total = q_rad + q_cond
    dT_dt   = (q_total * 1000) / (m * Cp_STL) * 60  # °C/min

    return {
        "q_rad_kW"        : round(q_rad,  2),
        "q_cond_kW"       : round(q_cond, 2),
        "q_total_kW"      : round(q_total,2),
        "dT_dt_C_per_min" : round(dT_dt,  3),
    }


def integrate_cooling(T_start_C, duration_min, slag_cover=True,
                      m_heat=None, dt_step=0.5):
    """
    Numerical T⁴ cooling integration.
    Returns (T_final_C, avg_cooling_rate °C/min, accumulated heat loss kW·min).
    """
    T, t, q_acc, dT_sum, steps = T_start_C, 0.0, 0.0, 0.0, 0
    while t < duration_min:
        dt     = min(dt_step, duration_min - t)
        hl     = heat_loss_rate_physics(T, slag_cover, m_heat)
        dT     = hl["dT_dt_C_per_min"] * dt
        q_acc += hl["q_total_kW"] * dt
        dT_sum+= hl["dT_dt_C_per_min"]
        T     -= dT
        t     += dt
        steps += 1
    return T, dT_sum / max(steps, 1), q_acc


def thermal_profile(T_LF, T_wait=0.0, m_heat=None,
                    dT_B_rate=None, T_cast_req=None):
    """
    Full temperature trajectory: tap → arrive LF → queue → arc → caster.
    Back-solves T_exit_LF so steel arrives at T_cast_req after the 5-min
    caster transfer.  Returns temperatures + 5-component arc energy breakdown.
    """
    p    = PARAMS
    m    = m_heat or p["m_heat_nominal"]
    dT_B = dT_B_rate or p["dT_zone_B_rate"]
    T_cr = T_cast_req or p["T_cast_req"]

    T_tap    = p["T_tap"]
    T_arr_LF = T_tap - p["dT_zone_A"]          # 1 610 °C

    slag_in_queue = (T_wait <= 2.0)
    T_after_wait, avg_B, _ = integrate_cooling(
        T_arr_LF, T_wait, slag_cover=slag_in_queue, m_heat=m)

    # Back-solve T_exit_LF with Brent's method
    def _tr2_residual(T_exit):
        T_end, _, _ = integrate_cooling(T_exit, p["T_tr2"],
                                        slag_cover=True, m_heat=m)
        return T_end - T_cr
    try:
        T_exit_LF = brentq(_tr2_residual, T_cr, T_cr + 80, xtol=0.1)
    except ValueError:
        T_exit_LF = T_cr + dT_B * p["T_tr2"]

    delta_T_arc = T_exit_LF - T_after_wait
    T_at_caster, _, _ = integrate_cooling(T_exit_LF, p["T_tr2"],
                                          slag_cover=True, m_heat=m)

    # ── 5-COMPONENT ENERGY MODEL ─────────────────────────────────────────────
    # [1] Steel sensible heat
    E_steel = Cp_kWh * 1000 * max(delta_T_arc, 0.0)

    # [2] Slag heating  (4% mass, +200 °C)
    m_slag_kg   = m * p["slag_frac"] * 1000
    Cp_slag_kWh = p["Cp_slag"] / 3600
    E_slag      = (m_slag_kg * Cp_slag_kWh * p["dT_slag"]) / m

    # [3] Refractory absorption — nominal preheated value
    E_ref = p["E_ref_nominal"]

    # [4] Dynamic radiation + conduction losses over arc-on time
    T_avg_LF = (T_after_wait + T_exit_LF) / 2
    _, _, q_LF_kWmin = integrate_cooling(T_avg_LF, T_LF,
                                         slag_cover=True, m_heat=m)
    E_loss = (q_LF_kWmin / 60) / m

    # [5] DeP exotherm credit (negative)
    delta_P_frac  = max(p["P_init"] - p["P_target"], 0)
    delta_P_kg_T  = delta_P_frac * 10
    mol_P_removed = (delta_P_kg_T * 1000) / p["MW_P"]
    E_rxn         = -(mol_P_removed * p["dH_deP_kJ_mol"]) / 3600

    eta       = p["eta_normal"]
    raw_total = E_steel + E_slag + E_ref + E_loss + E_rxn
    E_physics = raw_total / eta

    E_Pt = None
    if p.get("P_LF_MVA"):
        P_elec_kW = p["P_LF_MVA"] * 1000 * p["pf_LF"]
        E_Pt      = (P_elec_kW * (T_LF / 60)) / m

    E_total = E_physics if E_Pt is None else max(E_physics, E_Pt)

    return {
        "T_tap"           : T_tap,
        "T_arr_LF"        : round(T_arr_LF,    1),
        "T_after_wait"    : round(T_after_wait, 1),
        "T_exit_LF"       : round(T_exit_LF,   1),
        "T_at_caster"     : round(T_at_caster,  1),
        "T_cast_req"      : T_cr,
        "cast_temp_ok"    : T_at_caster >= T_cr - 0.5,
        "delta_T_arc_C"   : round(delta_T_arc,  1),
        "E_steel_kWh_T"   : round(E_steel,  2),
        "E_slag_kWh_T"    : round(E_slag,   2),
        "E_ref_kWh_T"     : round(E_ref,    2),
        "E_loss_kWh_T"    : round(E_loss,   2),
        "E_rxn_kWh_T"     : round(E_rxn,    2),
        "eta_used"        : eta,
        "E_physics_kWh_T" : round(E_physics,2),
        "E_Pt_kWh_T"      : round(E_Pt, 2) if E_Pt else None,
        "E_total_kWh_T"   : round(E_total,  2),
        "E_within_limit"  : E_total < p["E_LF_limit"],
        "T_wait_used"     : T_wait,
        "avg_dT_queue"    : round(avg_B, 3),
    }


def tonnage_sensitivity(T_LF, T_wait=1.0):
    results = {}
    for label, m in [("13 T (min)", 13), ("15 T (nom)", 15), ("17 T (max)", 17)]:
        prof = thermal_profile(T_LF, T_wait, m_heat=m)
        results[label] = {
            "m_heat"     : m,
            "T_exit_LF"  : prof["T_exit_LF"],
            "delta_T_arc": prof["delta_T_arc_C"],
            "E_kWh_T"    : prof["E_total_kWh_T"],
            "cast_ok"    : prof["cast_temp_ok"],
        }
    return results


def casting_temp_sensitivity(T_LF, T_wait=1.0):
    results = {}
    for label, Tc in [("1620 °C (lo)", 1620), ("1625 °C (mid)", 1625), ("1630 °C (hi)", 1630)]:
        prof = thermal_profile(T_LF, T_wait, T_cast_req=Tc)
        results[label] = {
            "T_cast_req" : Tc,
            "T_exit_LF"  : prof["T_exit_LF"],
            "delta_T_arc": prof["delta_T_arc_C"],
            "E_kWh_T"    : prof["E_total_kWh_T"],
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE P — PHOSPHORUS KINETICS
# ═══════════════════════════════════════════════════════════════════════════════

def k_LF_at_temperature(T_LF_C):
    """Arrhenius kinetics × thermodynamic correction. Net k peaks ~1 570 °C."""
    p   = PARAMS
    T_K = T_LF_C + 273.15
    T0  = p["T_ref_deP_K"]
    f_kin = math.exp(-p["Ea_deP"] / R_GAS * (1.0 / T_K - 1.0 / T0))
    f_thd = math.exp(3000.0 * (1.0 / T_K - 1.0 / T0))
    return max(p["k_LF_base"] * f_kin * f_thd, 0.005)


def compute_kLF_enhanced(T_LF_C=1570,
                          B_new=4.5,  B_old=2.5,
                          Q_new=0.4,  Q_old=0.1,
                          FeO_new=18, FeO_old=5):
    """
    Enhanced k_LF from improved slag chemistry + Ar stirring.
    Turkdogan (basicity), Mazumdar-Evans (stirring), FeO correlation.
    """
    k_T   = k_LF_at_temperature(T_LF_C)
    f_B   = (B_new  / B_old)   ** 0.5
    f_Q   = (Q_new  / Q_old)   ** 0.4
    f_FeO = (FeO_new / FeO_old) ** 0.5
    k_enh = k_T * f_B * f_Q * f_FeO
    return {
        "k_base"        : PARAMS["k_LF_base"],
        "k_at_T"        : round(k_T,   6),
        "T_LF_C"        : T_LF_C,
        "f_basicity"    : round(f_B,   3),
        "f_stirring"    : round(f_Q,   3),
        "f_FeO"         : round(f_FeO, 3),
        "k_LF_enhanced" : round(k_enh, 6),
        "enhancement_x" : round(k_enh / PARAMS["k_LF_base"], 2),
    }


def two_stage_deP(P_init, P_target, k_IF, t_IF_deP, k_LF):
    """First-order deP: Stage 1 in IF, Stage 2 in LF. Solves for t_LF."""
    P1 = P_init * math.exp(-k_IF * t_IF_deP)
    if P1 <= P_target:
        return {"P_after_IF": P1, "t_LF_deP_required": 0.0,
                "P_final": P1, "P_target_met": True, "kLF_t_product": 0.0}
    kt  = math.log(P1 / P_target)
    t_LF = kt / k_LF
    Pf  = P1 * math.exp(-k_LF * t_LF)
    return {
        "P_after_IF"        : round(P1,  6),
        "kLF_t_product"     : round(kt,  4),
        "t_LF_deP_required" : round(t_LF,3),
        "P_final"           : round(Pf,  6),
        "P_target_met"      : Pf <= P_target + 1e-6,
    }


def P_init_sensitivity(T_LF, t_IF_deP=40, T_op_C=1570):
    p    = PARAMS
    k_LF = compute_kLF_enhanced(T_LF_C=T_op_C)["k_LF_enhanced"]
    results = {}
    for label, Pi in [("0.08% (lo)", 0.08), ("0.09% (mid)", 0.09), ("0.10% (hi)", 0.10)]:
        d = two_stage_deP(Pi, p["P_target"], p["k_IF"], t_IF_deP, k_LF)
        results[label] = {
            "P_init"     : Pi,
            "P_after_IF" : d["P_after_IF"],
            "t_LF_deP"   : d["t_LF_deP_required"],
            "P_final"    : d["P_final"],
            "met"        : d["P_target_met"],
            "fits_in_TLF": d["t_LF_deP_required"] + 2 <= T_LF,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE Q — QUEUING THEORY (Erlang-C)
# ═══════════════════════════════════════════════════════════════════════════════

def erlang_c(c, rho):
    if rho >= 1.0:
        return 1.0
    a   = c * rho
    num = (a ** c) / (math.factorial(c) * (1 - rho))
    den = sum((a ** n) / math.factorial(n) for n in range(c)) + num
    return num / den


def queue_stats(M, N, T_LF, T_IF=None):
    """M/M/N Erlang-C model. Returns utilisation ρ and expected wait E[Wq]."""
    p    = PARAMS
    T_IF = T_IF or p["T_IF"]
    lam  = M / T_IF
    mu   = 1.0 / T_LF
    rho  = lam / (N * mu)
    if rho >= 1.0:
        return {"stable": False, "rho": rho,
                "E_Wq_analytical": float("inf"), "lambda": lam, "mu": mu}
    C    = erlang_c(N, rho) if N > 1 else rho
    E_Wq = C / (N * mu - lam)
    try:
        W95 = -math.log(0.05) / (N * mu - lam)
    except ZeroDivisionError:
        W95 = float("inf")
    return {
        "stable"          : True,
        "rho"             : round(rho,  5),
        "C"               : round(C,    5),
        "E_Wq_analytical" : round(E_Wq, 4),
        "W95_analytical"  : round(W95,  2),
        "lambda_per_min"  : round(lam,  6),
        "mu_per_min"      : round(mu,   6),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE O — INTEGRATED OPTIMISER
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_configuration(M, N, t_IF_deP=40, m_heat=None,
                            T_cast_req=None, verbose=True):
    """
    Scans T_LF to find the minimum feasible service time satisfying all 5
    constraints simultaneously: TAT ≤ 40, ρ < 1, E[Wq] ≤ 15, %P ≤ 0.04%,
    T_cast ≥ req, E_LF < 100 kWh/T.
    """
    p    = PARAMS
    m    = m_heat or p["m_heat_nominal"]
    T_cr = T_cast_req or p["T_cast_req"]

    T_LF_stable_max = N * p["T_IF"] / M * 0.995
    k_info  = compute_kLF_enhanced(T_LF_C=1625.0)
    k_LF    = k_info["k_LF_enhanced"]

    deP_wc     = two_stage_deP(p["P_init_hi"], p["P_target"],
                               p["k_IF"], t_IF_deP, k_LF)
    T_LF_min_P = deP_wc["t_LF_deP_required"] + 2.0
    deP_nom    = two_stage_deP(p["P_init"], p["P_target"],
                               p["k_IF"], t_IF_deP, k_LF)

    candidates = np.arange(max(T_LF_min_P, 5.0), min(T_LF_stable_max, 40.0), 0.1)
    results    = []

    for T_LF in candidates:
        qs   = queue_stats(M, N, T_LF)
        if not qs["stable"]:
            continue
        E_Wq = qs["E_Wq_analytical"]
        if E_Wq > p["W_hold_max"]:
            continue

        prof   = thermal_profile(T_LF, E_Wq, m_heat=m, T_cast_req=T_cr)
        T_op   = (prof["T_arr_LF"] + prof["T_exit_LF"]) / 2
        k_i    = compute_kLF_enhanced(T_LF_C=T_op)
        k_LF_i = k_i["k_LF_enhanced"]

        deP_i  = two_stage_deP(p["P_init_hi"], p["P_target"],
                               p["k_IF"], t_IF_deP, k_LF_i)
        deP_rp = two_stage_deP(p["P_init"],    p["P_target"],
                               p["k_IF"], t_IF_deP, k_LF_i)

        prof_cons = thermal_profile(T_LF, E_Wq,
                                    m_heat=p["m_heat_min"],
                                    T_cast_req=p["T_cast_hi"])

        TAT = p["T_tap_op"] + p["T_tr1"] + E_Wq + T_LF + p["T_tr2"]
        feasible = (
            TAT <= p["TAT_max"]                          and
            qs["rho"] < 1.0                              and
            E_Wq <= p["W_hold_max"]                      and
            deP_i["P_target_met"]                        and
            prof["cast_temp_ok"]                         and
            prof_cons["E_total_kWh_T"] < p["E_LF_limit"]
        )
        results.append({
            "T_LF"         : round(T_LF, 2),
            "TAT"          : round(TAT,  3),
            "rho"          : qs["rho"],
            "E_Wq"         : round(E_Wq, 3),
            "T_arr_LF"     : prof["T_arr_LF"],
            "T_after_wait" : prof["T_after_wait"],
            "T_exit_LF"    : prof["T_exit_LF"],
            "T_at_caster"  : prof["T_at_caster"],
            "delta_T_arc"  : prof["delta_T_arc_C"],
            "E_kWh_T_nom"  : prof["E_total_kWh_T"],
            "E_kWh_T_cons" : prof_cons["E_total_kWh_T"],
            "P_final_nom"  : deP_rp["P_final"],
            "P_final_wc"   : deP_i["P_final"],
            "k_LF_used"    : round(k_LF_i, 6),
            "T_op_C"       : round(T_op,   1),
            "feasible"     : feasible,
            "prof"         : prof,
        })

    feas    = [r for r in results if r["feasible"]]
    optimal = min(feas, key=lambda r: r["TAT"]) if feas else None

    if verbose and optimal:
        _print_report(M, N, optimal, deP_nom, deP_wc, k_info, t_IF_deP)
    elif verbose and not feas:
        best = min(results, key=lambda r: r["TAT"]) if results else None
        print(f"\n  ⚠  No fully feasible solution for {M} IF + {N} LF.")
        if best:
            print(f"     Closest: TAT={best['TAT']:.1f} min, "
                  f"E={best['E_kWh_T_nom']:.1f} kWh/T, "
                  f"%P={best['P_final_nom']*100:.4f}%")

    return {
        "M": M, "N": N,
        "optimal"        : optimal,
        "feasible"       : bool(feas),
        "feasible_range" : (feas[0]["T_LF"], feas[-1]["T_LF"]) if feas else None,
        "k_info"         : k_info,
        "deP_nominal"    : deP_nom,
        "deP_worst_case" : deP_wc,
        "t_IF_deP"       : t_IF_deP,
        "all_results"    : results,
    }


def _print_report(M, N, opt, deP_nom, deP_wc, k_info, t_IF_deP):
    """Structured console report for one M+N configuration."""
    p        = PARAMS
    W        = 68
    opt_prof = opt["prof"]

    print(f"\n{'═' * W}")
    print(f"  CASE: {M} IF  +  {N} LF")
    print(f"{'═' * W}")

    # Scheduling
    print(f"\n  ┌─ SCHEDULING  (Erlang-C queuing model) {'─'*27}┐")
    print(f"  │  Heat arrival rate λ       = {M/p['T_IF']:.4f} heats/min  ({p['T_IF']} min cycle)  │")
    print(f"  │  Optimal LF service time   = {opt['T_LF']:.1f} min                              │")
    print(f"  │  Server utilisation ρ      = {opt['rho']:.4f}   {'✓ stable' if opt['rho'] < 1 else '✗ UNSTABLE'}               │")
    print(f"  │  Expected queue wait E[Wq] = {opt['E_Wq']:.2f} min  {'✓' if opt['E_Wq'] <= 15 else '✗'} (limit 15 min)        │")
    print(f"  └{'─'*(W-4)}┘")

    # Thermal
    print(f"\n  ┌─ THERMAL PROFILE  (T_tap = 1 660 °C, confirmed) {'─'*16}┐")
    print(f"  │  Stage                     Temperature    Δ from previous       │")
    print(f"  │  {'─'*60}  │")
    print(f"  │  After tapping (IF)        1 660 °C       —  (fixed, confirmed) │")
    print(f"  │  Arrive at LF (−50 °C)     {opt['T_arr_LF']:.0f} °C       −50 °C empirical    │")
    print(f"  │  After queue wait          {opt['T_after_wait']:.0f} °C       −{opt['T_arr_LF']-opt['T_after_wait']:.1f} °C ({opt['E_Wq']:.1f} min)   │")
    print(f"  │  Exit LF  (arc heating ↑)  {opt['T_exit_LF']:.0f} °C       +{opt['delta_T_arc']:.1f} °C arc ΔT      │")
    print(f"  │  Arrive at caster          {opt['T_at_caster']:.0f} °C       (need ≥ {p['T_cast_req']:.0f} °C)        │")
    cast_ok = "✓ MET" if opt['T_at_caster'] >= p['T_cast_req'] - 0.5 else "✗ NOT MET"
    print(f"  │  Casting temperature:  {cast_ok}   (target 1625 °C)                  │")
    print(f"  └{'─'*(W-4)}┘")

    # Phosphorus
    print(f"\n  ┌─ PHOSPHORUS KINETICS {'─'*44}┐")
    print(f"  │  Enhanced k_LF @ {opt['T_op_C']:.0f} °C      = {opt['k_LF_used']:.5f} min⁻¹         │")
    print(f"  │  Enhancement factor vs baseline        = ×{opt['k_LF_used']/p['k_LF_base']:.1f}                   │")
    print(f"  │  IF pre-treatment time                 = {t_IF_deP} min                    │")
    p_nom = opt['P_final_nom'] * 100
    p_wc  = opt['P_final_wc']  * 100
    print(f"  │  Final %P  (P_init = 0.09%, nominal)   = {p_nom:.4f}%  {'✓' if p_nom <= 4.001 else '✗'} (limit 0.040%)   │")
    print(f"  │  Final %P  (P_init = 0.10%, worst-case)= {p_wc:.4f}%  {'✓' if p_wc  <= 4.001 else '✗'} (limit 0.040%)   │")
    print(f"  └{'─'*(W-4)}┘")

    # Energy — 5 components
    print(f"\n  ┌─ ARC ENERGY — 5-COMPONENT BREAKDOWN {'─'*27}┐")
    comp_labels = [
        ("Steel sensible heat  [1]", opt_prof['E_steel_kWh_T']),
        ("Slag heating 4% mass [2]", opt_prof['E_slag_kWh_T']),
        ("Refractory absorption[3]", opt_prof['E_ref_kWh_T']),
        ("Dynamic losses (rad) [4]", opt_prof['E_loss_kWh_T']),
        ("DeP reaction credit  [5]", opt_prof['E_rxn_kWh_T']),
    ]
    for lbl, val in comp_labels:
        print(f"  │    {lbl}   {val:>6.2f} kWh/T              │")
    print(f"  │  {'─'*60}  │")
    print(f"  │    Arc efficiency η = {opt_prof['eta_used']:.2f}  (normal slag cover)                  │")
    print(f"  │    Physics total  (÷η)               = {opt['E_kWh_T_nom']:>6.2f} kWh/T  {'✓' if opt['E_kWh_T_nom'] < 100 else '✗'}         │")
    print(f"  │    Conservative  (13T + 1630°C)      = {opt['E_kWh_T_cons']:>6.2f} kWh/T  {'✓' if opt['E_kWh_T_cons'] < 100 else '✗'}         │")
    if opt_prof['E_Pt_kWh_T'] is not None:
        print(f"  │    P×t cross-check                   = {opt_prof['E_Pt_kWh_T']:>6.2f} kWh/T              │")
    else:
        print(f"  │    P×t cross-check                   = N/A  (await transformer rating, Q7)│")
    print(f"  └{'─'*(W-4)}┘")

    # Final result
    print(f"\n  ┌─ FINAL RESULT {'─'*50}┐")
    all_ok = (
        opt["TAT"] <= 40 and
        opt["P_final_wc"] <= 0.040 + 1e-5 and
        opt["E_kWh_T_cons"] < 100 and
        opt["T_at_caster"] >= p["T_cast_req"] - 0.5
    )
    verdict = "ALL CONSTRAINTS SATISFIED ✓" if all_ok else "REVIEW FLAGGED CONSTRAINTS ✗"
    print(f"  │  Minimum feasible T_LF   = {opt['T_LF']:.1f} min                          │")
    print(f"  │  Total TAT               = {opt['TAT']:.2f} min     (limit 40 min)              │")
    print(f"  │  TAT headroom            = {40 - opt['TAT']:.2f} min remaining                    │")
    print(f"  │  {verdict:<64}│")
    print(f"  └{'─'*(W-4)}┘")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE S — MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_plant(M, N, T_LF_opt, n_heats=2000, seed=42):
    """
    2 000-heat stochastic simulation. Variability: Erlang-3 arrivals,
    log-normal service (CV=0.20), random tonnage / %P / Zone-B rate.
    T_tap fixed at 1 660 °C. Returns aggregate statistics + raw arrays.
    """
    rng = np.random.default_rng(seed)
    p   = PARAMS
    lam = M / p["T_IF"]

    k_arr = p["arrival_erlang_k"]
    inter = rng.gamma(shape=k_arr, scale=1.0/(k_arr*lam), size=n_heats)
    arr_t = np.cumsum(inter)

    cv      = p["service_cv"]
    s_mu    = math.log(T_LF_opt) - 0.5 * math.log(1 + cv**2)
    s_sigma = math.sqrt(math.log(1 + cv**2))
    svc_t   = rng.lognormal(s_mu, s_sigma, n_heats)

    m_heats   = rng.uniform(p["m_heat_min"],   p["m_heat_max"],   n_heats)
    P_inits   = rng.uniform(p["P_init_lo"],    p["P_init_hi"],    n_heats)
    dT_B_rats = rng.uniform(p["dT_zone_B_lo"], p["dT_zone_B_hi"], n_heats)

    lf_free = np.zeros(N)
    TATs, waits, energies, P_fins, temp_exits = [], [], [], [], []
    hold_v = E_v = P_v = T_v = 0

    for i in range(n_heats):
        lf_idx  = int(np.argmin(lf_free))
        t_start = max(arr_t[i], lf_free[lf_idx])
        wait    = t_start - arr_t[i]
        lf_free[lf_idx] = t_start + svc_t[i]
        if wait > p["W_hold_max"]:
            hold_v += 1

        prof_i = thermal_profile(svc_t[i], wait, m_heat=m_heats[i],
                                 dT_B_rate=dT_B_rats[i])
        T_op_i = (prof_i["T_arr_LF"] + prof_i["T_exit_LF"]) / 2
        k_i    = compute_kLF_enhanced(T_LF_C=T_op_i)["k_LF_enhanced"]
        deP_i  = two_stage_deP(P_inits[i], p["P_target"], p["k_IF"], 40, k_i)

        TAT_i = p["T_tap_op"] + p["T_tr1"] + wait + svc_t[i] + p["T_tr2"]
        TATs.append(TAT_i)
        waits.append(wait)
        energies.append(prof_i["E_total_kWh_T"])
        P_fins.append(deP_i["P_final"])
        temp_exits.append(prof_i["T_at_caster"])

        if deP_i["P_final"]         > p["P_target"] + 1e-5: P_v += 1
        if prof_i["E_total_kWh_T"] >= p["E_LF_limit"]:      E_v += 1
        if not prof_i["cast_temp_ok"]:                        T_v += 1

    TATs      = np.array(TATs)
    energies  = np.array(energies)
    P_fins    = np.array(P_fins)
    temp_exits= np.array(temp_exits)

    return {
        "n_heats"        : n_heats,
        "mean_TAT"       : round(float(np.mean(TATs)),                3),
        "std_TAT"        : round(float(np.std(TATs)),                 3),
        "p10_TAT"        : round(float(np.percentile(TATs, 10)),      2),
        "p50_TAT"        : round(float(np.median(TATs)),              2),
        "p90_TAT"        : round(float(np.percentile(TATs, 90)),      2),
        "p95_TAT"        : round(float(np.percentile(TATs, 95)),      2),
        "P_TAT_leq_40"   : round(float(np.mean(TATs <= 40)),          4),
        "mean_wait"      : round(float(np.mean(waits)),               3),
        "hold_violations": hold_v,
        "hold_viol_rate" : round(hold_v / n_heats,                    4),
        "mean_E_kWh_T"   : round(float(np.mean(energies)),            2),
        "max_E_kWh_T"    : round(float(np.max(energies)),             2),
        "E_violations"   : E_v,
        "E_viol_rate"    : round(E_v / n_heats,                       4),
        "mean_P_pct"     : round(float(np.mean(P_fins)) * 100,        4),
        "max_P_pct"      : round(float(np.max(P_fins)) * 100,         4),
        "P_violations"   : P_v,
        "P_viol_rate"    : round(P_v / n_heats,                       4),
        "temp_violations": T_v,
        "temp_viol_rate" : round(T_v / n_heats,                       4),
        "raw_TATs"       : TATs.tolist(),
        "raw_energies"   : energies.tolist(),
        "raw_P_fins"     : (P_fins * 100).tolist(),
        "raw_temp_exits" : temp_exits.tolist(),
        "raw_waits"      : np.array(waits).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE G — GENERAL M+N SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def general_MN_analysis(M_range=None, T_LF_target=12):
    if M_range is None:
        M_range = range(1, 40)
    p   = PARAMS
    out = []
    for M in M_range:
        N_min = math.ceil((M * T_LF_target) / (p["T_IF"] * (1 - p["rho_max"])))
        qs    = queue_stats(M, N_min, T_LF_target)
        E_Wq  = qs["E_Wq_analytical"] if qs["stable"] else float("inf")
        prof  = thermal_profile(T_LF_target, min(E_Wq, 15))
        TAT   = p["T_tap_op"] + p["T_tr1"] + E_Wq + T_LF_target + p["T_tr2"]
        out.append({
            "M"       : M,
            "N_min"   : N_min,
            "rho"     : qs.get("rho"),
            "E_Wq"    : round(E_Wq, 2) if math.isfinite(E_Wq) else None,
            "TAT"     : round(TAT,  2) if qs["stable"] else None,
            "T_exit"  : prof["T_exit_LF"],
            "E_kWh_T" : prof["E_total_kWh_T"],
            "feasible": (qs["stable"] and TAT <= 40 and prof["cast_temp_ok"]
                         and prof["E_total_kWh_T"] < p["E_LF_limit"]),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE V — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

STEEL_BLUE = "#1B4F72"
ORANGE     = "#D35400"
GREEN      = "#1E8449"
GREY       = "#707070"
RED        = "#C0392B"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F7F9FC",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "grid.color"        : "white",
    "grid.linewidth"    : 0.8,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.9,
})


def plot_erlang_c(c1, c2):
    """Plot 1 — Erlang-C queue wait and utilisation curves for both cases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Erlang-C Queue Analysis — LF Scheduling",
                 fontsize=14, fontweight="bold", y=1.01)

    for res, title, ax in [(c1, "Case 1 : 8 IF + 2 LF", axes[0]),
                            (c2, "Case 2 : 4 IF + 1 LF", axes[1])]:
        T_LFs = np.arange(5, 40, 0.2)
        rhos  = [queue_stats(res["M"], res["N"], T)["rho"]
                 if queue_stats(res["M"], res["N"], T)["stable"] else np.nan
                 for T in T_LFs]
        wqs   = [min(queue_stats(res["M"], res["N"], T)["E_Wq_analytical"], 30)
                 if queue_stats(res["M"], res["N"], T)["stable"] else np.nan
                 for T in T_LFs]
        ax2 = ax.twinx()
        ax.plot(T_LFs, wqs,  color=STEEL_BLUE, lw=2.0, label="E[Wq] (min)")
        ax2.plot(T_LFs, rhos, color=ORANGE,    lw=1.5, ls="--", label="ρ utilisation")
        if res["optimal"]:
            opt = res["optimal"]
            ax.axvline(opt["T_LF"], color=GREEN, lw=1.5, ls=":")
            ax.scatter([opt["T_LF"]], [opt["E_Wq"]], color=GREEN, s=70, zorder=5,
                       label=f"Optimal T_LF = {opt['T_LF']:.1f} min")
            ax.annotate(
                f"T_LF={opt['T_LF']:.1f}\nE[Wq]={opt['E_Wq']:.1f}\nρ={opt['rho']:.3f}",
                xy=(opt["T_LF"], opt["E_Wq"]),
                xytext=(opt["T_LF"]+3, opt["E_Wq"]+3),
                fontsize=8.5, color=STEEL_BLUE,
                arrowprops=dict(arrowstyle="->", color=STEEL_BLUE, lw=1.0))
        ax.axhline(15, color=RED, lw=1.0, ls=":", alpha=0.7, label="Hold limit 15 min")
        ax2.axhline(0.40, color=ORANGE, lw=1.0, ls=":", alpha=0.6, label="ρ advisory 0.40")
        ax.set_xlabel("LF Service Time T_LF (min)")
        ax.set_ylabel("Expected Queue Wait E[Wq]  (min)", color=STEEL_BLUE)
        ax2.set_ylabel("Server Utilisation  ρ", color=ORANGE)
        ax.set_ylim(0, 25); ax2.set_ylim(0, 1.2)
        ax.tick_params(axis="y", labelcolor=STEEL_BLUE)
        ax2.tick_params(axis="y", labelcolor=ORANGE)
        ax.set_title(title); ax.grid(True, lw=0.6)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, loc="upper right", fontsize=8)

    plt.tight_layout()
    path = "plots/01_erlang_c_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_thermal_profile(c1, c2):
    """Plot 2 — Temperature waterfall tap → LF → caster for both cases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Steel Temperature Profile Through Process Route",
                 fontsize=14, fontweight="bold")

    for res, title, ax in [(c1, "Case 1: 8 IF + 2 LF", axes[0]),
                            (c2, "Case 2: 4 IF + 1 LF", axes[1])]:
        if not res["optimal"]:
            ax.text(0.5, 0.5, "No feasible solution", ha="center", va="center",
                    transform=ax.transAxes); continue
        opt = res["optimal"]
        p   = PARAMS
        t0, t1 = 0, p["T_tap_op"] + p["T_tr1"]
        t2, t3 = t1 + opt["E_Wq"], t1 + opt["E_Wq"] + opt["T_LF"]
        t4 = t3 + p["T_tr2"]
        times  = [t0, t1, t2, t3, t4]
        temps  = [p["T_tap"], opt["T_arr_LF"], opt["T_after_wait"],
                  opt["T_exit_LF"], opt["T_at_caster"]]
        labels = ["Tap\n(IF)", "Arrive\nLF", "Arc\nStart", "Exit\nLF", "At\nCaster"]
        colours = [GREY, GREY, GREEN, GREY]
        for i in range(4):
            ax.plot([times[i], times[i+1]], [temps[i], temps[i+1]],
                    color=colours[i], lw=2.5)
            ax.fill_between([times[i], times[i+1]], [temps[i], temps[i+1]],
                            1580, alpha=0.08, color=colours[i])
        ax.scatter(times, temps, color=[STEEL_BLUE]*5, s=70, zorder=5)
        for t, T in zip(times, temps):
            ax.annotate(f"{T:.0f}°C", xy=(t, T), xytext=(t, T+4),
                        ha="center", fontsize=8.5, fontweight="bold", color=STEEL_BLUE)
        ax.axhline(p["T_cast_req"], color=RED, lw=1.2, ls="--",
                   label=f"Min casting {p['T_cast_req']} °C")
        ax.axhline(p["T_tap"], color=ORANGE, lw=1.0, ls=":", alpha=0.6,
                   label=f"Tap {p['T_tap']} °C")
        ax.set_xticks(times); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Steel Temperature (°C)"); ax.set_xlabel("Process Stage")
        ax.set_title(title); ax.set_ylim(1590, 1680)
        ax.legend(fontsize=8, loc="lower left"); ax.grid(True, axis="y", lw=0.6)
        mid_arc = (t2 + t3) / 2
        ax.annotate("", xy=(t3, opt["T_exit_LF"]), xytext=(t2, opt["T_after_wait"]),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
        ax.text(mid_arc, (opt["T_after_wait"] + opt["T_exit_LF"]) / 2 + 5,
                f"+{opt['delta_T_arc']:.0f}°C\n(arc)", ha="center",
                fontsize=8, color=GREEN, fontweight="bold")

    plt.tight_layout()
    path = "plots/02_thermal_profile.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_energy_breakdown(c1, c2):
    """Plot 3 — Stacked 5-component energy bars + arc efficiency sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    fig.suptitle("LF Arc Energy — 5-Component Breakdown",
                 fontsize=14, fontweight="bold")

    comp_names   = ["Steel\nSensible Heat", "Slag\nHeating",
                    "Refractory\nAbsorption", "Dynamic\nRadiation",
                    "DeP Reaction\n(credit)"]
    comp_colours = [STEEL_BLUE, "#2E86C1", "#1ABC9C", "#F39C12", GREEN]

    for ax, res, title in [(axes[0], c1, "Case 1\n8 IF + 2 LF"),
                            (axes[1], c2, "Case 2\n4 IF + 1 LF")]:
        if not res["optimal"]:
            ax.text(0.5, 0.5, "No feasible\nsolution", ha="center",
                    va="center", transform=ax.transAxes); continue
        prof = res["optimal"]["prof"]
        vals = [prof["E_steel_kWh_T"], prof["E_slag_kWh_T"],
                prof["E_ref_kWh_T"],   prof["E_loss_kWh_T"],
                prof["E_rxn_kWh_T"]]
        bottom_pos = 0
        for v, name, colour in zip(vals, comp_names, comp_colours):
            if v >= 0:
                ax.bar(0, v, bottom=bottom_pos, color=colour, edgecolor="white",
                       linewidth=0.8, label=name, width=0.5)
                if v > 1.0:
                    ax.text(0, bottom_pos + v/2, f"{v:.1f}", ha="center",
                            va="center", fontsize=8, color="white", fontweight="bold")
                bottom_pos += v
            else:
                ax.bar(0, v, bottom=0, color=colour, edgecolor="white",
                       linewidth=0.8, label=name, width=0.5, alpha=0.8)
                ax.text(0, v/2, f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        ax.axhline(100, color=RED, lw=1.5, ls="--", label="100 kWh/T limit")
        ax.axhline(prof["E_total_kWh_T"], color=ORANGE, lw=1.5, ls="-.",
                   label=f"Total = {prof['E_total_kWh_T']:.1f} kWh/T")
        ax.set_xticks([]); ax.set_ylabel("Arc Energy  (kWh/T)")
        ax.set_title(title); ax.set_ylim(-10, 110)
        ax.legend(fontsize=7.5, loc="upper right"); ax.grid(axis="y", lw=0.6)

    ax3 = axes[2]
    if c1.get("optimal"):
        prof = c1["optimal"]["prof"]
        raw  = (prof["E_steel_kWh_T"] + prof["E_slag_kWh_T"] +
                prof["E_ref_kWh_T"]   + prof["E_loss_kWh_T"] +
                prof["E_rxn_kWh_T"])
        etas        = [0.62, 0.68, 0.75, 0.78, 0.82]
        eta_labels  = ["Poor\n(open bath)", "Below\nnominal", "Nominal\nslag",
                       "Improved\nslag", "Foamy\nslag (best)"]
        vals        = [raw / e for e in etas]
        col_bar     = [RED, ORANGE, "#F1C40F", "#82E0AA", GREEN]
        bars = ax3.bar(eta_labels, vals, color=col_bar, edgecolor="white", lw=0.8)
        ax3.axhline(100, color=RED, lw=1.5, ls="--", label="100 kWh/T limit")
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2, val+0.8,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold")
        ax3.set_ylabel("Arc Energy  (kWh/T)")
        ax3.set_title("Case 1: Arc Efficiency Sensitivity\n(same ΔT, different slag practice)")
        ax3.set_ylim(0, 115); ax3.legend(fontsize=9); ax3.grid(axis="y", lw=0.6)

    plt.tight_layout()
    path = "plots/03_energy_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_monte_carlo(sim1, sim2, c1, c2):
    """Plot 4 — TAT histogram, CDF, energy scatter, %P violin for 2 000 heats."""
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.38)
    fig.suptitle("Monte Carlo Simulation — 2 000 Heats per Case\n"
                 "(Erlang-3 arrivals, log-normal service, stochastic tonnage / %P / cooling rate)",
                 fontsize=13, fontweight="bold")

    for row, (sim, res, case_label, colour) in enumerate([
        (sim1, c1, "Case 1: 8 IF + 2 LF", STEEL_BLUE),
        (sim2, c2, "Case 2: 4 IF + 1 LF", ORANGE),
    ]):
        TATs     = np.array(sim["raw_TATs"])
        energies = np.array(sim["raw_energies"])
        P_fins   = np.array(sim["raw_P_fins"])

        ax1 = fig.add_subplot(gs[row, 0])
        ax1.hist(TATs, bins=40, color=colour, edgecolor="white", lw=0.4, alpha=0.85)
        ax1.axvline(40, color=RED, lw=1.5, ls="--", label="40 min limit")
        ax1.axvline(sim["mean_TAT"], color="black", lw=1.2,
                    label=f"Mean {sim['mean_TAT']:.1f} min")
        ax1.axvline(sim["p95_TAT"], color=ORANGE, lw=1.2, ls="-.",
                    label=f"P95 {sim['p95_TAT']:.1f} min")
        ax1.set_xlabel("TAT (min)"); ax1.set_ylabel("Frequency")
        ax1.set_title(f"{case_label}\nTAT Distribution")
        ax1.legend(fontsize=7.5); ax1.grid(lw=0.5)

        ax2 = fig.add_subplot(gs[row, 1])
        sorted_T = np.sort(TATs)
        cdf = np.arange(1, len(sorted_T)+1) / len(sorted_T)
        ax2.plot(sorted_T, cdf*100, color=colour, lw=2.0)
        ax2.axvline(40, color=RED, lw=1.5, ls="--")
        ax2.axhline(sim["P_TAT_leq_40"]*100, color=GREEN, lw=1.2, ls=":",
                    label=f"{sim['P_TAT_leq_40']*100:.1f}% ≤ 40 min")
        ax2.set_xlabel("TAT (min)"); ax2.set_ylabel("Cumulative Probability (%)")
        ax2.set_title("TAT Cumulative Distribution")
        ax2.legend(fontsize=8); ax2.grid(lw=0.5); ax2.set_ylim(0, 102)

        ax3 = fig.add_subplot(gs[row, 2])
        sc = ax3.scatter(range(len(energies)), energies, c=TATs,
                         cmap="RdYlGn_r", s=5, alpha=0.5, rasterized=True)
        ax3.axhline(100, color=RED, lw=1.5, ls="--", label="100 kWh/T limit")
        ax3.axhline(sim["mean_E_kWh_T"], color=GREY, lw=1.0,
                    label=f"Mean {sim['mean_E_kWh_T']:.1f} kWh/T")
        plt.colorbar(sc, ax=ax3, label="TAT (min)", pad=0.02)
        ax3.set_xlabel("Heat number"); ax3.set_ylabel("Arc energy (kWh/T)")
        ax3.set_title("Arc Energy per Heat\n(colour = TAT)")
        ax3.legend(fontsize=7.5); ax3.grid(lw=0.5)

        ax4 = fig.add_subplot(gs[row, 3])
        ax4.violinplot([P_fins], positions=[0], showmedians=True, widths=0.6)
        ax4.axhline(4.0, color=RED, lw=1.5, ls="--", label="0.040% limit")
        viol_count = sim["P_violations"]
        ax4.scatter([0]*viol_count, [v for v in P_fins if v > 4.0],
                    color=RED, s=15, alpha=0.7, zorder=5,
                    label=f"{viol_count} violations ({sim['P_viol_rate']*100:.1f}%)")
        ax4.set_xticks([0]); ax4.set_xticklabels(["Final %P × 100"])
        ax4.set_ylabel("Final %P (×100 shown)")
        ax4.set_title("Phosphorus Distribution\n(violin = density)")
        ax4.legend(fontsize=7.5); ax4.grid(axis="y", lw=0.5)

    plt.savefig("plots/04_monte_carlo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/04_monte_carlo.png")


def plot_sensitivity_combined(c1):
    """Plot 5 — Tonnage, casting temperature, and %P sensitivity for Case 1."""
    if not c1.get("optimal"):
        return
    opt = c1["optimal"]
    p   = PARAMS
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Sensitivity Analysis — Case 1 (8 IF + 2 LF)",
                 fontsize=14, fontweight="bold")

    # Tonnage
    masses    = np.linspace(12, 18, 30)
    E_list    = [thermal_profile(opt["T_LF"], opt["E_Wq"], m_heat=m)["E_total_kWh_T"]
                 for m in masses]
    T_ex_list = [thermal_profile(opt["T_LF"], opt["E_Wq"], m_heat=m)["T_exit_LF"]
                 for m in masses]
    ax = axes[0]; ax2 = ax.twinx()
    ax.plot(masses, E_list, color=STEEL_BLUE, lw=2.0, label="Arc energy (kWh/T)")
    ax2.plot(masses, T_ex_list, color=ORANGE, lw=1.5, ls="--", label="T_exit_LF (°C)")
    ax.axvline(p["m_heat_min"], color=GREY, lw=1.0, ls=":")
    ax.axvline(p["m_heat_max"], color=GREY, lw=1.0, ls=":")
    ax.axhline(100, color=RED, lw=1.2, ls="--", alpha=0.7)
    ax.fill_betweenx([min(E_list)-2, 100], p["m_heat_min"], p["m_heat_max"],
                     alpha=0.10, color=GREEN, label="Operating range")
    ax.set_xlabel("Heat Tonnage (T)"); ax.set_ylabel("Arc Energy (kWh/T)", color=STEEL_BLUE)
    ax2.set_ylabel("T_exit_LF (°C)", color=ORANGE)
    ax.set_title("Heat Tonnage Uncertainty\n(13–17 T practical range)")
    ax.tick_params(axis="y", labelcolor=STEEL_BLUE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper right"); ax.grid(lw=0.5)

    # Casting temperature
    T_casts = np.linspace(1618, 1632, 30)
    E_cast  = [thermal_profile(opt["T_LF"], opt["E_Wq"], T_cast_req=Tc)["E_total_kWh_T"]
               for Tc in T_casts]
    T_ex_c  = [thermal_profile(opt["T_LF"], opt["E_Wq"], T_cast_req=Tc)["T_exit_LF"]
               for Tc in T_casts]
    ax = axes[1]; ax2 = ax.twinx()
    ax.plot(T_casts, E_cast, color=STEEL_BLUE, lw=2.0, label="Arc energy (kWh/T)")
    ax2.plot(T_casts, T_ex_c, color=ORANGE, lw=1.5, ls="--", label="T_exit_LF (°C)")
    ax.axvline(p["T_cast_lo"], color=GREY, lw=1.0, ls=":")
    ax.axvline(p["T_cast_hi"], color=GREY, lw=1.0, ls=":")
    ax.fill_betweenx([min(E_cast)-2, max(E_cast)+2],
                     p["T_cast_lo"], p["T_cast_hi"],
                     alpha=0.10, color=GREEN, label="Operating window")
    ax.axhline(100, color=RED, lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel("Casting Temperature Requirement (°C)")
    ax.set_ylabel("Arc Energy (kWh/T)", color=STEEL_BLUE)
    ax2.set_ylabel("T_exit_LF (°C)", color=ORANGE)
    ax.set_title("Casting Temperature Sensitivity\n(1620–1630°C operating window)")
    ax.tick_params(axis="y", labelcolor=STEEL_BLUE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper left"); ax.grid(lw=0.5)

    # %P sensitivity
    p_inits    = np.linspace(0.070, 0.110, 30)
    k_LF_e     = compute_kLF_enhanced(T_LF_C=opt["T_op_C"])["k_LF_enhanced"]
    t_LF_req   = [two_stage_deP(Pi, p["P_target"], p["k_IF"], 40, k_LF_e)
                  ["t_LF_deP_required"] + 2 for Pi in p_inits]
    P_fin_list = [two_stage_deP(Pi, p["P_target"], p["k_IF"], 40, k_LF_e)
                  ["P_final"] * 100 for Pi in p_inits]
    ax = axes[2]; ax2 = ax.twinx()
    ax.plot(p_inits*100, t_LF_req, color=STEEL_BLUE, lw=2.0,
            label="Min T_LF for deP (min)")
    ax2.plot(p_inits*100, P_fin_list, color=GREEN, lw=1.5, ls="--",
             label="Final %P achieved")
    ax.axvline(p["P_init_lo"]*100, color=GREY, lw=1.0, ls=":")
    ax.axvline(p["P_init_hi"]*100, color=GREY, lw=1.0, ls=":")
    ax.fill_betweenx([0, max(t_LF_req)+2],
                     p["P_init_lo"]*100, p["P_init_hi"]*100,
                     alpha=0.10, color=GREEN, label="Input range 0.08–0.10%")
    ax.axhline(opt["T_LF"], color=ORANGE, lw=1.5, ls="-.",
               label=f"Optimal T_LF = {opt['T_LF']:.1f} min")
    ax2.axhline(4.0, color=RED, lw=1.2, ls="--", alpha=0.7, label="%P limit 0.040%")
    ax.set_xlabel("Initial %P in IF (before LF treatment)")
    ax.set_ylabel("Min T_LF needed for dephosphorisation (min)", color=STEEL_BLUE)
    ax2.set_ylabel("Final %P achieved", color=GREEN)
    ax.set_title("Phosphorus Sensitivity\n(enhanced k_LF with slag + stirring upgrade)")
    ax.tick_params(axis="y", labelcolor=STEEL_BLUE)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper left"); ax.grid(lw=0.5)

    plt.tight_layout()
    path = "plots/05_sensitivity_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_tat_breakdown(case_result):
    """
    Plot 6 — TAT breakdown waterfall for Case 1.
    Each bar = one process stage; cumulative line shows total TAT.
    """
    if not case_result.get("optimal"):
        return
    opt = case_result["optimal"]
    p   = PARAMS

    stages    = ["Tapping\nOp", "IF→LF\nTravel", "Queue\nWait",
                 "LF\nService", "LF→Caster\nTravel"]
    durations = [p["T_tap_op"], p["T_tr1"], opt["E_Wq"], opt["T_LF"], p["T_tr2"]]
    colours   = [ORANGE, GREY, RED, STEEL_BLUE, GREY]

    fig, (ax_bar, ax_wf) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("TAT Breakdown — Case 1 (8 IF + 2 LF)",
                 fontsize=14, fontweight="bold")

    # Left: individual duration bars
    bars = ax_bar.bar(stages, durations, color=colours, edgecolor="white", lw=0.8)
    for bar, dur in zip(bars, durations):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height()/2, f"{dur:.1f}\nmin",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax_bar.axhline(40, color=RED, ls="--", lw=1.5, label="40 min limit")
    ax_bar.set_ylabel("Stage Duration (min)")
    ax_bar.set_title("Per-Stage Duration")
    ax_bar.legend(fontsize=9); ax_bar.grid(axis="y", lw=0.5)

    # Right: cumulative waterfall
    cumulative = [0]
    for d in durations:
        cumulative.append(cumulative[-1] + d)
    total_tat = cumulative[-1]

    x_pos = range(len(stages))
    for i, (stage, dur, col) in enumerate(zip(stages, durations, colours)):
        ax_wf.bar(i, dur, bottom=cumulative[i], color=col,
                  edgecolor="white", lw=0.8, label=stage)
        ax_wf.text(i, cumulative[i] + dur/2, f"{dur:.1f}",
                   ha="center", va="center", fontsize=9,
                   color="white", fontweight="bold")

    ax_wf.plot(range(len(cumulative)-1), cumulative[1:],
               "o--", color="black", lw=1.5, ms=6, label="Cumulative TAT")
    ax_wf.axhline(40, color=RED, ls="--", lw=1.5, label="40 min limit")
    ax_wf.set_xticks(list(x_pos))
    ax_wf.set_xticklabels(stages, fontsize=9)
    ax_wf.set_ylabel("Cumulative TAT (min)")
    ax_wf.set_title("Cumulative TAT Waterfall")
    ax_wf.set_ylim(0, max(total_tat * 1.15, 45))
    ax_wf.text(len(stages)-1, total_tat + 1.0,
               f"Total = {total_tat:.1f} min",
               fontsize=10, fontweight="bold", ha="right")
    ax_wf.legend(fontsize=8, loc="upper left"); ax_wf.grid(axis="y", lw=0.5)

    plt.tight_layout()
    path = "plots/06_tat_breakdown.png"
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = PARAMS

    print("\n" + "█" * 68)
    print("  AMALGAM STEEL × IITM — LADLE TAT OPTIMISATION  v5-final")
    print("  5-Component Industrial Energy Model")
    print("  Final Submission  |  2026-05-20  |  Company data integrated")
    print("█" * 68)

    print("\n" + "░" * 68)
    print("  CASE 1 — 8 Induction Furnaces + 2 Ladle Furnaces")
    print("░" * 68)
    c1 = optimise_configuration(8, 2, verbose=True)

    print("\n" + "░" * 68)
    print("  CASE 2 — 4 Induction Furnaces + 1 Ladle Furnace")
    print("░" * 68)
    c2 = optimise_configuration(4, 1, verbose=True)

    # ── Tonnage uncertainty ───────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  TONNAGE UNCERTAINTY ANALYSIS  (13–17 T practical range)")
    print(f"{'═' * 68}")
    if c1.get("optimal"):
        T_LF1, Ewq1 = c1["optimal"]["T_LF"], c1["optimal"]["E_Wq"]
        ts = tonnage_sensitivity(T_LF1, T_wait=Ewq1)
        print(f"\n  Case 1 — T_LF = {T_LF1:.1f} min,  Queue wait = {Ewq1:.2f} min")
        print(f"  {'Tonnage':>11}  {'T_exit_LF':>10}  {'Arc ΔT':>8}  {'E kWh/T':>9}  {'Cast OK':>8}")
        print(f"  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}")
        for lbl, r in ts.items():
            print(f"  {lbl:>11}  {r['T_exit_LF']:>10.1f}  {r['delta_T_arc']:>8.1f}  "
                  f"{r['E_kWh_T']:>9.2f}  {'✓' if r['cast_ok'] else '✗':>8}")

    # ── Casting temperature sensitivity ───────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  CASTING TEMPERATURE SENSITIVITY  (1620–1630°C operating window)")
    print(f"{'═' * 68}")
    if c1.get("optimal"):
        cs = casting_temp_sensitivity(T_LF1, T_wait=Ewq1)
        print(f"\n  Case 1 — T_LF = {T_LF1:.1f} min")
        print(f"  {'T_cast_req':>11}  {'T_exit_LF':>10}  {'Arc ΔT':>8}  {'E kWh/T':>9}")
        print(f"  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*9}")
        for lbl, r in cs.items():
            print(f"  {r['T_cast_req']:>11.0f}  {r['T_exit_LF']:>10.1f}  "
                  f"{r['delta_T_arc']:>8.1f}  {r['E_kWh_T']:>9.2f}")

    # ── Phosphorus sensitivity ────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  PHOSPHORUS SENSITIVITY  (P_init range: 0.08–0.10%)")
    print(f"{'═' * 68}")
    if c1.get("optimal"):
        ps = P_init_sensitivity(T_LF1, t_IF_deP=40, T_op_C=c1["optimal"]["T_op_C"])
        print(f"\n  Case 1 — T_LF = {T_LF1:.1f} min,  k_LF enhanced")
        print(f"  {'P_init':>10}  {'P_afterIF':>10}  {'t_deP':>8}  {'P_final':>9}  "
              f"{'Met':>5}  {'Fits TLF':>9}")
        print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*9}")
        for lbl, r in ps.items():
            print(f"  {r['P_init']:>10.3f}  {r['P_after_IF']:>9.5f}%  "
                  f"{r['t_LF_deP']:>8.2f}  {r['P_final']*100:>8.4f}%  "
                  f"{'✓' if r['met'] else '✗':>5}  {'✓' if r['fits_in_TLF'] else '✗':>9}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  MONTE CARLO SIMULATION  (2 000 heats each case)")
    print(f"{'═' * 68}")
    sim1 = sim2 = None
    for label, M, N, res in [("8 IF + 2 LF", 8, 2, c1), ("4 IF + 1 LF", 4, 1, c2)]:
        if not res.get("optimal"):
            print(f"\n  {label}: no feasible solution — simulation skipped.")
            continue
        T_LF_opt = res["optimal"]["T_LF"]
        sim = simulate_plant(M, N, T_LF_opt, n_heats=2000)
        if M == 8: sim1 = sim
        else:      sim2 = sim
        print(f"\n  ── {label}  (T_LF = {T_LF_opt:.1f} min) {'─'*30}")
        print(f"    Mean TAT           : {sim['mean_TAT']:.1f} min  ± {sim['std_TAT']:.1f} min")
        print(f"    P10 / P50 / P90    : {sim['p10_TAT']:.1f} / {sim['p50_TAT']:.1f} / {sim['p90_TAT']:.1f} min")
        print(f"    95th percentile TAT: {sim['p95_TAT']:.1f} min")
        print(f"    P(TAT ≤ 40 min)    : {sim['P_TAT_leq_40']*100:.1f}%")
        print(f"    Queue hold viol.   : {sim['hold_violations']}/{sim['n_heats']}  ({sim['hold_viol_rate']*100:.1f}%)")
        print(f"    Energy mean / max  : {sim['mean_E_kWh_T']:.1f} / {sim['max_E_kWh_T']:.1f} kWh/T  (violations: {sim['E_viol_rate']*100:.1f}%)")
        print(f"    %P violations      : {sim['P_violations']}/{sim['n_heats']}  ({sim['P_viol_rate']*100:.1f}%)")
        print(f"    Casting temp viol. : {sim['temp_violations']}/{sim['n_heats']}  ({sim['temp_viol_rate']*100:.1f}%)")

    # ── General M+N scaling ───────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  GENERAL SCALING TABLE  (T_tap = 1 660 °C, T_cast = 1 625 °C)")
    print(f"{'═' * 68}")
    print(f"  {'M':>3}  {'N':>4}  {'ρ':>7}  {'E[Wq]':>7}  {'TAT':>6}  "
          f"{'T_exit':>7}  {'E kWh/T':>9}  {'Feasible':>8}")
    print(f"  {'─'*3}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*8}")
    for r in general_MN_analysis():
        if r["M"] > 4 and r["M"] % 2 != 0:
            continue
        rho = f"{r['rho']:.3f}" if r["rho"] else "  ---"
        wq  = f"{r['E_Wq']:.1f}" if r["E_Wq"] else "  ---"
        tat = f"{r['TAT']:.1f}"  if r["TAT"]  else "  ---"
        print(f"  {r['M']:>3}  {r['N_min']:>4}  {rho:>7}  {wq:>7}  {tat:>6}  "
              f"{r['T_exit']:>7.1f}  {r['E_kWh_T']:>9.2f}  {'✓' if r['feasible'] else '✗':>8}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'█' * 68}")
    print("  FINAL SUMMARY — IMPROVEMENT RECOMMENDATIONS & CONDITIONS")
    print(f"{'█' * 68}")

    for case_name, res in [("Case 1 (8 IF + 2 LF)", c1), ("Case 2 (4 IF + 1 LF)", c2)]:
        print(f"\n  {'═' * 64}")
        print(f"  {case_name}")
        print(f"  {'─' * 64}")
        if not res.get("optimal"):
            print(f"  ✗  No feasible solution found with current constraints.")
            print(f"     → Consider adding an LF or reducing IF count in this bay.")
            continue
        opt = res["optimal"]
        flo, fhi = res["feasible_range"]
        print(f"  Optimal T_LF  :  {opt['T_LF']:.1f} min  (feasible window: {flo:.1f}–{fhi:.1f} min)")
        print(f"  Total TAT     :  {opt['TAT']:.1f} min  (headroom: {40 - opt['TAT']:.1f} min)")
        print(f"  Arc energy    :  {opt['E_kWh_T_nom']:.1f} kWh/T  (limit: 100 kWh/T)")
        print(f"  Final %P      :  {opt['P_final_wc']*100:.4f}%  (worst-case P_init = 0.10%)")
        print(f"  T_at_caster   :  {opt['T_at_caster']:.0f} °C  (required ≥ {p['T_cast_req']:.0f} °C)")
        print()
        print(f"  CHANGES REQUIRED vs CURRENT PRACTICE:")
        print(f"  ┌────────────────────────────────────────────────────────────┐")
        print(f"  │ 1. SLAG CHEMISTRY                                          │")
        print(f"  │    Current: CaO/SiO₂ basicity B ≈ 2.5                     │")
        print(f"  │    Required: B ≥ 4.5  (increase CaO addition by ~80%)     │")
        print(f"  │    Impact: k_LF × {opt['k_LF_used']/p['k_LF_base']:.1f}x → cuts deP time significantly  │")
        print(f"  │                                                            │")
        print(f"  │ 2. ARGON BOTTOM STIRRING                                   │")
        print(f"  │    Current: ~0.1 L/min/T                                   │")
        print(f"  │    Required: ≥ 0.4 L/min/T  (4× increase)                 │")
        print(f"  │    Impact: mass-transfer enhancement (Mazumdar-Evans)      │")
        print(f"  │                                                            │")
        print(f"  │ 3. FeO IN SLAG                                             │")
        print(f"  │    Current: ~5% FeO                                        │")
        print(f"  │    Required: ~18% FeO  (promotes P oxidation)             │")
        print(f"  │                                                            │")
        print(f"  │ 4. LADLE PRE-HEATING                                       │")
        print(f"  │    Required: inner face ≥ 900 °C before every heat        │")
        print(f"  │    Thermal benefit implicit in E_ref_nominal = 8 kWh/T    │")
        print(f"  │                                                            │")
        print(f"  │ 5. IF PRE-TREATMENT  (40 min dephosphorisation)            │")
        print(f"  │    Ensure IF slag at high basicity for full 40 min        │")
        print(f"  └────────────────────────────────────────────────────────────┘")
        print()
        print(f"  PENDING CONFIRMATIONS FROM PLANT:")
        print(f"    Q1. Casting temperature per grade — CRITICAL (each ±5°C ≈ ±1.5 kWh/T)")
        print(f"    Q7. LF transformer MVA rating — needed for P×t cross-check")
        print(f"    Q3. Current Ar stirring rate (L/min/T) — affects k_LF calibration")

    print(f"\n  {'─' * 64}")
    print(f"  CASE 3: GENERAL SCALING GUIDANCE")
    print(f"  {'─' * 64}")
    print(f"    • Each additional IF requires ~0.25 LF capacity")
    print(f"    • Below M = 4 IFs, a single LF is sufficient")
    print(f"    • Above M = 8 IFs, a third LF should be evaluated")
    print(f"    • TAT constraint (40 min) drives the design; energy not binding")
    print(f"    • Maintain ρ < 0.40 — Monte Carlo confirms P(TAT>40) ≈ 0 when ρ < 0.30")

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print("  GENERATING PLOTS  →  ./plots/")
    print(f"{'═' * 68}")
    plot_erlang_c(c1, c2)
    plot_thermal_profile(c1, c2)
    plot_energy_breakdown(c1, c2)
    _sim1 = sim1 if sim1 is not None else {
        "raw_TATs":[],"raw_energies":[],"raw_P_fins":[],"raw_temp_exits":[],
        "raw_waits":[],"mean_TAT":0,"std_TAT":0,"p10_TAT":0,"p50_TAT":0,
        "p90_TAT":0,"p95_TAT":0,"P_TAT_leq_40":0,"mean_E_kWh_T":0,
        "P_violations":0,"P_viol_rate":0,"hold_violations":0,"hold_viol_rate":0,
        "E_viol_rate":0,"temp_violations":0,"temp_viol_rate":0,
        "n_heats":0,"max_E_kWh_T":0}
    _sim2 = sim2 if sim2 is not None else _sim1
    if sim1 is not None:
        plot_monte_carlo(_sim1, _sim2, c1, c2)
    plot_sensitivity_combined(c1)
    plot_tat_breakdown(c1)
    print(f"\n  All plots saved to ./plots/")
    print(f"  ✓  v5-final run complete.\n")

    return c1, c2, sim1, sim2


if __name__ == "__main__":
    results = main()