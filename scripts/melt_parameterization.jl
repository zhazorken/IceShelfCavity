# =====================================================================================
#  Ice–ocean melt parameterization — max(shear, convective), resolution-independent.
#
#  Follows Wild et al. (v2.2) / Zhao et al. 2024 (GRL): the melt rate is the MAXIMUM of a
#  shear-driven three-equation estimate and a buoyancy-driven (convective) estimate — "which
#  process dominates the turbulent transfer of heat and salt at the boundary layer."
#
#    • SHEAR:      classic 3-equation thermodynamics with γ = Γ·u★ and a friction velocity
#                  u★ from the near-ice velocity. To make this RESOLUTION-INDEPENDENT we do
#                  not use √(Cd)·|U₁| directly (|U₁| is the value at the first cell, which
#                  moves with Δz); instead we use the log-law friction velocity
#                        u★_shear = κ |U₁| / ln(z₁/z₀),
#                  which is the (grid-invariant) wall-stress property of the log layer and
#                  recovers √(Cd)·U at a fixed reference height z_ref (Cd = 0.0022 there).
#    • CONVECTIVE: the buoyancy-driven, velocity-INDEPENDENT Kerr & McConnochie scaling
#                        ṁ_conv = c_B · sgn(ΔT) · |ΔT|^{4/3},
#                  expressed here as an equivalent friction velocity u★_conv ∝ |ΔT|^{1/3} so
#                  the same 3-equation solver produces it. This is what keeps melt realistic
#                  when the plume is slow (e.g. during spin-up) — it does not vanish at U→0.
#
#  The two are combined as u★ = max(u★_shear, u★_conv); momentum drag (the τ BCs in the
#  driver) is SEPARATE and still uses the roughness drag.
#
#  Constants are the Wild et al. / Davis et al. (2023, Thwaites East Ice Shelf) set; they give
#  Pine-Island-ballpark melt (≈ 3–50 m/yr across the cavity, higher on the deep warm steep
#  grounding line). --melt_Cd sets the shear drag (default 0.0022).
#
#  SCOPE / next refinement: the convective term here uses the base (vertical-wall) c_B; making
#  it explicitly basal-slope-dependent (McConnochie & Kerr) is the earmarked improvement.
# =====================================================================================

using Oceananigans.Units

#+++ Physical constants (Wild et al. v2.2 / Davis et al. 2023, Thwaites East)
const cᵖʷ = 3974.0      # seawater specific heat        [J kg⁻¹ K⁻¹]
const Lᶠ  = 3.35e5      # latent heat of fusion         [J kg⁻¹]
const ρʷ  = 1027.25     # reference seawater density     [kg m⁻³]
const ρⁱ  = 916.0       # ice density                    [kg m⁻³]

# Linear liquidus  T_f = λ₁·S + λ₂ + λ₃·z   (z negative downward)
const λ₁  = -5.73e-2    # salinity coefficient           [°C psu⁻¹]
const λ₂  =  8.32e-2    # offset                         [°C]
const λ₃  =  7.61e-4    # depth coefficient              [°C m⁻¹]

# Turbulent transfer coefficients (Stanton numbers), γ = Γ·u★
const Γᵀ  =  0.0235     # thermal
const Γˢ  =  6.7e-4     # haline

const c_B    = 2.85e-7  # convective (Kerr) melt constant [m s⁻¹] ; ṁ_conv = c_B·sgn(ΔT)·|ΔT|^{4/3}
const κᵛᵏ    = 0.4      # von Kármán constant (log-law friction velocity)
const z_ref  = 1.0      # reference height [m] at which the shear drag Cd applies
const u★min  = 1.0e-5   # tiny floor on u★ [m s⁻¹] (numerical safety only)
#---

#+++ Basal-slope factor for the convective branch (McConnochie & Kerr sloping-wall convection:
#    steep faces shed meltwater and convect vigorously; near-horizontal bases pool it, suppressing
#    convection). `sfac` multiplies the convective friction velocity; it is precomputed per x on
#    the CPU by the driver via `slope_factor(sinθ, sinθ_ref)` and looked up in the kernel, so the
#    kernel stays cheap and GPU-safe. Normalised to ≈1 at the reference slope; clamped to avoid
#    extremes. Calibrated so steep (~16°) slopes reach ~70 m/yr for warm PIG thermal driving.
const SLOPE_FMIN = 0.3
const SLOPE_FMAX = 3.0
@inline slope_factor(sinθ, sinθ_ref) = clamp(cbrt(max(sinθ, 1e-4) / sinθ_ref), SLOPE_FMIN, SLOPE_FMAX)
#---

#+++ Core solver
"""
    solve_melt(speed, T, S, z, Cd, z1)

Three-equation melt at one ice-base point, using u★ = max(shear, convective):
  • `speed` = |U| = √(u²+v²+w²) at the first ocean cell (height `z1` above the ice);
  • `Cd`    = shear drag coefficient at the reference height z_ref (default 0.0022);
  • the convective branch is velocity-independent (Kerr) so melt stays realistic at low shear.
Returns `(ṁ, T_b, S_b, γᵀ, γˢ)` with ṁ>0 for melting.
"""
@inline function solve_melt(speed, T, S, z, Cd, z1, sfac)
    # resolution-independent shear friction velocity (log law); z₀ chosen so u★=√(Cd)·U at z_ref
    z₀ = z_ref * exp(-κᵛᵏ / sqrt(Cd))
    u★_shear = κᵛᵏ * speed / log(max(z1, 1.001z₀) / z₀)

    # convective (buoyancy-driven) equivalent friction velocity: γᵀ_conv = ρⁱLᶠc_B|ΔT|^{1/3}/(ρʷcᵖʷ),
    # times the basal-slope factor `sfac` (=1 on the reference slope; larger on steep faces).
    ΔT = T - (λ₁ * S + λ₂ + λ₃ * z)                         # thermal driving
    u★_conv = sfac * (ρⁱ * Lᶠ * c_B / (ρʷ * cᵖʷ * Γᵀ)) * cbrt(abs(ΔT))

    u★ = max(u★_shear, u★_conv, u★min)
    γᵀ = Γᵀ * u★
    γˢ = Γˢ * u★

    c₀ = λ₂ + λ₃ * z
    r  = cᵖʷ * γᵀ / Lᶠ
    A  = r * λ₁
    B  = -(γˢ + r * (T - c₀))
    C  = γˢ * S
    Δ   = sqrt(max(B^2 - 4A * C, zero(B)))
    S_b = clamp((-B - Δ) / (2A), zero(S), S)
    T_b = λ₁ * S_b + c₀
    ṁ   = ρʷ * cᵖʷ * γᵀ * (T - T_b) / (ρⁱ * Lᶠ)             # sgn from (T-T_b): melt>0, refreeze<0
    return ṁ, T_b, S_b, γᵀ, γˢ
end
#---

#+++ Tracer flux boundary conditions at the ice underside (2-D: x,z ; 3-D: x,y,z)
# `p` carries (Cd, z1, sfac, dx, Nx): sfac is the per-x basal-slope factor array (built by the
# driver), looked up by the cell's x-index. Signs: contact with ice cools the near-ice ocean
# toward freezing and freshens it. If a short test warms/salinifies, flip the sign of the returns.
@inline _slopefac(x, p) = @inbounds p.sfac[clamp(unsafe_trunc(Int, x / p.dx) + 1, 1, p.Nx)]

@inline function melt_heat_flux(x, z, t, u, v, w, T, S, p)
    _, T_b, _, γᵀ, _ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cd, p.z1, _slopefac(x, p))
    return γᵀ * (T_b - T)        # < 0 when T > T_b  ⇒ ocean loses heat to melting
end

@inline function melt_salt_flux(x, z, t, u, v, w, T, S, p)
    _, _, S_b, _, γˢ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cd, p.z1, _slopefac(x, p))
    return γˢ * (S_b - S)        # < 0 when S > S_b  ⇒ freshening near the ice
end

# 3-D variants (x, y, z, t, …) for the y-extruded periodic run (iceshelfcavity3d.jl). The slope
# factor depends on x only (geometry is extruded in y).
@inline function melt_heat_flux_3d(x, y, z, t, u, v, w, T, S, p)
    _, T_b, _, γᵀ, _ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cd, p.z1, _slopefac(x, p))
    return γᵀ * (T_b - T)
end

@inline function melt_salt_flux_3d(x, y, z, t, u, v, w, T, S, p)
    _, _, S_b, _, γˢ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cd, p.z1, _slopefac(x, p))
    return γˢ * (S_b - S)
end
#---
