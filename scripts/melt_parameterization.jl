# =====================================================================================
#  Three-equation ice–ocean melt parameterization  (ice-SHELF-cavity version)
#  (Holland & Jenkins 1999; Jenkins 2011; Asay-Davis et al. 2016 / ISOMIP+ coefficients.)
#
#  Applied as field-dependent FLUX boundary conditions on T and S at the underside of the
#  ice shelf (the `top` faces of immersed ice cells). Turbulent exchange velocities are
#  tied to the local friction velocity u★ = √(Cᴰ)·|U|, i.e. they use the SAME drag
#  coefficient as the momentum boundary condition, so momentum, heat and salt exchange are
#  mutually consistent.
#
#  This is the shear-dependent closure discussed in the talk. The depth term of the
#  liquidus (λ₃·z) is what makes it appropriate for a DEEP cavity: near the Pine Island
#  grounding line (z ≈ −1000 m) the in-situ freezing point is ≈ −2.6 °C, so warm modified
#  CDW at ~+1 °C gives a large thermal driving and the high grounding-line melt rates the
#  talk highlights.
#
#  NOTE ON SCOPE: melt is applied on downward-facing ice (`top=` immersed faces: the ice
#  base and any keel tips). Melt on steeply sloping ice faces (east/west immersed faces)
#  is a deliberate v1 simplification — distinguishing ice flanks from sea-floor faces
#  needs face-type tagging and is exactly the boundary-parameterization refinement the
#  cavity-LES program earmarks for development.
# =====================================================================================

using Oceananigans.Units

#+++ Physical constants
const cᵖʷ = 3974.0      # seawater specific heat        [J kg⁻¹ K⁻¹]
const Lᶠ  = 3.34e5      # latent heat of fusion         [J kg⁻¹]
const ρʷ  = 1027.0      # reference seawater density     [kg m⁻³]
const ρⁱ  = 920.0       # ice density                    [kg m⁻³]

# Linear liquidus  T_f = λ₁·S + λ₂ + λ₃·z   (z negative downward)
const λ₁  = -0.0573     # salinity coefficient           [°C psu⁻¹]
const λ₂  =  0.0832     # offset                         [°C]
const λ₃  =  7.53e-4    # depth coefficient              [°C m⁻¹]  (≈ −0.75 °C at z=−1000 m)

# Turbulent transfer coefficients (Stanton numbers), γ = Γ·u★
const Γᵀ  =  0.0110     # thermal
const Γˢ  =  0.00031    # haline

const u★min = 1.0e-3    # floor on u★ [m s⁻¹] so exchange does not vanish at zero shear
                        # (crudely stands in for residual convective exchange; the
                        #  buoyancy-vs-shear transition itself is a science question —
                        #  the "shear–convection tug of war" of the talk)
#---

#+++ Core solver
"""
    solve_melt(speed, T, S, z, Cᴰ)

Solve the three-equation system at one point on the ice base for the interface
salinity/temperature and the melt rate. `speed` is the FULL near-ice flow speed
|U| = √(u²+v²+w²). Returns `(ṁ, T_b, S_b, γᵀ, γˢ)`:

  • `ṁ`      melt rate [m s⁻¹], positive ⇒ melting
  • `T_b`    interface (freezing) temperature [°C]
  • `S_b`    interface salinity [psu]
  • `γᵀ,γˢ`  turbulent exchange velocities [m s⁻¹]

Eliminating T_b (liquidus) and ṁ (heat balance) from the salt balance gives a
quadratic A·S_b² + B·S_b + C = 0; the physical root is `(-B - √Δ)/(2A)`.
"""
@inline function solve_melt(speed, T, S, z, Cᴰ)
    u★ = max(sqrt(Cᴰ) * speed, u★min)
    γᵀ = Γᵀ * u★
    γˢ = Γˢ * u★

    c₀ = λ₂ + λ₃ * z                      # pressure/depth term of the liquidus
    r  = cᵖʷ * γᵀ / Lᶠ

    A  = r * λ₁                           # < 0  (λ₁ < 0)
    B  = -(γˢ + r * (T - c₀))
    C  = γˢ * S

    Δ   = sqrt(max(B^2 - 4A * C, zero(B)))
    S_b = clamp((-B - Δ) / (2A), zero(S), S)

    T_b = λ₁ * S_b + c₀
    ṁ   = ρʷ * cᵖʷ * γᵀ * (T - T_b) / (ρⁱ * Lᶠ)
    return ṁ, T_b, S_b, γᵀ, γˢ
end
#---

#+++ Tracer flux boundary conditions at the ice underside
# Kinematic fluxes (tracer units × m s⁻¹). Signs are set so that contact with the ice
# drives the near-ice ocean toward the freezing point (cooling) and freshens it
# (melt-water input). If a test run shows the near-ice layer warming/salinifying,
# flip the sign of the returns — the immersed `top`-face flux convention is the one
# thing worth verifying on the first short run.
@inline function melt_heat_flux(x, z, t, u, v, w, T, S, p)
    _, T_b, _, γᵀ, _ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cᴰ)
    return γᵀ * (T_b - T)        # < 0 when T > T_b  ⇒ ocean loses heat to melting
end

@inline function melt_salt_flux(x, z, t, u, v, w, T, S, p)
    _, _, S_b, _, γˢ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cᴰ)
    return γˢ * (S_b - S)        # < 0 when S > S_b  ⇒ freshening near the ice
end

# 3-D variants (x, y, z, t, …) for the y-extruded periodic run (iceshelfcavity3d.jl).
@inline function melt_heat_flux_3d(x, y, z, t, u, v, w, T, S, p)
    _, T_b, _, γᵀ, _ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cᴰ)
    return γᵀ * (T_b - T)
end

@inline function melt_salt_flux_3d(x, y, z, t, u, v, w, T, S, p)
    _, _, S_b, _, γˢ = solve_melt(sqrt(u^2 + v^2 + w^2), T, S, z, p.Cᴰ)
    return γˢ * (S_b - S)
end
#---
