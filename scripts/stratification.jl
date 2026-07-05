# =====================================================================================
#  Idealized two-layer Antarctic ambient stratification for the ice-shelf cavity.
#
#  Cold, fresh near-surface water (Winter Water) over warm, salty modified Circumpolar
#  Deep Water (mCDW) at depth — the classic warm-cavity structure (e.g. Pine Island).
#  Same GPU-safe analytic form as the landfast CTD fit,
#        f(z) = a + c·tanh((z − z0)/h)          (z negative downward),
#  so it can be evaluated inside Oceananigans' GPU kernels (no interpolation object).
#
#  These are REPRESENTATIVE, TUNABLE values — not a specific cast. Adjust:
#    • warm/cold end members  → aT, cT  (and aS, cS)
#    • thermocline/pycnocline depth → z0T, z0S   (here ≈ −600 m)
#    • sharpness of the interface   → hT, hS      (smaller ⇒ sharper two-layer)
#
#  End members implied by the constants below:
#    deep  (z ≈ −1000 m):  T ≈ +1.0 °C,  S ≈ 34.70 psu   (warm mCDW)
#    shelf (z ≈ −350 m):   T ≈ −1.5 °C,  S ≈ 34.02 psu   (cold Winter Water)
# =====================================================================================

# Temperature fit  T(z) = aT + cT·tanh((z − z0T)/hT)
const aT, cT, z0T, hT = -0.30, -1.30, -600.0, 150.0
# Salinity fit     S(z) = aS + cS·tanh((z − z0S)/hS)
const aS, cS, z0S, hS = 34.35, -0.35, -600.0, 150.0

@inline T_profile(z) = aT + cT * tanh((z - z0T) / hT)
@inline S_profile(z) = aS + cS * tanh((z - z0S) / hS)
