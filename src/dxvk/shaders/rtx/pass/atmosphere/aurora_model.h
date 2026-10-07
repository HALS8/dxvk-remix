#pragma once

// Constants of the aurora model shared by the shader that draws it
// (atmosphere_aurora.slangh) and the CPU estimate of the light it casts
// (RtxAtmosphere::syncDistantLights). The two describe one display, so what
// sets its brightness and where its light sits lives here once.

#define AURORA_MAX_ARCS 4u

// Linear Rec.709 luminance (cd/m^2) per kilorayleigh of each line: the photon
// radiance of 1 kR (1e9 / 4pi photons cm^-2 s^-1 sr^-1) times the photon
// energy, times 683 lm/W and the CIE 1931 colour matching functions, through
// the XYZ to Rec.709 matrix with out-of-gamut components clipped as Baranoski
// et al. do. Spectral lines sit outside any display gamut, so the clip is the
// whole colour story.
#define AURORA_GREEN_PER_KR vec3(5.19e-5f, 2.577e-4f, 0.0f)    // O   557.7 nm
#define AURORA_RED_PER_KR   vec3(2.869e-4f, 0.0f, 0.0f)        // O   630.0 nm
#define AURORA_BLUE_PER_KR  vec3(4.84e-5f, 0.0f, 3.320e-4f)    // N2+ 427.8 nm

// Peak altitude of each line above the lower border, and the scale height
// below the peak, in km. Above the peak the green and blue lines fall off over
// the display's vertical extent; red is quenched below ~200 km and has its own.
#define AURORA_GREEN_PEAK_KM   10.0f
#define AURORA_GREEN_BELOW_KM  3.5f
#define AURORA_BLUE_PEAK_KM    4.0f
#define AURORA_BLUE_BELOW_KM   2.5f
#define AURORA_BLUE_EXTENT     0.5f    // of the green line's extent
#define AURORA_RED_PEAK_KM     130.0f
#define AURORA_RED_BELOW_KM    40.0f
#define AURORA_RED_ABOVE_KM    70.0f

// How much further a ray bundle stands than the curtain between bundles.
#define AURORA_RAY_EXTENT_GAIN 3.0f

// Arcs dim poleward of the nearest one.
#define AURORA_ARC_BRIGHTNESS(k) (1.0f - 0.18f * float(k))
