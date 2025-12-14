/**
 * @file main_sim.c
 * @brief Host-side simulation entry point.
 */

#include <stdio.h>
#include "physics_constants.h"
#include "rotor_control.h"
#include "threshold_logic.h"
#include "pulse_control.h"

static void on_event(ThresholdEvent_t event, float threshold, int direction) {
    (void)threshold;
    const char *dir = (direction > 0) ? "+" : "-";
    printf("EVENT %d dir=%s z=%.4f tier=%s phase=%s\n",
           (int)event, dir,
           ThresholdLogic_GetZ(),
           ThresholdLogic_GetTierName(ThresholdLogic_GetTier()),
           ThresholdLogic_GetPhaseName(ThresholdLogic_GetPhase()));
}

int main(void) {
    printf("Nuclear Spinner SIM | φ=%.6f z_c=%.6f σ=%.0f\n", PHI, Z_CRITICAL, SIGMA);

    PulseControl_Init();
    RotorControl_Init();
    ThresholdLogic_Init();
    ThresholdLogic_SetEventCallback(on_event);

    // Sweep z from 0.30 -> 0.95 in steps
    for (float z = 0.30f; z <= 0.95f; z += 0.01f) {
        RotorControl_SetZ(z);
        // simulate a control update
        RotorControl_Update();
        ThresholdLogic_Update(RotorControl_GetZ(), 0.95f, 0.70f, (int)(10*compute_complexity(z)));
        if (ThresholdLogic_IsAtLens()) {
            // run a quick echo at the lens
            PulseControl_SpinEcho(0.8f, 1000);
        }
    }

    printf("SIM done.\n");
    return 0;
}
