#ifndef HEATMAP_OVERLAY_H
#define HEATMAP_OVERLAY_H

#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Attach a buffer probe to `pad` that splats a detection-density heatmap
 * directly onto the RGBA pixels flowing through. Reads NvDsBatchMeta for
 * tracked-object centers; runs two CUDA kernels per active source; writes in place.
 *
 * Config priority (highest wins):
 *   1. INI file at $HEATMAP_OVERLAY_CONFIG, else ./heatmap_overlay.ini
 *   2. env vars HEATMAP_OVERLAY_* (legacy)
 *   3. defaults
 *
 * INI keys (under [heatmap_overlay]):
 *   enable=1
 *   source_id=0       (-1 overlays all sources 0-6)
 *   decay=0.985
 *   sigma_px=28
 *   alpha_max=0.65
 *   splat=0.55
 *
 * Returns 0 if probe attached, 1 if disabled by config, <0 on error.
 */
int heatmap_overlay_attach(GstPad *pad);

#ifdef __cplusplus
}
#endif

#endif
