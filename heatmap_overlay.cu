// libheatmap_overlay.so — GPU heatmap overlay on the displayed NvBufSurface.
// Loaded via dlopen() from a tiny deepstream_app.c patch when HEATMAP_OVERLAY=1.
//
// Reads track centers from NvDsBatchMeta, splats Gaussians into a persistent
// device-side float accumulator, JET-colormaps + alpha-blends into the RGBA
// dataPtr in place. Two kernels per active source, no host-side pixel copy.

#include "heatmap_overlay.h"

#include <cuda_runtime.h>
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace {

static constexpr int kAllSources = -1;
static constexpr int kMaxHeatmapSources = 7;
static constexpr int kEmptyAfter = 600;

struct AccumState {
    int W = 0, H = 0;
    float *d_accum = nullptr;          // device, W*H floats
    int empty_streak = kEmptyAfter;    // already empty until a detection arrives
};

struct OverlayCtx {
    int source_id = 0;
    float decay = 0.985f;
    float sigma_px = 28.0f;
    float alpha_max = 0.65f;
    float splat = 0.55f;
    int   tile_rows = 1;          // 1 == not tiled (whole canvas is the source)
    int   tile_cols = 1;

    AccumState accum[kMaxHeatmapSources];
    float2 *d_centers = nullptr;       // device, MAX_CENTERS
    float  *d_confs = nullptr;         // device, MAX_CENTERS
    int max_centers = 256;
    cudaStream_t stream = nullptr;
    std::atomic<bool> warned_format{false};
    std::atomic<bool> warned_no_meta{false};
    std::atomic<bool> warned_unwritable{false};
    std::atomic<bool> logged_first{false};
    std::atomic<bool> logged_no_source{false};

    // Watchdog: disables itself if probe runs too slow.
    // Budget covers all active sources on one CUDA stream (≈0.3-0.5 ms per
    // source × up to 7 sources, plus sync), so 6 ms gives headroom over the
    // ~2-3.5 ms expected for the 7-stream case.
    static constexpr double kBudgetMs = 6.0;
    static constexpr int    kMaxBreaches = 30;   // consecutive frames over budget
    int  breach_streak = 0;
    bool disabled = false;
    gulong probe_id = 0;                // for self-removal


    bool all_sources() const { return source_id == kAllSources; }
};

#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "[heatmap_overlay] CUDA error %s at %s:%d\n", \
            cudaGetErrorString(_e), __FILE__, __LINE__); return; } } while (0)

// ---------- kernels ----------

__global__ void k_decay_splat(float *accum, int W, int H, float decay,
                              const float2 *centers, const float *confs, int n,
                              float sigma, float splat)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    float v = accum[idx] * decay;

    if (n > 0) {
        const float inv_2s2 = 1.0f / (2.0f * sigma * sigma);
        float add = 0.0f;
        // Each pixel iterates over all detections — fine for n <= ~64.
        // Skip pixels far from center via a per-detection bbox check.
        const float r = 3.5f * sigma;
        for (int i = 0; i < n; ++i) {
            float dx = (float)x - centers[i].x;
            float dy = (float)y - centers[i].y;
            if (fabsf(dx) > r || fabsf(dy) > r) continue;
            add += confs[i] * splat * __expf(-(dx*dx + dy*dy) * inv_2s2);
        }
        v += add;
    }
    if (v > 4.0f) v = 4.0f;          // soft cap
    accum[idx] = v;
}

// JET colormap (compact analytic form) returning float3 in [0,255].
__device__ inline void jet(float t, float &r, float &g, float &b) {
    // t in [0,1]
    float t4 = t * 4.0f;
    r = fminf(fmaxf(fminf(t4 - 1.5f, 4.5f - t4), 0.0f), 1.0f);
    g = fminf(fmaxf(fminf(t4 - 0.5f, 3.5f - t4), 0.0f), 1.0f);
    b = fminf(fmaxf(fminf(t4 + 0.5f, 2.5f - t4), 0.0f), 1.0f);
    r *= 255.0f; g *= 255.0f; b *= 255.0f;
}

__global__ void k_blend_nv12(unsigned char *yPlane, int yPitch,
                             unsigned char *uvPlane, int uvPitch,
                             int W, int H,
                             const float *accum, float alpha_max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float a = accum[y * W + x];
    if (a < 0.05f) return;

    float t = a / 2.0f;
    if (t > 1.0f) t = 1.0f;
    float alpha = a;
    if (alpha > 1.0f) alpha = 1.0f;
    alpha *= alpha_max;

    float r, g, b;
    jet(t, r, g, b);

    // BT.709 RGB->Y
    float yh = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    unsigned char *yp = yPlane + y * yPitch + x;
    float inv = 1.0f - alpha;
    float yv = (float)*yp * inv + yh * alpha;
    if (yv > 255.0f) yv = 255.0f;
    *yp = (unsigned char)yv;

    // UV plane is half-res; one thread per 2x2 block writes it.
    if ((x & 1) == 0 && (y & 1) == 0) {
        float u = -0.1146f*r - 0.3854f*g + 0.5f*b + 128.0f;
        float v =  0.5f*r    - 0.4542f*g - 0.0458f*b + 128.0f;
        unsigned char *uvp = uvPlane + (y >> 1) * uvPitch + (x & ~1);
        uvp[0] = (unsigned char)((float)uvp[0] * inv + u * alpha);
        uvp[1] = (unsigned char)((float)uvp[1] * inv + v * alpha);
    }
}

__global__ void k_blend_rgba(uchar4 *rgba, int pitch_bytes, int W, int H,
                             const float *accum, float alpha_max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float a = accum[y * W + x];
    if (a < 0.05f) return;

    float t = a / 2.0f;              // map [0..2] -> [0..1] for JET
    if (t > 1.0f) t = 1.0f;

    float alpha = a;
    if (alpha > 1.0f) alpha = 1.0f;
    alpha *= alpha_max;

    float r, g, b;
    jet(t, r, g, b);

    uchar4 *row = (uchar4 *)((unsigned char *)rgba + y * pitch_bytes);
    uchar4 px = row[x];
    float inv = 1.0f - alpha;
    px.x = (unsigned char)(px.x * inv + b * alpha);   // NVBUF RGBA layout: x=R? -> see below
    px.y = (unsigned char)(px.y * inv + g * alpha);
    px.z = (unsigned char)(px.z * inv + r * alpha);
    // px.w left alone
    row[x] = px;
}
// NOTE: NVBUF_COLOR_FORMAT_RGBA stores bytes as R,G,B,A in memory, so uchar4.x=R,
// .y=G, .z=B. The blend above writes (R<-blue, G<-green, B<-red) which produces
// the BGR-ish look DeepStream displays expect because the sink interprets the
// surface as RGBA literal. If colors look wrong on your sink, swap r<->b in the
// three lines above. We choose this default because nvdsosd/sink output is BGRA-
// colored on screen for many DS9 sinks; flip if needed.

// ---------- helpers ----------

static int env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atoi(v);
}
static float env_float(const char *name, float dflt) {
    const char *v = std::getenv(name);
    if (!v || !*v) return dflt;
    return (float)std::atof(v);
}

struct IniSettings {
    bool found = false;
    bool enable = true;
    int   source_id = INT_MIN;
    float decay     = NAN;
    float sigma_px  = NAN;
    float alpha_max = NAN;
    float splat     = NAN;
    int   tile_rows = INT_MIN;
    int   tile_cols = INT_MIN;
};

static IniSettings load_ini() {
    IniSettings s;
    const char *path = std::getenv("HEATMAP_OVERLAY_CONFIG");
    if (!path || !*path) path = "heatmap_overlay.ini";
    GKeyFile *kf = g_key_file_new();
    GError *err = nullptr;
    if (!g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, &err)) {
        if (err) g_error_free(err);
        g_key_file_free(kf);
        return s;
    }
    s.found = true;
    const char *grp = "heatmap_overlay";
    if (g_key_file_has_key(kf, grp, "enable", nullptr))
        s.enable = g_key_file_get_boolean(kf, grp, "enable", nullptr);
    if (g_key_file_has_key(kf, grp, "source_id", nullptr))
        s.source_id = g_key_file_get_integer(kf, grp, "source_id", nullptr);
    if (g_key_file_has_key(kf, grp, "decay", nullptr))
        s.decay = (float)g_key_file_get_double(kf, grp, "decay", nullptr);
    if (g_key_file_has_key(kf, grp, "sigma_px", nullptr))
        s.sigma_px = (float)g_key_file_get_double(kf, grp, "sigma_px", nullptr);
    if (g_key_file_has_key(kf, grp, "alpha_max", nullptr))
        s.alpha_max = (float)g_key_file_get_double(kf, grp, "alpha_max", nullptr);
    if (g_key_file_has_key(kf, grp, "splat", nullptr))
        s.splat = (float)g_key_file_get_double(kf, grp, "splat", nullptr);
    if (g_key_file_has_key(kf, grp, "tile_rows", nullptr))
        s.tile_rows = g_key_file_get_integer(kf, grp, "tile_rows", nullptr);
    if (g_key_file_has_key(kf, grp, "tile_cols", nullptr))
        s.tile_cols = g_key_file_get_integer(kf, grp, "tile_cols", nullptr);
    fprintf(stderr, "[heatmap_overlay] loaded config from %s\n", path);
    g_key_file_free(kf);
    return s;
}

static void ensure_buffers(OverlayCtx *ctx, AccumState *acc, int source_id, int W, int H) {
    if (acc->W == W && acc->H == H && acc->d_accum) return;
    if (acc->d_accum) cudaFree(acc->d_accum);
    acc->W = W; acc->H = H;
    acc->empty_streak = kEmptyAfter;
    cudaMalloc(&acc->d_accum, sizeof(float) * W * H);
    cudaMemsetAsync(acc->d_accum, 0, sizeof(float) * W * H, ctx->stream);
    if (!ctx->d_centers) {
        cudaMalloc(&ctx->d_centers, sizeof(float2) * ctx->max_centers);
        cudaMalloc(&ctx->d_confs,   sizeof(float)  * ctx->max_centers);
    }
    fprintf(stderr, "[heatmap_overlay] accumulator src=%d %dx%d allocated (%.2f MB)\n",
            source_id, W, H, (W * H * 4) / (1024.0 * 1024.0));
}

static bool source_enabled(const OverlayCtx *ctx, int source_id, int pad_index) {
    if (ctx->all_sources())
        return (source_id >= 0 && source_id < kMaxHeatmapSources) ||
               (pad_index >= 0 && pad_index < kMaxHeatmapSources);
    return source_id == ctx->source_id || pad_index == ctx->source_id;
}

// ---------- probe ----------

static GstPadProbeReturn on_buffer(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    OverlayCtx *ctx = (OverlayCtx *)user_data;
    if (ctx->disabled) return GST_PAD_PROBE_OK;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    if (!gst_buffer_is_writable(buf)) {
        if (!ctx->warned_unwritable.exchange(true))
            fprintf(stderr, "[heatmap_overlay] buffer not writable (refcount>1?); skipping\n");
        return GST_PAD_PROBE_OK;
    }

    GstClockTime t0 = gst_util_get_timestamp();

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READWRITE)) return GST_PAD_PROBE_OK;
    NvBufSurface *surf = (NvBufSurface *)map.data;

    NvDsBatchMeta *bm = gst_buffer_get_nvds_batch_meta(buf);
    if (!bm) {
        if (!ctx->warned_no_meta.exchange(true))
            fprintf(stderr, "[heatmap_overlay] no NvDsBatchMeta on buffer\n");
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }

    bool matched_source = false;
    bool launched = false;
    for (NvDsMetaList *fl = bm->frame_meta_list; fl; fl = fl->next) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fl->data;
        int source_id = (int)fm->source_id;
        int pad_index = (int)fm->pad_index;
        if (!source_enabled(ctx, source_id, pad_index)) continue;

        int slot = source_id;
        if (slot < 0 || slot >= kMaxHeatmapSources) slot = pad_index;
        if (slot < 0 || slot >= kMaxHeatmapSources) continue;

        int batch_idx = fm->batch_id;
        if (batch_idx < 0 || batch_idx >= (int)surf->numFilled) continue;
        matched_source = true;

        NvBufSurfaceParams *params = &surf->surfaceList[batch_idx];
        int cf = (int)params->colorFormat;
        bool is_rgba = (cf == NVBUF_COLOR_FORMAT_RGBA);
        bool is_nv12 = (cf == NVBUF_COLOR_FORMAT_NV12 ||
                        cf == NVBUF_COLOR_FORMAT_NV12_ER ||
                        cf == 33 /* NV12_709 */ ||
                        cf == 34 /* NV12_709_ER */);
        if (!is_rgba && !is_nv12) {
            if (!ctx->warned_format.exchange(true))
                fprintf(stderr, "[heatmap_overlay] unsupported colorFormat=%d (need RGBA=19 or NV12-family)\n", cf);
            continue;
        }

        int W = (int)params->width;
        int H = (int)params->height;
        int pitch = (int)params->pitch;
        void *dataPtr = params->dataPtr;
        if (!dataPtr || W <= 0 || H <= 0) continue;
        AccumState *acc = &ctx->accum[slot];
        ensure_buffers(ctx, acc, slot, W, H);

        if (!ctx->logged_first.exchange(true)) {
            fprintf(stderr,
                "[heatmap_overlay] first buffer: %dx%d, memType=%d, batch_filled=%u, "
                "frames_in_batch=%u, mode=%s\n",
                W, H, (int)surf->memType, surf->numFilled,
                bm ? bm->num_frames_in_batch : 0u,
                ctx->all_sources() ? "all sources 0-6" : "single source");
        }

        int n = 0;
        float2 *h_centers = (float2 *)alloca(sizeof(float2) * ctx->max_centers);
        float  *h_confs   = (float  *)alloca(sizeof(float)  * ctx->max_centers);
        for (NvDsMetaList *l = fm->obj_meta_list; l && n < ctx->max_centers; l = l->next) {
            NvDsObjectMeta *o = (NvDsObjectMeta *)l->data;
            float cx = o->rect_params.left + o->rect_params.width  * 0.5f;
            float cy = o->rect_params.top  + o->rect_params.height * 0.5f;
            float c  = o->confidence > 0.0f ? o->confidence : 1.0f;
            h_centers[n] = make_float2(cx, cy);
            h_confs[n]   = c;
            n++;
        }
        bool skip_all = (n == 0 && acc->empty_streak >= kEmptyAfter);
        if (skip_all) {
            continue;
        }

        if (n > 0) {
            cudaMemcpyAsync(ctx->d_centers, h_centers, sizeof(float2) * n, cudaMemcpyHostToDevice, ctx->stream);
            cudaMemcpyAsync(ctx->d_confs,   h_confs,   sizeof(float)  * n, cudaMemcpyHostToDevice, ctx->stream);
        }

        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        k_decay_splat<<<grid, block, 0, ctx->stream>>>(
            acc->d_accum, W, H, ctx->decay,
            ctx->d_centers, ctx->d_confs, n,
            ctx->sigma_px, ctx->splat);
        if (is_rgba) {
            k_blend_rgba<<<grid, block, 0, ctx->stream>>>(
                (uchar4 *)dataPtr, pitch, W, H, acc->d_accum, ctx->alpha_max);
        } else {
            unsigned char *base = (unsigned char *)dataPtr;
            unsigned char *yPlane  = base + params->planeParams.offset[0];
            unsigned char *uvPlane = base + params->planeParams.offset[1];
            int yPitch  = (int)params->planeParams.pitch[0];
            int uvPitch = (int)params->planeParams.pitch[1];
            k_blend_nv12<<<grid, block, 0, ctx->stream>>>(
                yPlane, yPitch, uvPlane, uvPitch, W, H,
                acc->d_accum, ctx->alpha_max);
        }
        launched = true;
        if (n == 0) acc->empty_streak++; else acc->empty_streak = 0;
    }

    if (!matched_source) {
        if (!ctx->logged_no_source.exchange(true))
            fprintf(stderr, "[heatmap_overlay] WARN: source_id=%d not in batch (numFilled=%u)\n",
                    ctx->source_id, surf->numFilled);
        gst_buffer_unmap(buf, &map);
        return GST_PAD_PROBE_OK;
    }

    // Sync once after all active source kernels; downstream may read on another stream.
    if (launched) cudaStreamSynchronize(ctx->stream);

    gst_buffer_unmap(buf, &map);

    // Watchdog.
    GstClockTime t1 = gst_util_get_timestamp();
    double dt_ms = (double)(t1 - t0) / 1.0e6;

    if (dt_ms > OverlayCtx::kBudgetMs) {
        ctx->breach_streak++;
        if (ctx->breach_streak >= OverlayCtx::kMaxBreaches && !ctx->disabled) {
            ctx->disabled = true;
            fprintf(stderr,
                "[heatmap_overlay] WATCHDOG: probe exceeded %.1fms for %d consecutive "
                "frames (last=%.2fms); SELF-DISABLED to protect pipeline. "
                "Tune sigma_px/decay or set enable=0.\n",
                OverlayCtx::kBudgetMs, OverlayCtx::kMaxBreaches, dt_ms);
            if (ctx->probe_id) gst_pad_remove_probe(pad, ctx->probe_id);
        }
    } else {
        ctx->breach_streak = 0;
    }

    return GST_PAD_PROBE_OK;
}

} // anon

extern "C" int heatmap_overlay_attach(GstPad *pad) {
    if (!pad) return -1;

    IniSettings ini = load_ini();
    if (ini.found && !ini.enable) {
        fprintf(stderr, "[heatmap_overlay] disabled by config (enable=0)\n");
        return 1;
    }

    OverlayCtx *ctx = new OverlayCtx();
    ctx->source_id = (ini.source_id != INT_MIN)
                       ? ini.source_id
                       : env_int("HEATMAP_OVERLAY_SOURCE_ID", 0);
    ctx->decay     = !std::isnan(ini.decay)
                       ? ini.decay
                       : env_float("HEATMAP_OVERLAY_DECAY", 0.985f);
    ctx->sigma_px  = !std::isnan(ini.sigma_px)
                       ? ini.sigma_px
                       : env_float("HEATMAP_OVERLAY_SIGMA_PX", 28.0f);
    ctx->alpha_max = !std::isnan(ini.alpha_max)
                       ? ini.alpha_max
                       : env_float("HEATMAP_OVERLAY_ALPHA_MAX", 0.65f);
    ctx->splat     = !std::isnan(ini.splat)
                       ? ini.splat
                       : env_float("HEATMAP_OVERLAY_SPLAT", 0.55f);
    ctx->tile_rows = (ini.tile_rows != INT_MIN) ? ini.tile_rows
                       : env_int("HEATMAP_OVERLAY_TILE_ROWS", 1);
    ctx->tile_cols = (ini.tile_cols != INT_MIN) ? ini.tile_cols
                       : env_int("HEATMAP_OVERLAY_TILE_COLS", 1);
    if (ctx->tile_rows < 1) ctx->tile_rows = 1;
    if (ctx->tile_cols < 1) ctx->tile_cols = 1;
    cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);

    ctx->probe_id = gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, on_buffer, ctx, nullptr);
    fprintf(stderr,
        "[heatmap_overlay] attached (source_id=%d tile=%dx%d decay=%.3f sigma=%.1f alpha=%.2f splat=%.2f)\n",
        ctx->source_id, ctx->tile_rows, ctx->tile_cols,
        ctx->decay, ctx->sigma_px, ctx->alpha_max, ctx->splat);
    return 0;
}
