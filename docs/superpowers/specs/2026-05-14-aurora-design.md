# Aurora design — Phase 3.5

Status: approved-for-implementation
Target: `feinorgh/the-dark-candle` master, Bevy 0.18, Rust edition 2024
Related roadmap entry: "Phase 3 — Visual effects, 3.5 Aurora"

## Goal

Add a *visible-from-orbit-and-from-the-ground* aurora effect over the
polar regions of the planet. The aurora must:

1. Be a real piece of geometry in render-space (not a sky-dome
   overlay), so it can be observed both from a surface camera looking
   up *and* from an orbital camera looking down.
2. Read as the iconic green-with-red-top-fringe **curtain** silhouette,
   not as a uniform glowing band.
3. Only appear under physically plausible conditions (high latitude,
   night side / deep twilight).
4. Be a single integration in the lighting plugin — no new top-level
   plugin, no new asset RON schema.

## Non-goals

- Coupling with solar-wind / coronal-mass-ejection events. The aurora
  is always-on inside the gating envelope; a future
  geomagnetic-storm system can modulate a `strength` field.
- A real magnetic dipole simulation. We expose a knob for it (see
  §Configuration) but the default behaviour is "magnetic pole =
  geographic pole".
- Aurora response to the player flying *through* the curtain. The
  shell is rendered as additive volume; collision and interactive
  attenuation are out of scope.
- Day-side aurora (it exists physically but is invisible against
  scattered sunlight).

## Architecture

### Render approach: dedicated annular-shell mesh + custom material

Approach B from the brainstorming session (volumetric shell), rejected
A (sky-dome additive) because A would not be visible from orbit.

```
                    ┌──────────────────────────────────────┐
                    │       Aurora (this design)           │
                    │                                      │
                    │   AuroraShellMesh ─┐                 │
                    │                    │                 │
                    │   AuroraMaterial   │  forward, add   │
                    │   (custom Mat'l) ──┘  blend, no Z    │
                    │                                      │
                    └──────────┬───────────────────────────┘
                               │ rendered after opaque + before
                               │ sky dome's Transparent3d sun/stars
                               ▼
                        (existing pipeline)
```

#### AuroraShellMesh

A spherical annular shell (one mesh) of inner radius
`sea_level_radius + 95_000 m` and outer radius
`sea_level_radius + 300_000 m`. Generated once at planet spawn time as
two concentric icosphere-derived hulls joined by side strips, OR more
simply as a single outer sphere with the fragment shader doing the
inner-radius ray-march (we'll take this simpler form — see below).

In practice we will use **a single outer-shell sphere mesh** with
`Visibility::Hidden` until the planet is loaded. The fragment shader
performs an analytic ray-sphere intersection against the inner radius
to find the entry and exit points of the view ray through the shell,
and integrates emission along that segment. This avoids generating a
genuinely toroidal mesh while still giving correct volumetric
silhouettes.

Mesh radius: `sea_level_radius + 300_000 m` (outer atmosphere top).
Frustum culling: `NoFrustumCulling` (same convention as sky dome) so
the aurora draws even when the camera is inside the shell.

#### AuroraMaterial

A `bevy::pbr::Material` with `AlphaMode::Add` so it bypasses the
depth-write pass and additively blends over whatever has already been
rendered (terrain, sky dome, stars). Non-opaque materials in Bevy
0.18 are forward-rendered by default, so no `OpaqueRendererMethod`
override is needed.

Uniforms (uploaded each frame by `update_aurora_material`):

| Binding | Field | Type | Meaning |
|---|---|---|---|
| 0 | `planet_center_render` | `Vec4` | Planet center in render-space (= `-RenderOrigin.0` cast to f32). w unused. |
| 1 | `magnetic_north_axis` | `Vec4` | Unit vector pointing to the magnetic north pole, in **render-space**, derived from `planet_north_axis` plus `magnetic_pole_offset_deg`. xyz used. |
| 2 | `sun_world_direction` | `Vec4` | Unit vector from surface toward sun. Same source as the caustic / sky dome uniforms. |
| 3 | `aurora_params` | `Vec4` | x = inner radius (m), y = outer radius (m), z = base strength, w = elapsed time (s) for animation. |
| 4 | `aurora_band` | `Vec4` | x = oval-center latitude (rad, +ve = north, mirrored south), y = oval half-width (rad), z = curtain frequency, w = curtain animation speed (rad/s). |

No textures. All pattern data is procedural.

#### Fragment shader (`assets/shaders/aurora.wgsl`)

Per fragment:

1. Compute `world_view = normalize(world_pos - view.world_position)`.
2. Ray-intersect against the inner shell sphere (`planet_center`,
   `r_inner`) and outer shell sphere (`r_outer`). Two intersections
   form the integration interval `[t_enter, t_exit]`.
3. March the segment with N samples (target N = 12; cheap because the
   integrand is smooth).
4. For each sample point `p` along the ray:
   - `r       = length(p - planet_center)`
   - `up      = (p - planet_center) / r`
   - `cos_lat_to_mag_pole = dot(up, magnetic_north_axis)`
   - `lat_mag = asin(clamp(cos_lat_to_mag_pole, -1, 1))` — actually
     the "magnetic latitude" of this sample
   - **Band mask** = `smoothstep` over a ring centered on
     `|lat_mag| = band_center_lat` with half-width
     `band_half_width`. Two ovals (north + south) — use `abs(lat_mag)`.
   - **Vertical color ramp**: a normalized vertical coordinate
     `h_norm = (r - r_inner) / (r_outer - r_inner)`. Bottom → green
     (0.2, 1.0, 0.4), top → red/purple (0.6, 0.1, 0.7). Mix linearly.
   - **Curtain pattern**: 2-D noise in `(longitude_around_axis, h_norm)`
     with a slow time term: `noise(curtain_freq * lon + speed*t,
     h_norm * 4.0)`. Use the existing hash22 / Worley helpers from
     `terrain_caustic.wgsl` so we don't grow the WGSL helper library.
     Sharpen with `pow(noise, 2)` to get curtain silhouettes.
   - **Day-side gate**: factor
     `clamp(-dot(up, sun_world_direction) * 5.0 + 0.2, 0, 1)`
     — aurora only contributes where the sun is below the local
     horizon by more than a few degrees.
5. Sum the per-sample emission, scale by `(t_exit - t_enter) / N`, and
   multiply by `aurora_params.z` (overall strength).
6. Output `vec4(emission, 1.0)`. Additive blending writes pure
   emission onto the framebuffer.

Cost: 12 samples × (1 hash, 1 smoothstep, 1 mix, 1 dot) per fragment.
Mid-range GPU at 1080p, shell covering ~15 % of screen → well under
0.5 ms.

### Update system

`update_aurora_material()` in `src/lighting/aurora.rs`, scheduled in
`LightingPlugin`'s `PostUpdate` set alongside
`anchor_sky_dome_to_camera` and `update_terrain_caustic_uniform`
(*after* `TransformSystems::Propagate` so it sees the post-rebase
`RenderOrigin`):

```rust
fn update_aurora_material(
    render_origin: Res<RenderOrigin>,
    sun: Res<SunWorldDirection>,
    planet: Res<PlanetConfig>,
    time: Res<Time>,
    handle: Res<AuroraMaterialHandle>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) { ... }
```

The aurora shell anchors to the planet (NOT to the camera) — it lives
at the planet's render-space center (`-RenderOrigin.0`). A separate
small system (`anchor_aurora_shell_to_planet`) sets its `Transform`
each frame.

### Plugin wiring (`src/lighting/mod.rs`)

```rust
pub mod aurora;
// in build():
app.add_plugins(MaterialPlugin::<aurora::AuroraMaterial>::default())
   .add_systems(Startup, aurora::spawn_aurora_shell)
   .add_systems(Update, aurora::update_aurora_material)
   .add_systems(
       PostUpdate,
       aurora::anchor_aurora_shell_to_planet.after(TransformSystems::Propagate),
   );
```

The Startup spawn is deferred until `PlanetConfig` is available — same
"wait until planet loaded" pattern that `sky_dome::spawn_sky_dome`
uses.

## Configuration

New fields on `PlanetConfig` (loaded from `assets/data/*.planet.ron`):

| Field | Type | Default | Future-magnetic-field hook |
|---|---|---|---|
| `magnetic_pole_offset_deg` | `Vec2` | `(0.0, 0.0)` | `(lat, lon)` offset from geographic north. `(0,0)` ⇒ magnetic pole coincides with geographic pole — current Earth-like default. |
| `aurora_strength` | `f32` | `1.0` | Overall scalar multiplier. `0.0` = aurora disabled for this planet. |
| `aurora_band_center_deg` | `f32` | `67.0` | Magnetic latitude of the oval centroid. |
| `aurora_band_half_width_deg` | `f32` | `5.0` | Oval half-width. |

These four fields cover every iteration we want to do in the near
future (storm-time widening, strong-field planets pushing the oval
equatorward, etc.) without forcing us to invent a magnetic-field
solver now.

### Future magnetic-field implementations (out of scope; planning note)

The current model assumes a **static dipole** anchored to the
geographic axis (offset = 0). Real planetary magnetism varies wildly,
and the aurora is the *most visible* consequence of that variation, so
the design must not preclude these future moves:

1. **Stronger / weaker fields** — already covered by
   `aurora_strength`. A storm system can multiply this transiently.
   Stronger steady fields push the oval poleward (smaller
   `band_center_deg`), weaker fields push it equatorward (larger).
   We may want to add a `magnetic_field_strength_relative` field to
   `PlanetConfig` and derive the band center from it via Chapman-style
   scaling.
2. **Tilted dipole** — already accounted for by
   `magnetic_pole_offset_deg`. A tilt of e.g. (10°, −70°) makes the
   oval visibly off-axis (cf. Earth's actual magnetic pole near
   Greenland). Loading code reads the offset and rotates
   `planet_north_axis` accordingly into a `magnetic_north_axis`
   resource.
3. **Quadrupolar / non-dipolar fields** — Neptune-style geometry. Will
   require replacing the single `magnetic_north_axis` uniform with a
   spherical-harmonic coefficient buffer and rewriting the shader's
   "magnetic latitude" computation as a field-line trace. The
   shader-side change is local to `aurora.wgsl`; the Rust-side
   `AuroraMaterial` schema gains a storage buffer binding.
4. **Dynamic / wandering poles** — geomagnetic secular variation or
   pole flips. Just animate `magnetic_pole_offset_deg` over game time.
   No shader change needed.
5. **No magnetic field at all** — set `aurora_strength = 0.0`. The
   shell entity stays in the scene but contributes nothing; cost is a
   single ALU op in the fragment shader to early-out.
6. **Multiple bands** — Jupiter has *three* auroral ovals (main +
   satellite-induced). Would mean an array of `(band_center,
   half_width, strength)` triples uniform-uploaded. Trivial extension
   of `aurora_band` from one Vec4 to an array.

## Data flow

```
PlanetConfig (RON) ─────┐
                        │
SunWorldDirection ──────┼─► update_aurora_material
RenderOrigin ──────────┤    ─────┐
Time ──────────────────┘         │ writes uniforms
                                 ▼
                          AuroraMaterial ◄── owned by AuroraMaterialHandle
                                 │
                                 ▼
                          aurora.wgsl fragment
                                 │
                                 ▼
                          additive frame buffer
```

```
RenderOrigin ─► anchor_aurora_shell_to_planet ─► Transform on shell entity
```

## Testing

### Unit tests (`src/lighting/aurora.rs` `#[cfg(test)] mod tests`)

1. `aurora_band_mask_peaks_at_band_center_lat` — pure-Rust
   reimplementation of the band-mask `smoothstep`, asserts that the
   mask peaks at `band_center_lat` and ramps to ≈0 at
   `band_center_lat ± 2 × half_width`. Mirror of what the shader
   does — same role `caustic_tile.rs` plays for the caustic shader.
2. `magnetic_north_axis_default_matches_geographic` — given
   `magnetic_pole_offset_deg = (0, 0)`, the computed axis equals
   `planet_north_axis` to within 1e-6.
3. `magnetic_north_axis_with_tilt_offsets_correctly` — a (10°, 0°)
   offset rotates the axis by 10° about the +X-axis. Verifies the
   sign convention.
4. `aurora_day_side_factor_is_zero_at_zenith_sun` — the day-side gate
   returns 0 when `dot(up, sun) > 0.1` and 1 when `dot(up, sun) <
   −0.1`.
5. `aurora_disabled_when_strength_is_zero` — material `update`
   correctly propagates `aurora_strength = 0.0` from `PlanetConfig` to
   the GPU uniform.

### Integration / visual

- **agent-capture screenshots** at four canonical configurations:
  - `lat=70, lon=0, night side` → expects bright curtains.
  - `lat=70, lon=0, day side` (180° in time) → expects dark / no
    contribution.
  - `lat=0, lon=0, night side` → expects no contribution (equatorial).
  - **Orbital view**, camera at altitude 800 km above 70°N, looking
    nadir → expects a visible green ring around the pole.
- These captures live in `agent_captures/<timestamp>-aurora-*/`
  and are referenced from the new visual-showcases doc (see
  follow-up).

No new simulation `.simulation.ron` — aurora is rendering-only.

## Performance

- Single draw call, one material, no state churn between frames.
- 12 ray-march samples per fragment, on a shell that covers ~10-15%
  of pixels in typical surface views and ~30-40% in nadir orbital
  views.
- Budget: < 1 ms at 1080p on mid-range GPUs.
- No CPU work per frame beyond the 4-uniform upload.

## Error handling

- If `PlanetConfig` is missing → shell entity is not spawned (Startup
  system early-returns).
- If the inner-radius ray-sphere intersection has no real solutions
  (camera is well inside the inner shell looking up, ray exits through
  the outer shell without entering the inner) → fragment marches the
  full ray segment with `t_inner = 0`. Correctness preserved.
- `aurora_strength = 0.0` is a valid runtime state and yields zero
  emission with no NaN risk.

## File-level plan

```
NEW  src/lighting/aurora.rs             ~ 250 LOC
NEW  assets/shaders/aurora.wgsl         ~ 180 LOC
MOD  src/lighting/mod.rs                 +5 lines (mod + plugin wiring)
MOD  src/world/planet/config.rs          +4 fields on PlanetConfig
MOD  assets/data/*.planet.ron            +4 default fields (or use serde defaults so we don't have to touch RON)
MOD  docs/ROADMAP.md                     + "Aurora ✅" section
MOD  issues.json                         + AURORA-001 if we leave the
                                           multi-band / dynamic-field
                                           items as known follow-ups
```

Where possible we use `#[serde(default)]` on the new PlanetConfig
fields so existing RON files keep loading without edits.

## Out-of-scope follow-ups (filed under AURORA-001 in `issues.json`)

- Multi-band aurora (Jupiter-style).
- Spherical-harmonic magnetic-field model.
- Coupling to a future geomagnetic-storm system.
- True ray-marched volumetric self-occlusion (approach C from
  brainstorming).
- A scripted "visual showcase tour" runner that regression-captures
  one frame per visual feature, with the aurora as the first showcase
  to use it.
