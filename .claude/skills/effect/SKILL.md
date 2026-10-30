---
name: effect
description: Apply a named video effect to an mp4 with `/effect <effect_name>`, or learn a new effect from a reference video's motion graphics (zoom punches, camera shake, flashes, glitches, RGB split, blur, vignette, grain, colour pulses) and save it for reuse. Use whenever the user types /effect, asks to "apply an effect/glitch/shake/VHS look to video.mp4", wants to "copy / recreate / extract the effect from this video", or wants to turn a reference clip's motion graphics into a reusable Python effect, even if they don't say "skill".
argument-hint: "<effect_name> [video.mp4] | learn <effect_name> <reference.mp4> | list | show <effect_name>"
---

# /effect

Learns effects from reference videos and replays them on other videos, in Python (OpenCV + numpy, ffmpeg for encode and audio).

An effect is a small JSON **spec**: keyframed tracks (zoom, shake, flash, glitch, rgb_split, ...) on the effect's own timeline, plus optional constants and an optional Python **plugin** for things that can't be measured (overlays, text, particles). Learned effects live in `${CLAUDE_SKILL_DIR}/effects/`; built-ins are in `scripts/presets.py`.

```
reference.mp4 --learn--> effects/<name>.json (+ optional <name>.py) --apply--> video_<name>.mp4
```

All commands go through one script (run from the user's project directory so relative video paths resolve):

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/effect.py <command> ...
```

Needs `opencv-python`, `numpy`, `scipy`, and `ffmpeg` on PATH. If an import fails, `pip install opencv-python numpy scipy`; if ffmpeg is missing, say so rather than working around it.

## Routing `$ARGUMENTS`

| User typed | Do |
|---|---|
| `/effect list` (or nothing) | `effect.py list`, show the table |
| `/effect show <name>` | `effect.py show <name>` |
| `/effect learn <name> <reference>` | follow **Learning** below |
| `/effect <name> [video]` | follow **Applying** below |

If the first word isn't `list`, `show` or `learn`, it is an effect name.

## Applying

1. Video: the second argument, else `video.mp4` in the working directory. If neither exists, look for a single `.mp4` in the directory; if still ambiguous, ask which file.
2. If the effect name is unknown, run `list` and suggest the closest names. Don't invent an effect.
3. Run `effect.py apply <name> <video>`. It writes `<video>_<name>.mp4` beside the input and keeps the audio. Pass `-o` only if the user named an output.
4. For long videos, do a quick preview first with `--seconds 3` so a wrong choice costs seconds rather than minutes. Mention the preview if you used one.
5. Report the output path, the effect's timing mode, and anything the user can tune. Only mention what's relevant:
   - `--intensity 1.5` scales every track (flash, zoom, blur, ...) away from neutral. Learned blur and flash tend to come out 30-50% weaker than the reference, so suggest 1.5-2 if the user says it looks too subtle.
   - `--set KEY=VALUE` (repeatable) overrides a plugin param for one run, e.g. `--set mode=around_subject`, `--set cards=12`. Values are parsed as JSON when possible.
   - `--timing loop|stretch|once` overrides how the effect's timeline maps onto the video: `loop` repeats it, `stretch` fits it to the whole video, `once` plays it a single time at `--start SECONDS`.

Looking at the result is cheap and worth it for anything the user will keep: `effect.py inspect <output.mp4> -n 8` writes a contact-sheet PNG you can open to confirm the effect landed where expected.

## Learning

`learn` measures what the reference does and writes the spec. It does not understand the footage, so it only recovers what leaves a measurable signature:

| Track | Measured from |
|---|---|
| `zoom`, `shift_x/y`, `rotate` | dominant global motion (feature tracking + RANSAC); slow drift is high-passed away so camera pans and estimator bias are not learned |
| `brightness`, `contrast`, `saturation` | deviation from a rolling-median baseline (flashes and dips; cuts are steps and don't count) |
| `blur` | sharpness loss, calibrated by re-blurring a typical frame |
| `rgb_split`, `rgb_split_y` | red/blue channel misregistration |
| `glitch` (+ `glitch_amp`) | rows that jump sideways relative to the previous frame |
| `vignette`, `grain` | clip-wide statistics |
| `grade` (opt-in `--grade`) | LAB mean/std look transfer; content dependent, so off by default |

Steps:

1. `effect.py learn <name> <reference> [--description "..."]`. Slugify the name (lowercase, underscores). Useful flags: `--timing loop|stretch|once` (default: loop if the reference is 4 s or shorter, else stretch), `--max-seconds` (default 20), `--motion full` (keep pans/zooms that accumulate instead of high-passing them), `--grade`.
2. Read the printed table. It lists each track, its peak, and how much of the time it is active. This is your evidence for what was learned.
3. Open `effects/<name>.reference.png` (contact sheet of the reference, timestamped) and compare it with the table. Ask: is there visible motion graphics the table doesn't explain?
   - **Everything visible is explained** (e.g. a punch-in plus flash shows up as zoom + brightness): done.
   - **Something is missing**, such as text, shapes, lower-thirds, light leaks, particles, film burn, scanlines, speed ramps, or transitions: the tracks can't capture it. Write a plugin layer (next section) rather than telling the user it's unsupported.
   - **Table shows nothing, or tracks that contradict the footage**: the reference is likely all overlays, or the footage itself moves a lot (handheld, fast action) and fooled the estimator. Say which, and offer the plugin route or a cleaner reference.
4. Verify by applying the new effect to a short clip: `effect.py apply <name> <some video> --seconds 3`, then `inspect` it. Tell the user honestly how close it is. A good learned effect reproduces timing and rough magnitude, not pixel-exact rendering.
5. Summarise: the effect name, what was captured, what wasn't, and how to use it (`/effect <name>`).

### Reference quality (tell the user if it matters)

Best results come from a short reference (1-10 s) where the effect is the main thing happening on fairly static footage. Strong scene motion, handheld footage and many hard cuts blur the line between "effect" and "content"; the dead-bands and median baselines reduce false positives but can't eliminate them.

## Plugin layer: for what can't be measured

When the reference has graphics beyond the tracks, author a Python plugin that draws them procedurally. This is also how to build an effect from a description with no reference at all.

1. Write `effects/<name>.py` defining `apply(frame, t, info)` (BGR uint8 in, BGR uint8 out) and optionally `setup(info)`. See [examples/plugin_example.py](examples/plugin_example.py) for the contract and a working overlay. `info` carries `w, h, fps, duration, n_frames, src` (path of the input video, so a plugin can sample stills from it), `params` (the spec's free-form `"params"` dict, the place for user-tunable settings), `effect_duration`, `timing` and `start` (also available in `setup`, so a plugin can lay out its timeline for loop vs once), and per frame `frame_index`, `effect_t` and `effect_duration`. Use `effect_t` for animation on the effect's timeline, `t` for absolute video time, and size everything relative to `frame.shape` so it works at any resolution. A plugin may return a completely new image: it doesn't have to composite over the frame. Derive randomness from `info["frame_index"]` so re-renders are identical.
2. Add `"plugin": "<name>.py"` to `effects/<name>.json`. If the effect has no measurable tracks, create the JSON by hand: `{"name": "...", "description": "...", "duration": 2.0, "timing": "loop", "tracks": {}, "plugin": "<name>.py"}`.
3. Match what you saw in the contact sheet (positions, colours, timings, easing). Preview with `--seconds 3` and `inspect`, then adjust.

Plugins run as ordinary Python with the user's permissions. Only write code that does image drawing; no network, no file writes.

## Bundled custom effects

- **`circular_ring`**: a 3D ring carousel of upright, double-sided cards. Card faces are set by the `faces` param: `"devops"` (the shipped default: 10 logos from `effects/icons/`, Docker, Kubernetes, AWS, Azure, GCP, Terraform, Packer, Terragrunt, GitHub, OPA, each on a clean tile with its name and a category tag), a list of names such as `["docker", "kubernetes", "aws"]` (any `<name>.png` in `effects/icons/`; transparent PNGs, wide wordmarks work best), or `"stills"` (stills sampled across the input video, `--set faces=stills`). The card count follows the logo list unless `cards` is set. Card look: `card_style` `"full"` (default) makes the face fill the whole card with rounded corners and a lime rim on the lit card, no frame, number or label; `"classic"` is the framed panel with numbers and tags. `corner_radius` (0.12 of the short side), `card_size` (`[360, 360]` full, `[340, 460]` classic), `logo_scale` (0.78, how much of the tile the logo fills). When the user asks for a different card look, change these params first (e.g. `--set corner_radius=0.3`, `--set 'card_size=[420,300]'`, `--set card_scale=0.9`) and only edit `circular_ring.py` for things they can't express. The logos are third-party trademarks copied from a local Go module cache (see `effects/icons/NOTICE.md`); mention that to the user if they plan to publish commercially. Low-resolution logos (~195px wide) look soft on big cards, so swap in better PNGs if the cards are large on screen. Constant drift plus stepped "push and dock" (8 steps per revolution, 5.6 s loop at 30 fps), lime spotlight and 44px lift on the front card, ground disc with a 96-tick scale ring, faded reflection, and a rolling step subtitle. In gallery mode it replaces the picture (audio is kept), so tell the user the original video is only visible as a faint blurred backdrop and as the card stills. Tunables in `effects/circular_ring.json` under `params`: `cards` (8), `radius` (700, keep radius / 340 >= 2 or neighbours collide), `tilt` (-9 degrees), `bg_video` (0.15, how much blurred live video shows behind), `reflection` (true), `names` (card titles, default "SHOT 01"...). `--timing stretch` makes the whole video exactly one revolution. **Around a talking head**: when the user says "around the talking head / speaker / person" (or the video is clearly a talking head and they want the ring around them), run it with `--set mode=around_subject` (or use the ready-made `circular_ring_head`). The original video stays live, the ring is centred on the head and follows it, the far half of the ring is composited under a person matte and the near half over it, and the cards are portrait crops of the speaker. It uses `scripts/subject.py`: MediaPipe selfie segmenter + YuNet face detector, both bundled offline in `models/` (licences in `models/NOTICE.md`), about 3 s of extra setup per 100 frames. Tunables: `ring_to_face` (2.1, ring radius as a multiple of face width; must clear the shoulders or the far cards hide behind the torso, so raise it for tight close-ups), `card_scale` (0.72), `cards` (10), `tilt` (-26), `ring_y` (0.1, centre below the chin in face heights), `dim` (0), `ground` (true, tick ring), `subtitle` (false). If no face is found it falls back to gallery mode and says so. Edges are those of a 256px segmenter, so fine hair is soft and a hand crossing the ring may be cut roughly; check a contact sheet before handing it over. Source: `effects/circular_ring.py`, a software port of a CSS 3D recipe; the geometry comments there explain why cards never mirror (back face = same content, flipped) and why the loop closes exactly.

- **`element_body_moves`** (and `element_body_moves_reference`): "contact-shadow-lift" roll call, a recipe for giving elements a *body*. Cards sit on an oblique table (camera rotateX 24 to 20 degrees); ONE card at a time rises (translateZ 96 plus scale 1.05, clamped so the on-screen lift is at least 12px) while its two table shadows (tight contact core, wide ambient) grow, soften and slide to the lower right; it drops back in 8f with a 2f micro-press and 5f rebound. The last card springs up, gets an accent outline and check badge, and hovers (~50f). Default `timing` is `once` (hover until the end of the video, fade out in the last 0.4 s); `--timing loop` lets the last card fall back too so the roll call repeats every 5 s. It composites over the input video (the video stays the backdrop), or over a desk with `--set background=desk`. `element_body_moves_reference` reproduces the look of the reference clip (`input/element-body-moves.mov`: light desk, "Pick your shot." headline, three recipe cards with number chips).
  - **16:9 and 9:16 are automatic**: units are relative to the frame's short side (1080), and the layout follows the orientation: a row of tall cards in 16:9, a stack of wide cards in 9:16, a grid above 4 cards (the layout is shrunk so lifted, perspective-enlarged cards stay inside the frame). Override with `--set layout=row|column|grid`, `--set cols=2`.
  - **Where the cards go**: `--set anchor=center|top|bottom` or `--set 'region=[x0,y0,x1,y1]'` (fractions of the frame). The default centre covers a talking head's face, so for talking heads use `anchor=bottom` (leaves room under the cards for the shadow too).
  - **Card content is user input** via `params.items`, a list of `{title, value, text, icon, chips, chips_on}` (`icon` is a logo key from `effects/icons/`; `chips` = number chips, `chips_on` = which are filled). Edit the effect's JSON for custom copy; `--set` takes JSON, e.g. `--set 'items=[{"title":"Step 1","value":"Build","text":"Compile and test."}]'`. `cards` sets the count (items cycle), `order` the call order, `theme` light|dark, `accent` a hex colour, `title`/`caption` an optional headline.
  - **Other knobs**: `view=2d` (flat: -28px and scale 1.08 instead of translateZ), `lift_z`, `lift_scale`, `tilt=[24,20]`, `persp`, `speed`, `shadow_scale` (auto: 1.0 desk, 1.4 dark, 1.8 over video because the recipe's alphas are tuned for a light desk), `background` video|desk|dark with `bg_blur`/`bg_dim`/`veil`, `corner_radius`, `intro`, `outro`.
  - Known limits from the recipe: lifts under 12 visible px read as a mistake (clamped); lifting two cards at once makes the audience lose who was called (the timeline never overlaps calls); the demo values were tuned on grey placeholders, so confirm on real footage. Pair it with a lift/drop sound if the video has sound design.
  - Needs Pillow for the text cards. Fonts (Inter, Space Mono, OFL) are in `fonts/`; the shared card helpers are in `scripts/cardkit.py` (also used by `circular_ring`).

## Spec reference (for hand-editing a learned effect)

```json
{
  "name": "zoom_flash", "description": "...",
  "duration": 1.0, "timing": "loop",
  "tracks": {
    "zoom": [[0, 1.0], [0.1, 1.22], [0.6, 1.0]],
    "brightness": [[0, 0], [0.05, 0.25], [0.3, 0]],
    "vignette": 0.3
  },
  "glitch_amp": 0.06
}
```

A track is either a list of `[seconds, value]` keyframes (linearly interpolated, held at the ends) or a constant. Units are resolution independent: `zoom` scale factor; `shift_x/y` fraction of frame width/height; `rotate` degrees; `blur` sigma as a fraction of width; `rgb_split` per-channel offset as a fraction of width; `glitch` 0-1; `brightness` additive 0-1; `contrast`/`saturation` multipliers; `vignette` 0-1; `grain` noise sigma 0-1. The full list with neutral values is `TRACKS` in `scripts/engine.py`. Built-ins in `scripts/presets.py` are worked examples.

## Known limits (say these rather than overselling)

- Effects are recovered as parameters, then re-rendered by this engine; the output resembles the reference's look and timing but isn't a pixel copy.
- No speed ramps / time remapping, and no per-region effects (masks). Overlays need a plugin.
- Rendering is CPU-only Python: roughly 30 fps at 720p for track effects, slower at 4K; `circular_ring` is about 9 fps at 1080p. Use `--seconds` previews.
- Output is H.264 (CRF 18) with AAC audio, whatever the input codec.
