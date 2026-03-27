# Audio Assets

Place `.ogg` files here for game audio. The audio system loads these paths:

## Ambient
- `ambient_wind.ogg` — looping wind/nature background

## Footsteps (per material category)
- `footstep_stone.ogg`
- `footstep_dirt.ogg`
- `footstep_wood.ogg`
- `footstep_sand.ogg`
- `footstep_water.ogg`

## Block Interaction
- `block_break.ogg` — block destroyed
- `block_place.ogg` — block placed

## UI
- `ui_click.ogg` — menu button click

All sounds are loaded at startup. Missing files are handled gracefully (logged
as warnings, no crash).
