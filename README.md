# Tekken 3 recomp module

Tekken 3 (Namco, 1998), statically recompiled for the TrimUI Brick. Deadstation 1.0.0 or newer hosts it when the game's Core is set to
it: Deadstation's menu, Options and Recomp Store stay Deadstation's, and this module is the game.

## Supported Formats

| Region | Format | SHA-1 |
|---|---|---|
| USA | CHD | `319ec7377eb0c3ef0b6049e440937d26ae743290` |

The SHA-1 is the CHD header's, shown by `chdman info` as **SHA1**. Other images are currently not supported.

## Features

- **Resolution:** the game drawn at its own resolution, or at twice it on the Brick's 1024x768 screen (Options).
- **Shadows:** the fighters' bodies cast their shadows on the floor along the stage's sun, each spot of floor
  darkened once by the sun's share of its light, in place of the game's flat black ones. Always on.
- **Lighting:** the fighters shade themselves and each other from the sun pixel by pixel (Options: Enhanced or Original).
- **Spark Lighting:** every hit spark and the game's effect light light the fighters and the floor round them in their
  colours, each casting its own planar shadows of the fighters on the floor, so a hit can throw several shadows at once.
  Options: 0 (none) to 10, applied at once.

- **Stage moods:** with Lighting Enhanced, stages get a fixed mood of their own: a golden dawn in Law's courtyard, a wet
  street on Lei's, torches in the temple, two searchlights patrolling Nina's stage in the dark, a storm with lightning
  beyond the hall's windows, dusk in the schoolyard.
- **Stage flavour:** something of its own on most stages: rain and mist on Lei's street, mist in the forest and on Nina's stage, snow at the palace, embers in the temple, petals at the school, a leaking roof in the storm hall, fireworks at the amusement park, cloud shadows on the rooftop and more.
- **Fight GUI:** the health bars, timer and names can be hidden (Options, applied at once).
- **Unlock Everything:** every fighter, costume and mode (Options, applied at once; saved with your game).
- **Controls:** every button can be remapped, set in Options. `module.cfg` declares them; the SDK's README explains how.

## Files

```
module.cfg          the manifest
store.json          what its page in the Recomp Store says
game.toml           the game: its disc, boot file and runtime settings (PSXRecomp)
game_options.toml   the game's own options PSXRecomp keeps
seeds/              the functions the translation starts from
symbols.toml, symbols/, psx_symbols.h   names found in the game's code
catalog_identity.json   the disc's identity, as PSXRecomp's probe found it
pipelines.keys      the GPU pipelines it draws, built ahead of use
CMakeLists.txt      the module's build
src/plugin.cpp      the module's additions, a plugin inside the game (its hooks: game.toml)
mods/               the plugin's package, which turns it on
sdk/                the runtime and port layer
```

## Building

From your own disc and its overlay captures in `overlays/` (they hold the game's code: never in git; the SDK's README says how):

```
git submodule update --init sdk
git -C sdk submodule update --init psxrecomp
sdk/tools/toolchain/toolchain.sh
sdk/tools/build-module.sh "/path/to/Tekken 3 (USA).chd"
sdk/tools/pack-module.sh
```

## Credits

Built with [PSXRecomp](https://github.com/RetroPortingToolKit/psxrecomp).

## Licence

PolyForm Noncommercial 1.0.0 (`LICENSE`).
