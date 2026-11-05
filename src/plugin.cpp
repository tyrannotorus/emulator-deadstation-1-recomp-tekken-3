// Deadstation's plugin in the Tekken 3 recomp module (PSXRecomp's trusted static plugins: runtime/include/
// mod_plugins.h), activated by mods/preloaded/packages/deadstation.tekken3. Hooks at the game's own functions
// (symbols/game_names.tsv), listed in game.toml's mod_function_entry_funcs.
//
// Shadows along the sun: the game already draws each fighter's shadow as a planar projection (camera x P,
// P at 0x800AFB50, built by 0x80035F2C from the stage's "floor tilt" angles); P is built from the stage's sun
// instead (the angles the game lights the fighters with), so the shadows fall where the light comes from.
//
// Shadows that darken the floor: the game draws them as solid polygons in the stage's shadow colour, hiding the
// floor's texture. The 3D layer draws them instead, along the sun: the game's chain comes out of the
// ordering table and the floor darkens where the sun is blocked. Not a choice: the only shadows the module supports.
//
// The fight's sparks as lights (Spark Lighting, 0-10, applied at once): on the fighters and the floor, each casting its own
// planar shadows of the fighters on the floor (src/shaders/scene_cast.vert), as many as there are sparks.
//
// Lighting per part: self-shadowing and the hit flashes as coloured lights (below, at part_draw).
//
// DEADSTATION_TEKKEN3_STATS logs counts.
#include "cpu_state.h"
#include "mod_plugins.h"
#include "../sdk/brick/host_link.h"
#include "scene.h"
#include "../sdk/brick/psx_sdk.h"
#include "../sdk/brick/replay.h"

extern "C" {
#include "interrupts.h"
#include "psx_cycles.h"
#include "gpu_render.h"
extern uint8_t *g_psx_ram;
extern void (*psx_gpu_list_node_hook)(uint32_t addr);	// patch 0024
void psx_gpu_draw_offset(int32_t *x, int32_t *y);
extern uint32_t deadstation_game_frames;
void deadstation_scene_perspective(int on);
extern uint32_t i_mask;
uint32_t timers_cycles_to_irq(uint32_t i_mask);
uint16_t sio_get_pad_buttons_slot(int slot);
}

extern "C" int psx_native_cd_read(uint32_t lba, uint32_t dest, uint32_t bytes);
extern "C" int psx_cdrom_read_user_sector(uint32_t lba, uint8_t *out);
extern "C" void overlay_capture_before_dma(uint32_t load_addr, uint32_t size);
extern "C" void overlay_capture_on_dma(uint32_t load_addr, uint32_t size, const uint8_t *data);
extern "C" void dirty_ram_mark_executable_range(uint32_t phys, uint32_t len);
extern "C" void psx_write_word(uint32_t address, uint32_t value);
extern "C" uint8_t *memory_get_ram_ptr(void);

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

extern "C" void psx_dispatch_call(CPUState *cpu, uint32_t addr, uint32_t return_addr);

namespace
{
constexpr const char *Id = "deadstation.tekken3";
constexpr uint32_t ShadowProjection = 0x80035F2C;	// P (0x800AFB50) from a stage's tilt angles
constexpr uint32_t StageLights = 0x80096C78;		// 0x1C per stage: +0 ambient x 16, +8, +9, +10 the sun's blue, green, red, +12 a, +14 b (its angles)
int currentStage = -1;								// as the game last set it up
constexpr uint32_t SinTable = 0x8001E8C4, CosTable = 0x8001F0C4;	// s16[4096], 4.12
constexpr uint32_t P = 0x800AFB50;					// MATRIX: s16 m[3][3]

// fighter_render (0x8003B4F8): the shadow's packets (0x80037B28: the fighter's shadow mesh, one of its two packet
// sets, chained to what the ordering table entry held), the fighter's parts, then (shadows on) the entry set to
// the chain's last packet, then 0x8003172C(fighter): the place to finish the shadow
constexpr uint32_t AfterShadow = 0x8003172C, AfterShadowReturn = 0x8003B5A0;
constexpr uint32_t OrderingTables = 0x800A8C54;		// *(*this + 4) + 0xFBC: the shadows' entry
constexpr uint32_t ShadowEntry = 0xFBC;
constexpr uint32_t PacketSet = 0x800ADEFC;			// which of the two packet sets (0, 1) this frame
constexpr uint32_t ShadowsOn = 0x800ADFD0;
constexpr uint32_t GameMode = 0x800AFA88;			// 8: another shadow (0x800793B8), left alone
constexpr uint32_t FighterShadowMesh = 0x1274;		// fighter -> its shadow mesh (+0, +4: its packet sets)

bool stats = false;
unsigned statShadows = 0, statPolygons = 0, statParts = 0;
bool sceneCapture = false;		// the fighters' 3D layer for the presenter (per-pixel self-shadowing)
uint32_t gameColours[5];	// the game's LC (0x8009E5C0), as the GTE packs it
void readGameColours();

uint32_t rw(uint32_t a) { return psx_mod_read_word(a); }
uint32_t ram(uint32_t link) { return 0x80000000u | (link & 0xFFFFFF); }

int16_t sin12(uint32_t angle) { return (int16_t)psx_mod_read_half(SinTable + (angle & 0xFFF) * 2); }
int16_t cos12(uint32_t angle) { return (int16_t)psx_mod_read_half(CosTable + (angle & 0xFFF) * 2); }

// 0x80035F2C(stage), with the sun's angles: the light's direction as the game computes it, then P as the game
// builds it from a direction (x - y Lx/Ly, 0, z - y Lz/Ly)
int onShadowProjection(struct CPUState *cpu, uint32_t)
{
	currentStage = (int)cpu->gpr[4];
	const uint32_t record = StageLights + cpu->gpr[4] * 0x1C;
	const uint32_t a = psx_mod_read_half(record + 12), b = psx_mod_read_half(record + 14);
	const int32_t lx = (cos12(b - 0x800) * cos12(a)) >> 12;
	const int32_t ly = sin12(a);
	const int32_t lz = (sin12(b - 0x800) * cos12(a)) >> 12;
	if (ly == 0)
		return 0;		// the sun in the floor's plane: the game's own
	const int16_t m[9] = { 0x1000, (int16_t)-((lx << 12) / ly), 0, 0, 0, 0, 0, (int16_t)-((lz << 12) / ly), 0x1000 };
	for (int i = 0; i < 9; i++)
		psx_mod_write_half(P + i * 2, (uint16_t)m[i]);
	return 1;
}

void onAfterShadow(struct CPUState *cpu, uint32_t)
{
	if (cpu->gpr[31] == AfterShadowReturn)
	{
		readGameColours();
		std::memcpy(&cpu->gte_ctrl[16], gameColours, sizeof(gameColours));
	}
	// the layer draws the fighters' sun shadows on the floor: the game's chain out of the ordering table
	if (!sceneCapture || cpu->gpr[31] != AfterShadowReturn || rw(ShadowsOn) == 0 || rw(GameMode) == 8)
		return;
	const uint32_t set = rw(PacketSet) & 1;
	const uint32_t entry = rw(rw(OrderingTables) + 4) + ShadowEntry;
	const uint32_t mesh = rw(cpu->gpr[4] + FighterShadowMesh);
	const uint32_t first = rw(mesh + 4 * set), head = rw(entry);
	if ((first & 0xFF000000) != 0x80000000)
		return;
	// the chain, from the entry's packet back to the first: untextured polygons only, or it is not the chain
	// this knows and is left alone
	unsigned n = 0;
	for (uint32_t p = ram(head);; p = ram(rw(p)))
	{
		const uint8_t code = psx_mod_read_byte(p + 7);
		if ((code & 0xE0) != 0x20 || (code & 0x04) || ++n > 2048)
			return;
		if (p == first)
			break;
	}
	psx_mod_write_word(entry, (head & 0xFF000000) | (rw(first) & 0xFFFFFF));
	statShadows++;
	statPolygons += n;
}

// ---- Lighting, per part (fighter_draw_parts 0x80037924: for each of the 22 slots, the light matrix = the game's
// world lights (0x8009E5A0: rows sun, effect light, fill) x the bone's rotation, the bone's view matrix, then
// part_draw (0x80036CAC), whose NCCT lights the part's vertices by their normals with the GTE's L, LC and BK).
// At part_draw's entry the part's lights are changed:
//  - self-shadowing: the sun's colour (LC column 1) dimmed as far as the fighters' other parts (spheres where
//    they were the frame before) stand between this part and the sun;
//  - flash lights: light 2 (the game's effect light, 0x80039F0C, which it aims once per fighter) aimed at this
//    part from the effect light and from each hit spark showing (0x800A3408), each falling off with the square
//    of its distance, their colours summed.
// After each fighter the game's colours go back into the GTE.
constexpr uint32_t PartDraw = 0x80036CAC, PartDrawReturn = 0x80037A08;	// (its first call per slot)
constexpr uint32_t PartDrawReturn2 = 0x80037A38;		// (the slot's second mesh, when it has one)
constexpr uint32_t WorldLights = 0x8009E5A0, LightColours = 0x8009E5C0;	// MATRIX each
constexpr uint32_t EffectLightAt = 0x8009E5E0, EffectLightFrames = 0x8009E5EC;	// 3 x s32; frames left of 0x30
constexpr uint32_t Camera = 0x800ADF70;				// MATRIX: world -> view
// (0x80076E70 draws them) +2 frame (from 1), +4 frames, +6, +8 a frame's size in texels, +0x18, +0x1A its place in VRAM,
// +0x20, +0x22 its CLUT's, +0x24 x,y,z (128ths), +0x3E 1: added, 0: averaged (smoke)
constexpr uint32_t Sparks = 0x800A3408, SparkSlots = 3, SparkSize = 0x40;
constexpr uint32_t SharedSparkGrids = 0x800279EC + 0x60, SharedSparkCount = 4;	// 12 bytes each: CLUT x, y, ...

constexpr double PartRadius = 110;		// a part as an occluder (the game's units: mm)
constexpr float SparkStandoff = 300;	// a flash out of the point of the hit toward the eye, in front of the bodies, as its sprite (the Pixi port's)
int sparkLevel = 5;						// Spark Lighting, 0: none
float sparkGain() { return sparkLevel * 0.4f; }	// an enhancement that shows: at 5 (as it comes) twice the Pixi port's brightest, at 10 four times
int testLights = 0;						// DEADSTATION_TEKKEN3_TESTLIGHTS (tests): that many flashes always on, round the fighters
float testMid[3] = { 0, 0, 0 };			// the fighters' middle last frame, view space
constexpr float ShadowLength = 7000;		// the longest sun shadow drawn on the floor (a low sun), from the fighter's footprint
constexpr float ShadowMargin = 500;		// round the fighters' footprints and their shadows (soft edges, limbs between corners)
constexpr float FloorReach = 3000;		// the floor a flash lights round it (the layer's light reaches 3 m), each way

struct V3
{
	double x = 0, y = 0, z = 0;
};
V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

int16_t lo16(uint32_t v) { return (int16_t)(v & 0xFFFF); }
int16_t hi16(uint32_t v) { return (int16_t)(v >> 16); }
uint32_t pack16(int a, int b) { return (uint32_t)(uint16_t)(int16_t)a | (uint32_t)(uint16_t)(int16_t)b << 16; }

// the GTE's 3x3 control matrices are 9 s16 packed in 5 words (row by row)
void unpack(const uint32_t *w, double m[3][3])
{
	const int16_t e[9] = { lo16(w[0]), hi16(w[0]), lo16(w[1]), hi16(w[1]), lo16(w[2]), hi16(w[2]), lo16(w[3]), hi16(w[3]), lo16(w[4]) };
	for (int i = 0; i < 9; i++)
		m[i / 3][i % 3] = e[i];
}
void pack(const int e[9], uint32_t *w)
{
	w[0] = pack16(e[0], e[1]);
	w[1] = pack16(e[2], e[3]);
	w[2] = pack16(e[4], e[5]);
	w[3] = pack16(e[6], e[7]);
	w[4] = (uint32_t)(uint16_t)(int16_t)e[8];
}

bool invert3(const double a[3][3], double out[3][3])
{
	double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
		a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
	if (std::fabs(det) < 1e-12)
		return false;
	double k = 1.0 / det;
	out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * k;
	out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * k;
	out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * k;
	out[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * k;
	out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * k;
	out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * k;
	out[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * k;
	out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * k;
	out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * k;
	return true;
}


void readGameColours()
{
	int e[9];
	for (int i = 0; i < 9; i++)
		e[i] = (int16_t)psx_mod_read_half(LightColours + 2 * i);
	pack(e, gameColours);
}

void relight(struct CPUState *cpu)
{
	if (cpu->gpr[31] != PartDrawReturn)
		return;
	// the presenter's layer lights the flashes and the self-shadowing per pixel: here only the game's colours, its second
	// light off (no world positions, matrices or inverses for it)
	readGameColours();
	cpu->gte_ctrl[9] &= 0xFFFF, cpu->gte_ctrl[10] = 0;
	cpu->gte_ctrl[16] = gameColours[0] & 0xFFFF, cpu->gte_ctrl[17] = gameColours[1], cpu->gte_ctrl[18] = gameColours[2] & 0xFFFF0000u;
	cpu->gte_ctrl[19] = gameColours[3] & 0xFFFF, cpu->gte_ctrl[20] = gameColours[4];
	if (stats)
		statParts++;
}

// ---- The fighters' 3D layer (for lighting the presented frame per pixel): every polygon part_draw draws, as the
// game projected it. part_draw (a0: the part's 3DMK record; its +12: the polygon block) runs here, then its
// polygons are read: four kinds (flat triangles, flat quads, smooth triangles, smooth quads), each a u32 count
// and entries of 8 bytes (12 for smooth quads) whose first word holds the corners' vertex indices (bits 2-8,
// 9-15, 16-22, and 25-31 for quads); a vertex
// i's screen position is at 0x1F800000 + 4i and its depth (SZ) at 0x1F8001B8 + 2i, where part_draw left them.
using deadstation::scene::Corner;
deadstation::scene::Frame building;	// this game frame's, published when the next begins
uint32_t sceneSet = ~0u;
uint64_t sceneSerial = 0;
bool inPartDraw = false;

float drawnFloor[4] = { 0, 0, 0, 0 };	// the floor's grid as the game drew it this frame, in the world: x0, x1, z0, z1
bool drawnFloorFresh = false;			// (none drawn natively: no limit)

// ---- Stage moods ----
//
// One fixed mood per stage, made of what the game already lights with: its record of the stage's lights (0x80096C78: the sun's
// colour and angles and the ambient, patched once before any stage loads, so the fighters, the sun's shadows and the game's own
// sums all follow), a tint on the backdrop and the floor (their native polygons' colours, so the HUD keeps its own), and lamps:
// point lights as the sparks are, each casting its own shadows. The arena has no fixed place (the floor's pattern repeats and the
// backdrop follows the fighters), so a lamp keeps its place from the fighters' middle, in the world's axes.
struct Lamp
{
	float x, y, z;			// from the fighters' middle (y < 0: above the floor)
	float colour[3];
	float strength;
	float reach;
	int flicker;			// 0 steady, 1 a flame, 2 a failing tube
};
// A searchlight on its mast: where its lamp stands from the fighters' middle, and the pool of light it patrols the floor with round them
struct Searchlight
{
	float x, y, z;
	float colour[3];
	float strength;
	float reach;
	float radius;			// of the pool, square on to the beam
	float sweepX, sweepZ;	// how far it patrols each way
	float rateX, rateZ;		// how fast (radians a second)
	float phase;
};
struct Mood
{
	int stage;
	int ambient;			// 0-255, -1 as the game has it
	int sun[3];				// red, green, blue, -1 as the game has it
	int elevation;			// 4096ths of a turn, -1 as the game has it
	float backdrop[3], floor[3];	// multiplied into their colours
	bool lightning;
	int lamps;
	Lamp lamp[3];
	int searchlights;
	Searchlight searchlight[2];
};
constexpr Mood Moods[] = {
	// the temple: three torches, flickering, shadows swaying; deep shade between them
	{ 6, 80, { -1, -1, -1 }, -1, { 1.0f, 0.9f, 0.78f }, { 1.05f, 0.88f, 0.72f }, false, 3, {
		{ -2600, -2300, -700, { 1.0f, 0.5f, 0.15f }, 11, 5000, true },
		{ 2600, -2300, -700, { 1.0f, 0.46f, 0.12f }, 11, 5000, true },
		{ 0, -2500, 2600, { 1.0f, 0.55f, 0.18f }, 9, 5000, true } } },
	// the guarded yard at night: the main light down, two searchlights patrolling the floor, each throwing its own shadows
	{ 9, 38, { 0x28, 0x2C, 0x40 }, -1, { 0.36f, 0.4f, 0.56f }, { 0.36f, 0.39f, 0.46f }, false, 0, {}, 2, {
		{ -5200, -5200, 3400, { 0.92f, 0.96f, 1.f }, 700, 13000, 1700, 2600, 1800, 0.43f, 0.31f, 0.f },
		{ 5200, -5200, -3400, { 0.92f, 0.96f, 1.f }, 700, 13000, 1700, 2600, 2000, 0.37f, 0.47f, 2.4f } } },
	// the ruin in the storm: cold light and lightning
	{ 12, 90, { 0x90, 0xA8, 0xE0 }, -1, { 0.72f, 0.78f, 0.95f }, { 0.74f, 0.8f, 0.95f }, true, 0, {} },
	// the lab: two cold tubes overhead, one of them failing
	{ 14, -1, { -1, -1, -1 }, -1, { 1.f, 1.f, 1.f }, { 1.f, 1.f, 1.f }, false, 2, {
		{ -1700, -3000, 600, { 0.82f, 1.f, 0.9f }, 16, 5500, 0 },
		{ 1800, -3000, -500, { 0.82f, 1.f, 0.9f }, 16, 5500, 2 } } },
	// Law's courtyard at dawn: the sun just up, low and gold, long shadows, the sky and stone warm with it
	{ 1, 92, { 0xFF, 0xAC, 0x48 }, 0xA0, { 1.12f, 0.82f, 0.50f }, { 1.20f, 0.90f, 0.56f }, false, 0, {} },
	// Lei's street in the rain: a grey sky, the road dark and wet
	{ 5, -1, { 0xA8, 0xB4, 0xD0 }, -1, { 0.84f, 0.88f, 0.98f }, { 0.68f, 0.72f, 0.84f }, false, 0, {} },
	// the schoolyard at dusk: a low orange sun, long shadows
	{ 7, 90, { 0xFF, 0x98, 0x50 }, 0xB0, { 1.08f, 0.86f, 0.72f }, { 1.12f, 0.85f, 0.68f }, false, 0, {} },
};
bool moodsPatched = false;
float lightningNow = 0;		// 0-1: how bright the strike is this frame
uint32_t flavourClock = 0;	// the game's frames: what the stages' flavours move by (nothing of the game's own)

inline uint32_t mix32(uint32_t v)
{
	v ^= v >> 16, v *= 0x7FEB352Du, v ^= v >> 15, v *= 0x846CA68Bu, v ^= v >> 16;
	return v;
}

// The amusement park's fireworks (stage 3): a few bursts on their own beats, each opening in the sky ahead of the camera as it is then and
// from then on a place in the world, a ball of sparks the camera pans and tilts across, behind the buildings and in front of the sky; each
// its colour, and a flash of that light on the floor and the fighters as it opens
constexpr int FireworkStage = 3, Fireworks = 3;
constexpr float FireworkLife = 1.7f, FireworkDistance = 26000.f;
struct Firework
{
	bool live;
	uint32_t round, seed;
	float age;				// seconds since it opened
	float x, y, z, radius;	// in the world
	float dirX, dirZ;		// which way, from the fighters
	float colour[3];
};
Firework fireworks[Fireworks] = {};

// The rooftop's cloud shadows (stage 11): a floor vertex's colour darkened as clouds pass overhead
constexpr int CloudStage = 11;
uint32_t cloudShade(uint32_t rgb, float x, float z)
{
	const float t = (float)flavourClock / 60.f;
	const float n = std::sin(x * 0.0008f + t * 0.35f) * std::sin(z * 0.00065f - t * 0.27f) + 0.6f * std::sin((x + z) * 0.00041f + t * 0.21f);
	const float a = std::clamp((n - 0.1f) / 0.6f, 0.f, 1.f);
	const float k = 1.f - 0.5f * a * a * (3.f - 2.f * a);
	uint32_t out = 0;
	for (int i = 0; i < 3; i++)
		out |= (uint32_t)((float)((rgb >> (8 * i)) & 255) * k) << (8 * i);
	return out;
}
uint32_t moodClock = 0;

// a searchlight now: its lamp and the point on the floor it aims at
void searchlightNow(const Searchlight& l, float midX, float midZ, float lamp[3], float aim[3])
{
	const float t = (float)moodClock / 60.f;
	lamp[0] = midX + l.x, lamp[1] = l.y, lamp[2] = midZ + l.z;
	aim[0] = midX + l.sweepX * std::sin(t * l.rateX + l.phase), aim[1] = 0.f, aim[2] = midZ + l.sweepZ * std::sin(t * l.rateZ + l.phase * 1.7f);
}
bool testStrike = false;	// DEADSTATION_TEKKEN3_TESTSTRIKE (tests): the lightning held on, to see it
bool noMoods = false;		// DEADSTATION_SDK_OFF=moods: the stages as the game lights them

const Mood *currentMood()
{
	if (noMoods)
		return nullptr;
	for (const Mood& m : Moods)
		if (m.stage == currentStage)
			return &m;
	return nullptr;
}

// the moods into the game's records, once, before a stage loads
void patchMoods()
{
	if (moodsPatched || noMoods)
		return;
	moodsPatched = true;
	for (const Mood& m : Moods)
	{
		const uint32_t record = StageLights + (uint32_t)m.stage * 0x1C;
		if (m.ambient >= 0)
			psx_mod_write_half(record, (uint16_t)(m.ambient << 4));
		for (int i = 0; i < 3; i++)
			if (m.sun[i] >= 0)
				psx_mod_write_byte(record + 10 - (uint32_t)i, (uint8_t)m.sun[i]);
		if (m.elevation >= 0)
			psx_mod_write_half(record + 12, (uint16_t)m.elevation);
	}
}

// a backdrop's or floor's colour in the stage's mood; in a strike, brighter and whiter
uint32_t moodColour(uint32_t rgb, bool floor)
{
	const Mood *m = currentMood();
	if (m == nullptr)
		return rgb;
	const float *t = floor ? m->floor : m->backdrop;
	const float strike = 1.0f + 1.3f * lightningNow;
	uint32_t out = 0;
	for (int i = 0; i < 3; i++)
	{
		const float c = (float)((rgb >> (8 * i)) & 255) * t[i] * strike;
		out |= (uint32_t)std::min(c, 255.f) << (8 * i);
	}
	return out;
}

// the lightning: now and then two quick flashes, gone in a sixth of a second
float lightningLevel()
{
	static uint32_t nextStrike = 240, random = 0x2545F491;
	static const float Pulse[] = { 1.0f, 0.3f, 0.05f, 0.85f, 0.55f, 0.35f, 0.2f, 0.1f, 0.05f };
	if (testStrike)
		return 1.f;
	if (moodClock < nextStrike)
		return 0;
	const uint32_t at = moodClock - nextStrike;
	if (at < sizeof(Pulse) / sizeof(Pulse[0]))
		return Pulse[at];
	random ^= random << 13, random ^= random >> 17, random ^= random << 5;
	nextStrike = moodClock + 300 + random % 600;	// 5 to 15 seconds on
	return 0;
}

// What a spark's frame glows: its texels (4-bit, through its CLUT), each counted by how bright it is, so its faint edge
// counts for little. The colours summed, for the spark's whole life: a spark keeps one colour, not its white-hot start's.
struct SparkGlow
{
	float sum[3] = { 0, 0, 0 };
	int lastFrame = 0;
} sparkGlows[SparkSlots];

bool sparkFrameColour(uint32_t slot, float out[3])
{
	const uint16_t *vram = deadstation_vram_view();
	if (vram == nullptr)
		return false;
	const int x = (uint16_t)psx_mod_read_half(slot + 0x18) & 0x3FF, y = (uint16_t)psx_mod_read_half(slot + 0x1A) & 0x1FF;
	const int cx = (uint16_t)psx_mod_read_half(slot + 0x20) & 0x3F0, cy = (uint16_t)psx_mod_read_half(slot + 0x22) & 0x1FF;
	const int w = std::min<int>((uint16_t)psx_mod_read_half(slot + 6) & 0xFF, 64), h = std::min<int>((uint16_t)psx_mod_read_half(slot + 8) & 0xFF, 64);
	const uint16_t *clut = vram + cy * 1024 + cx;
	out[0] = out[1] = out[2] = 0;
	for (int j = 0; j < h; j++)
		for (int i = 0; i < w; i++)
		{
			const uint16_t word = vram[((y + j) & 0x1FF) * 1024 + ((x + i / 4) & 0x3FF)];
			const uint16_t c = clut[(word >> ((i & 3) * 4)) & 15];
			if (c == 0)
				continue;
			const float r = (float)(c & 31), g = (float)((c >> 5) & 31), b = (float)((c >> 10) & 31);
			const float weight = r + g + b;
			out[0] += r * weight, out[1] += g * weight, out[2] += b * weight;
		}
	return out[0] + out[1] + out[2] > 0;
}

// The frame's flashes, for the layer: the game's effect light and every hit spark showing, as view-space point
// lights (colour x strength, 1 = the sun at a metre; a flash at its brightest twice that, as the Pixi port's),
// brightest first, and the floor round them (the game's floor: the plane y = 0), one quad.
struct FloorView
{
	float cam[9], camT[3], yScale, ofx, ofy, h;
	float x0, x1, z0, z1;	// round the flashes (empty: none)
	bool valid = false;
} floorView;

void flashes(deadstation::scene::Frame& f, const float cam[9], float yScale, float ofx, float ofy, float h)
{
	f.lights.clear();
	const float camT[3] = { (float)(int32_t)rw(Camera + 20), (float)(int32_t)rw(Camera + 24), (float)(int32_t)rw(Camera + 28) };
	auto toView = [&](const float w[3], float v[3]) {
		for (int i = 0; i < 3; i++)
			v[i] = cam[3 * i] * w[0] + cam[3 * i + 1] * w[1] + cam[3 * i + 2] * w[2] + camT[i];
	};
	struct Flash
	{
		float world[3], colour[3], strength;
		float reach = FloorReach;
		bool lamp = false;
		bool spot = false;			// a searchlight: lights only its cone, down to aim
		float aim[3] = { 0, 0, 0 };
		float radius = 0;
	};
	Flash found[SparkSlots + 1 + 4 + 3 + 1 + Fireworks];
	unsigned n = 0;
	const float gain = sparkGain();
	const int effectFrames = (int32_t)rw(EffectLightFrames);
	Flash *effect = nullptr;
	// the colour of the fighter's own spark that brought the effect light, once one has shown
	static float ownSpark[3];
	static bool ownSparkSeen = false;
	if (effectFrames <= 0)
		ownSparkSeen = false;
	if (gain > 0 && testLights > 0)
	{
		// (tests) that many at once, in four colours, over the fighters' middle: what the worst moment costs and looks like
		double c[3][3], inv[3][3];
		for (int i = 0; i < 9; i++)
			c[i / 3][i % 3] = cam[i];
		if (invert3(c, inv))
		{
			const double d[3] = { testMid[0] - camT[0], testMid[1] - camT[1], testMid[2] - camT[2] };
			static const float at[4][3] = { { 0, -1100, 0 }, { 900, -1400, 200 }, { -900, -1400, -200 }, { 0, -1800, 700 } };
			static const float colours[4][3] = { { 1, 0.8f, 0.5f }, { 1, 0.15f, 0.1f }, { 0.15f, 0.3f, 1 }, { 0.2f, 1, 0.3f } };
			for (int k = 0; k < testLights && k < 4; k++)
			{
				Flash& t = found[n++];
				for (int i = 0; i < 3; i++)
				{
					t.world[i] = (float)(inv[i][0] * d[0] + inv[i][1] * d[1] + inv[i][2] * d[2]);
					t.colour[i] = colours[k][i];
				}
				t.world[1] = at[k][1], t.world[0] += at[k][0], t.world[2] += at[k][2];
				t.strength = 2.0f - 0.01f * k;
			}
		}
	}
	else if (gain > 0 && effectFrames > 0)
	{
		Flash& e = found[n++];
		effect = &e;
		for (int i = 0; i < 3; i++)
			e.world[i] = (float)(int32_t)rw(EffectLightAt + 4 * i);
		e.colour[0] = (int16_t)psx_mod_read_half(LightColours + 2) / 4096.0f;
		e.colour[1] = (int16_t)psx_mod_read_half(LightColours + 8) / 4096.0f;
		e.colour[2] = (int16_t)psx_mod_read_half(LightColours + 14) / 4096.0f;
		// fading as the sparks do, so the light goes with the spark rather than glowing on after it
		const float effectLeft = effectFrames / 48.0f;
		e.strength = 2.0f * effectLeft * effectLeft;
	}
	for (uint32_t k = 0; k < SparkSlots && testLights == 0; k++)
	{
		const uint32_t slot = Sparks + k * SparkSize;
		const int frame = (int16_t)psx_mod_read_half(slot + 2), frames = (int16_t)psx_mod_read_half(slot + 4);
		SparkGlow& glow = sparkGlows[k];
		if (frame <= 0 || frames <= 0 || frame > frames)
		{
			glow = SparkGlow();
			continue;
		}
		// a new spark in this slot; this frame's colours added to its own
		if (frame == 1 || frame < glow.lastFrame)
			glow = SparkGlow();
		float now[3];
		const bool lit = frame != glow.lastFrame && sparkFrameColour(slot, now);
		if (lit)
			for (int i = 0; i < 3; i++)
				glow.sum[i] += now[i];
		glow.lastFrame = frame;
		// a fighter's own spark brings the game's effect light with it (its colour, its CLUT not one of the shared ones'): no
		// second light. Smoke (averaged) gives none.
		bool shared = false;
		const uint16_t cx = (uint16_t)psx_mod_read_half(slot + 0x20), cy = (uint16_t)psx_mod_read_half(slot + 0x22);
		for (uint32_t v = 0; v < SharedSparkCount; v++)
			shared = shared || (psx_mod_read_half(SharedSparkGrids + v * 12) == cx && psx_mod_read_half(SharedSparkGrids + v * 12 + 2) == cy);
		const bool added = (uint16_t)psx_mod_read_half(slot + 0x3E) == 1;
		const float brightest = std::max({ glow.sum[0], glow.sum[1], glow.sum[2] });
		if (stats)
			std::printf("deadstation tekken3: spark %u frame %d/%d mode %u at %u,%u clut %u,%u %s; now %.2f %.2f %.2f; whole %.2f %.2f %.2f; effect light %d colour %d %d %d\n", k, frame, frames,
					(unsigned)(uint16_t)psx_mod_read_half(slot + 0x3E), (unsigned)psx_mod_read_half(slot + 0x18), (unsigned)psx_mod_read_half(slot + 0x1A), cx, cy, shared ? "shared" : "own",
					lit ? now[0] / std::max({ now[0], now[1], now[2] }) : 0.f, lit ? now[1] / std::max({ now[0], now[1], now[2] }) : 0.f, lit ? now[2] / std::max({ now[0], now[1], now[2] }) : 0.f,
					brightest > 0 ? glow.sum[0] / brightest : 0.f, brightest > 0 ? glow.sum[1] / brightest : 0.f, brightest > 0 ? glow.sum[2] / brightest : 0.f, effectFrames, (int16_t)psx_mod_read_half(LightColours + 2), (int16_t)psx_mod_read_half(LightColours + 8), (int16_t)psx_mod_read_half(LightColours + 14));
		if (added && !shared && brightest > 0)
		{
			for (int i = 0; i < 3; i++)
				ownSpark[i] = glow.sum[i] / brightest;
			ownSparkSeen = true;
		}
		if (gain <= 0 || !added || !shared || brightest <= 0)
			continue;
		Flash& s = found[n++];
		for (int i = 0; i < 3; i++)
		{
			s.world[i] = (int32_t)rw(slot + 0x24 + 4 * i) / 128.0f;
			s.colour[i] = glow.sum[i] / brightest;
		}
		const float left = 1 - (frame - 1) / (float)frames;
		s.strength = 2.0f * left * left;
	}
	// the effect light in its spark's colour (without one, as for fire or a glow: the game's)
	if (effect != nullptr && ownSparkSeen)
		std::memcpy(effect->colour, ownSpark, sizeof(ownSpark));
	// the stage's mood: its lamps, its lightning; round the fighters' middle
	moodClock++;
	lightningNow = 0;
	if (const Mood *mood = currentMood())
	{
		const float midX = ((int32_t)rw(0x800AA190) + (int32_t)rw(0x800ABA1C)) * 0.5f, midZ = ((int32_t)rw(0x800AA198) + (int32_t)rw(0x800ABA24)) * 0.5f;
		for (int k = 0; k < mood->lamps; k++)
		{
			const Lamp& l = mood->lamp[k];
			Flash& f2 = found[n++];
			// a torch's flame: brighter and dimmer on no beat, moving a little, so its shadows sway
			const float t = (float)moodClock + (float)k * 37.f;
			float flame = l.flicker == 1 ? 0.8f + 0.14f * std::sin(t * 0.31f) + 0.09f * std::sin(t * 0.83f) + 0.05f * std::sin(t * 1.9f) : 1.f;
			// a failing tube: now and then a few moments' stutter
			if (l.flicker == 2 && std::sin(t * 0.021f) * std::sin(t * 0.137f) > 0.55f && (moodClock / 3) % 3 != 0)
				flame = 0.12f;
			const bool flames = l.flicker == 1;
			f2.world[0] = midX + l.x + (flames ? 45.f * std::sin(t * 0.47f) : 0.f);
			f2.world[1] = l.y + (flames ? 35.f * std::sin(t * 0.71f) : 0.f);
			f2.world[2] = midZ + l.z + (flames ? 45.f * std::sin(t * 0.59f) : 0.f);
			for (int i = 0; i < 3; i++)
				f2.colour[i] = l.colour[i];
			f2.strength = l.strength * flame;
			f2.reach = l.reach;
			f2.lamp = true;
		}
		for (int k = 0; k < mood->searchlights; k++)
		{
			const Searchlight& l = mood->searchlight[k];
			Flash& f2 = found[n++];
			searchlightNow(l, midX, midZ, f2.world, f2.aim);
			for (int i = 0; i < 3; i++)
				f2.colour[i] = l.colour[i];
			f2.strength = l.strength;
			f2.reach = l.reach;
			f2.radius = l.radius;
			f2.lamp = f2.spot = true;
		}
		if (mood->lightning)
		{
			lightningNow = lightningLevel();
			if (lightningNow > 0)
			{
				Flash& f2 = found[n++];
				f2.world[0] = midX - 2800, f2.world[1] = -6000, f2.world[2] = midZ + 1800;
				f2.colour[0] = 0.85f, f2.colour[1] = 0.9f, f2.colour[2] = 1.0f;
				f2.strength = 90.f * lightningNow;
				f2.reach = 13000;
				f2.lamp = true;
			}
		}
	}
	// a firework opening: its colour on the floor and the fighters, for a moment
	if (!noMoods && currentStage == FireworkStage)
	{
		const float midX = ((int32_t)rw(0x800AA190) + (int32_t)rw(0x800ABA1C)) * 0.5f, midZ = ((int32_t)rw(0x800AA198) + (int32_t)rw(0x800ABA24)) * 0.5f;
		for (const Firework& fw : fireworks)
		{
			if (!fw.live || fw.age > 0.5f)
				continue;
			const float left = 1.f - fw.age / 0.5f;
			Flash& f2 = found[n++];
			f2.world[0] = midX + fw.dirX * 3500.f, f2.world[1] = -4800, f2.world[2] = midZ + fw.dirZ * 3500.f;
			std::memcpy(f2.colour, fw.colour, sizeof(fw.colour));
			f2.strength = 34.f * left * left;
			f2.reach = 9500;
			f2.lamp = true;
		}
	}
	// the flashes first, brightest first (the brightest's shadow map); what doesn't fit is a lamp
	std::sort(found, found + n, [](const Flash& a, const Flash& b) {
		if (a.lamp != b.lamp)
			return b.lamp;
		return a.strength * (a.colour[0] + a.colour[1] + a.colour[2]) > b.strength * (b.colour[0] + b.colour[1] + b.colour[2]);
	});
	floorView.valid = false;
	float x0 = 1e30f, x1 = -1e30f, z0 = 1e30f, z1 = -1e30f;
	unsigned spots = 0;
	for (unsigned i = 0; i < n; i++)
		spots += found[i].spot ? 1 : 0;
	for (unsigned i = 0; i < n && f.lights.size() < deadstation::scene::MaxLights; i++)
	{
		// (room kept for the searchlights: they don't go out in a shower of sparks)
		if (!found[i].spot && f.lights.size() + spots >= deadstation::scene::MaxLights)
			continue;
		float v[3];
		toView(found[i].world, v);
		// (a searchlight's lamp may stand behind the eye: its pool doesn't)
		if (v[2] <= 0 && !found[i].spot)
			continue;
		const float k = found[i].strength * (found[i].lamp ? 1.f : gain);
		// out of the point of the hit, toward the eye: a flash inside a body would light nothing and cast everything
		const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] / (yScale * yScale) + v[2] * v[2]);
		const float keep = found[i].lamp ? 1.f : length > SparkStandoff * 2 ? 1 - SparkStandoff / length : 0.5f;
		f.lights.push_back({ v[0] * keep, v[1] / yScale * keep, v[2] * keep, found[i].colour[0] * k, found[i].colour[1] * k, found[i].colour[2] * k, found[i].reach });
		float atX = found[i].world[0], atZ = found[i].world[2], around = found[i].reach;
		if (found[i].spot)
		{
			float a[3];
			toView(found[i].aim, a);
			const float axis[3] = { a[0] - v[0], (a[1] - v[1]) / yScale, a[2] - v[2] };
			const float beam = std::max(std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]), 1.f);
			deadstation::scene::Light& light = f.lights.back();
			for (int j = 0; j < 3; j++)
				light.axis[j] = axis[j] / beam;
			light.cone = beam / std::sqrt(beam * beam + found[i].radius * found[i].radius);
			// the floor it can light: its pool, drawn out by the beam's slant
			atX = found[i].aim[0], atZ = found[i].aim[2];
			around = found[i].radius * 1.2f * beam / std::max(-found[i].world[1], 1.f);
		}
		x0 = std::min(x0, atX - around), x1 = std::max(x1, atX + around);
		z0 = std::min(z0, atZ - around), z1 = std::max(z1, atZ + around);
	}
	f.flashMap = n > 0 && !found[0].lamp;
	std::memcpy(floorView.cam, cam, sizeof(floorView.cam));
	std::memcpy(floorView.camT, camT, sizeof(camT));
	floorView.yScale = yScale, floorView.ofx = ofx, floorView.ofy = ofy, floorView.h = h;
	floorView.x0 = x0, floorView.x1 = x1, floorView.z0 = z0, floorView.z1 = z1;
	floorView.valid = true;
}

// The floor under the layer (the game's: the plane y = 0), as a grid of cells in front of the camera: round the
// flashes (their light's reach), and round the fighters when the layer draws their sun shadows on it.
void finishFloor(deadstation::scene::Frame& f)
{
	f.floor.clear();
	const FloorView& v = floorView;
	f.floorBase = 0;
	if (!v.valid)
		return;
	float x0 = 1e30f, x1 = -1e30f, z0 = 1e30f, z1 = -1e30f;
	// where the fighters' sun shadows can fall: each corner's footprint and its projection along the sun onto the
	// floor (y = 0, at most ShadowLength away), with a margin (view -> world)
	double c[3][3], inv[3][3];
	for (int i = 0; i < 9; i++)
		c[i / 3][i % 3] = v.cam[i];
	if (!invert3(c, inv))
		return;
	const float sun[3] = { (float)(int16_t)psx_mod_read_half(WorldLights), (float)(int16_t)psx_mod_read_half(WorldLights + 2),
		(float)(int16_t)psx_mod_read_half(WorldLights + 4) };
	const float sunLength = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
	const float sunAcross = std::sqrt(sun[0] * sun[0] + sun[2] * sun[2]);
	for (size_t k = 0; k < f.corners.size(); k += 7)
	{
		const deadstation::scene::Corner& q = f.corners[k];
		const double d[3] = { q.x - v.camT[0], q.y * v.yScale - v.camT[1], q.z - v.camT[2] };
		const float wx = (float)(inv[0][0] * d[0] + inv[0][1] * d[1] + inv[0][2] * d[2]);
		const float wy = (float)(inv[1][0] * d[0] + inv[1][1] * d[1] + inv[1][2] * d[2]);
		const float wz = (float)(inv[2][0] * d[0] + inv[2][1] * d[1] + inv[2][2] * d[2]);
		// along the sun from the corner down to the floor (none with the sun at or below the horizon), clamped
		float sx = wx, sz = wz;
		if (sunLength > 0 && sun[1] < 0 && wy < 0)
		{
			float t = wy / sun[1];
			if (sunAcross * t > ShadowLength)
				t = ShadowLength / sunAcross;
			sx = wx - sun[0] * t, sz = wz - sun[2] * t;
		}
		x0 = std::min({ x0, wx - ShadowMargin, sx - ShadowMargin }), x1 = std::max({ x1, wx + ShadowMargin, sx + ShadowMargin });
		z0 = std::min({ z0, wz - ShadowMargin, sz - ShadowMargin }), z1 = std::max({ z1, wz + ShadowMargin, sz + ShadowMargin });
	}
	// no further than the floor the game draws this frame: past it lies the backdrop
	float lx0 = v.x0, lx1 = v.x1, lz0 = v.z0, lz1 = v.z1;	// round the lights
	if (drawnFloorFresh)
	{
		x0 = std::max(x0, drawnFloor[0]), x1 = std::min(x1, drawnFloor[1]);
		z0 = std::max(z0, drawnFloor[2]), z1 = std::min(z1, drawnFloor[3]);
		lx0 = std::max(lx0, drawnFloor[0]), lx1 = std::min(lx1, drawnFloor[1]);
		lz0 = std::max(lz0, drawnFloor[2]), lz1 = std::min(lz1, drawnFloor[3]);
	}
	drawnFloorFresh = false;
	const float up[3] = { 0, -1, 0 };
	float nv[3];
	for (int i = 0; i < 3; i++)
		nv[i] = v.cam[3 * i] * up[0] + v.cam[3 * i + 1] * up[1] + v.cam[3 * i + 2] * up[2];
	nv[1] /= v.yScale;
	const float nn = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
	// the plane the flashes cast onto, and where a point on it lies on the screen (the world's origin is on the floor)
	for (int i = 0; i < 3; i++)
		f.floorPlane[i] = nv[i] / nn;
	f.floorPlane[3] = (nv[0] * v.camT[0] + nv[1] * v.camT[1] / v.yScale + nv[2] * v.camT[2]) / nn;
	f.project[0] = v.ofx, f.project[1] = v.ofy, f.project[2] = v.h, f.project[3] = v.yScale;
	auto corner = [&](float wx, float wz, deadstation::scene::Corner& out) {
		float p[3];
		for (int i = 0; i < 3; i++)
			p[i] = v.cam[3 * i] * wx + v.cam[3 * i + 2] * wz + v.camT[i];
		if (p[2] < 64)
			return false;
		out = { v.ofx + p[0] * v.h / p[2], v.ofy + p[1] * v.h / p[2], p[0], p[1] / v.yScale, p[2], nv[0] / nn, nv[1] / nn, nv[2] / nn };
		return true;
	};
	const auto grid = [&](float gx0, float gx1, float gz0, float gz1) {
		constexpr int Cells = 12;
		if (gx0 >= gx1 || gz0 >= gz1)
			return;
		for (int j = 0; j < Cells; j++)
			for (int i = 0; i < Cells; i++)
			{
				const float ax = gx0 + (gx1 - gx0) * i / Cells, bx = gx0 + (gx1 - gx0) * (i + 1) / Cells;
				const float az = gz0 + (gz1 - gz0) * j / Cells, bz = gz0 + (gz1 - gz0) * (j + 1) / Cells;
				deadstation::scene::Corner q[4];
				if (!corner(ax, az, q[0]) || !corner(bx, az, q[1]) || !corner(ax, bz, q[2]) || !corner(bx, bz, q[3]))
					continue;
				for (int k : { 0, 1, 2, 1, 3, 2 })
					f.floor.push_back(q[k]);
			}
	};
	grid(x0, x1, z0, z1);
	f.floorBase = (uint32_t)f.floor.size();
	if (!f.lights.empty())
		grid(lx0, lx1, lz0, lz1);
}

// A new game frame's layer begun when the packet set changes; the frame's header (the sun, the flashes) with its first
// part. The projection and the camera's vertical scale (its row 1's length) back; false: no projection yet.
bool sceneHeaderDone = false, modelsThisFrame = false, layerCleared = true;
bool sceneFrame(struct CPUState *cpu, float& ofx, float& ofy, float& h, float& yScale)
{
	bool& headerDone = sceneHeaderDone;
	static float headerOfx, headerOfy, headerH, headerY;
	const uint32_t set = rw(PacketSet) & 1;
	modelsThisFrame = true;
	if (headerDone && set == sceneSet)
	{
		ofx = headerOfx, ofy = headerOfy, h = headerH, yScale = headerY;
		return true;
	}
	if (set != sceneSet)
	{
		sceneSet = set;
		if (!building.corners.empty())
		{
			if (testLights > 0)
			{
				float sum[3] = { 0, 0, 0 };
				size_t counted = 0;
				for (size_t k = 0; k < building.corners.size(); k += 7, counted++)
				{
					const deadstation::scene::Corner& q = building.corners[k];
					sum[0] += q.x, sum[1] += q.y * headerY, sum[2] += q.z;
				}
				for (int i = 0; i < 3 && counted > 0; i++)
					testMid[i] = sum[i] / (float)counted;
			}
			finishFloor(building);
			deadstation::scene::publish(building);
		}
		building.corners.clear();
		building.serial = ++sceneSerial;
		headerDone = false;
	}
	// the projection, the camera's vertical scale (its row 1's length), the sun into view space
	ofx = (int32_t)cpu->gte_ctrl[24] / 65536.0f, ofy = (int32_t)cpu->gte_ctrl[25] / 65536.0f;
	h = (float)(uint16_t)cpu->gte_ctrl[26];
	float cam[9];
	for (int i = 0; i < 9; i++)
		cam[i] = (int16_t)psx_mod_read_half(Camera + 2 * i) / 4096.0f;
	yScale = std::sqrt(cam[3] * cam[3] + cam[4] * cam[4] + cam[5] * cam[5]);
	if (h <= 0 || yScale <= 0)
		return false;
	if (building.corners.empty())
	{
		float sw[3], sv[3];
		for (int i = 0; i < 3; i++)
			sw[i] = (int16_t)psx_mod_read_half(WorldLights + 2 * i);
		for (int i = 0; i < 3; i++)
			sv[i] = (cam[3 * i] * sw[0] + cam[3 * i + 1] * sw[1] + cam[3 * i + 2] * sw[2]) / (i == 1 ? yScale : 1.0f);
		const float n = std::sqrt(sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2]);
		for (int i = 0; i < 3; i++)
			building.sun[i] = n > 0 ? sv[i] / n : 0;
		building.width = (float)psx_mod_display_width();	// the screen the game draws to, as presented at 1x
		building.height = (float)psx_mod_display_height();
		// what stays in the shade (the background light and half the fill) against the sun, by their brightness
		auto bright = [](int r, int g, int b) { return 0.299f * r + 0.587f * g + 0.114f * b; };
		readGameColours();
		double lc[3][3];
		unpack(gameColours, lc);
		const float sunLight = bright((int)lc[0][0], (int)lc[1][0], (int)lc[2][0]);
		const float fill = bright((int)lc[0][2], (int)lc[1][2], (int)lc[2][2]);
		const float back = bright((int32_t)cpu->gte_ctrl[13], (int32_t)cpu->gte_ctrl[14], (int32_t)cpu->gte_ctrl[15]);
		building.ambient = sunLight > 0 ? (back + 0.5f * fill) / sunLight : 1.0f;
		flashes(building, cam, yScale, ofx, ofy, h);
	}
	headerDone = true;
	headerOfx = ofx, headerOfy = ofy, headerH = h, headerY = yScale;
	return true;
}

void capture(struct CPUState *cpu, uint32_t record)
{
	float ofx, ofy, h, yScale;
	if (!sceneFrame(cpu, ofx, ofy, h, yScale))
		return;
	uint32_t p = rw(record + 12);
	if ((p & 0xFF000000) != 0x80000000)
		return;
	// the part's vertices where part_draw left them: screen, depth (SZ / 32) -> view space
	struct Vertex
	{
		float sx, sy, x, y, z, nx = 0, ny = 0, nz = 0;
		bool used = false;
	} v[128];
	auto vertex = [&](uint32_t i) -> Vertex& {
		Vertex& out = v[i & 127];
		if (!out.used)
		{
			const uint32_t sxy = rw(0x1F800000 + 4 * i);
			out.sx = (int16_t)(sxy & 0xFFFF);
			out.sy = (int16_t)(sxy >> 16);
			out.z = psx_mod_read_half(0x1F8001B8 + 2 * i) * 32.0f;
			out.x = (out.sx - ofx) * out.z / h;
			out.y = (out.sy - ofy) * out.z / h / yScale;
			out.used = true;
		}
		return out;
	};
	uint32_t tris[1024][3];
	unsigned count = 0;
	for (int kind = 0; kind < 4; kind++)
	{
		const uint32_t n = rw(p);
		p += 4;
		if (n > 512)
			return;
		const bool quad = kind == 1 || kind == 3;
		for (uint32_t k = 0; k < n && count + 2 <= 1024; k++)
		{
			const uint32_t w0 = rw(p);
			p += kind == 3 ? 12 : 8;
			const uint32_t i0 = (w0 >> 2) & 0x7F, i1 = (w0 >> 9) & 0x7F, i2 = (w0 >> 16) & 0x7F, i3 = w0 >> 25;
			tris[count][0] = i0, tris[count][1] = i1, tris[count][2] = i2, count++;
			if (quad)
				tris[count][0] = i1, tris[count][1] = i3, tris[count][2] = i2, count++;
		}
	}
	// normals: each face's outward one (the game's front faces wind so -(P1-P0)x(P2-P0) faces out), summed at
	// its corners
	for (unsigned t = 0; t < count; t++)
	{
		Vertex &a = vertex(tris[t][0]), &b = vertex(tris[t][1]), &c = vertex(tris[t][2]);
		if (a.z <= 0 || b.z <= 0 || c.z <= 0)
			continue;
		const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z, wx = c.x - a.x, wy = c.y - a.y, wz = c.z - a.z;
		const float nx = -(uy * wz - uz * wy), ny = -(uz * wx - ux * wz), nz = -(ux * wy - uy * wx);
		for (Vertex *q : { &a, &b, &c })
			q->nx += nx, q->ny += ny, q->nz += nz;
	}
	for (unsigned t = 0; t < count; t++)
	{
		const Vertex *q[3] = { &vertex(tris[t][0]), &vertex(tris[t][1]), &vertex(tris[t][2]) };
		if (q[0]->z <= 0 || q[1]->z <= 0 || q[2]->z <= 0)
			continue;
		for (const Vertex *c : q)
		{
			const float n = std::sqrt(c->nx * c->nx + c->ny * c->ny + c->nz * c->nz);
			const float k = n > 0 ? 1 / n : 0;
			building.corners.push_back({ c->sx, c->sy, c->x, c->y, c->z, c->nx * k, c->ny * k, c->nz * k });
		}
	}
}

// ---- The fighters natively: part_draw (0x80036CAC) as host code ----
//
// No GTE and no PlayStation packets. Each part: the game's matrices for it (fighter_draw_parts sets the GTE's rotation,
// translation and light matrix; relight above adjusts the lights), its 3DMK record read as data, its vertices and normals
// computed here (the vertex program: imports, own vertices, seam blends, exports,
// in view space where the game works in screen space, so a
// seam's average is the true one), and its polygons kept as the frame's triangles: screen corners with sub-pixel
// precision, their true depth, colours lit here every frame (the game relights on alternate frames). The packets the game
// prebuilt from the texture block (0x80035190) give the UVs, CLUT and page; nothing is written to them, nothing goes in
// the ordering table. The backdrop and the floor go to the GPU as the walk of its ordering table starts, at its top entry
// (patch 0024), before everything: the HUD's bars lie at far entries (0xFC4 down), the game's backdrop and floor farther still.
// The fighters' at their own entries. Everything nearer is depth tested against them, or (no depth: the HUD) drawn over them.
// (An entry reached, not its own: the game's frame end, 0x80029A28, unlinks runs of empty entries, and it reorders the table's halves, so the walk neither
// visits every entry nor only descends: walkRank. Every entry with packets is visited, and every walk starts at the top entry, which is never unlinked.)
// DEADSTATION_SDK_OFF=fighters: the game's own part_draw.
constexpr uint32_t OtTop = 0xFFC;			// the top entry, where a walk starts (GsOT's tag: 1024 entries)
constexpr uint32_t PartColour = 0x800ADE1C;	// the GTE colour (RGBC) the game lights a part's normals with
bool nativeFighters = false;

struct NativeV3
{
	float x, y, z;
};
struct NativeColour
{
	uint32_t rgb;
	float nx, ny, nz;	// view space, unit (the 3D layer's normals)
};
struct NativeCornerOut
{
	float sx, sy, z;
	uint32_t rgb;
	uint16_t uv;
};
struct NativeTri
{
	NativeCornerOut c[3];
	uint16_t page, clut;
	uint8_t kind = 0;		// 0 textured, 1 flat (no texture), bit 1: semi-transparent (its mode from the page)
	uint16_t entry = 0;		// a fighter's polygon: its entry in the ordering table, by the game's own formula
};
// (the backdrop's: painted first with no depth at all, as the game's own table for it is, linked at 0xFC8: its lower walls
// lie in front of the floor and the fighters in space, and everything else paints over them)
struct FighterList
{
	std::vector<NativeTri> tris;
	std::vector<NativeTri> backdrop;	// no depth, over the floor's far edge
	std::vector<NativeTri> sky;			// the amusement park's: under the backdrop, so the fireworks can go between
	std::vector<DeadstationGlow> fireworks;
	std::vector<NativeTri> stage;		// the floor, with depth
	std::vector<DeadstationGlow> mist;	// a stage's ground mist and what floats behind the fighters: over the floor
	std::vector<DeadstationGlow> front;	// what floats nearer than them: over them
	bool frontDone = false;				// in this walk of its table
	size_t fightersDone = 0;			// in this walk of the table: of tris (sorted by their entries, farthest first)
	bool sorted = false;
	uint32_t ot = 0, serial = 0;
	bool drawn = false;		// in this walk of its table
};
FighterList fighterLists[2];	// by packet set: one being built, the other being drawn
uint32_t frameSerial = 0;		// the game's frame ends so far
uint32_t nativeSet = ~0u;
// the vertex and colour programs' buffers: the scratch one (the part's list: imports then its own), and the global one
// the fighter's parts leave seam vertices in. Slots are the game's bytes: slot / 2 - 1.
NativeV3 vScratch[256], vGlobal[256];
NativeColour cScratch[256], cGlobal[256];

inline uint32_t ramWord(uint32_t a) { return *(const uint32_t *)(g_psx_ram + (a & 0x1FFFFC)); }
inline int16_t ramHalf(uint32_t a) { return *(const int16_t *)(g_psx_ram + (a & 0x1FFFFE)); }
inline uint32_t slotIndex(uint32_t slot) { return ((slot >> 1) - 1) & 255; }
inline NativeV3 half(NativeV3 v) { return { v.x * 0.5f, v.y * 0.5f, v.z * 0.5f }; }
inline NativeV3 add(NativeV3 a, NativeV3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

// a list of 9-bit entries (slot byte, halve flag) packed in words; a word is used until its remaining bits are zero
template <typename F>
uint32_t nineBits(uint32_t p, F each)
{
	const uint32_t n = ramWord(p);
	p += 4;
	uint32_t word = 0;
	for (uint32_t i = 0; i < n; i++)
	{
		if (word == 0)
		{
			word = ramWord(p);
			p += 4;
		}
		each(word & 0xFF, (word & 0x100) != 0);
		word >>= 9;
	}
	return p;
}
// the two import lists at p: 8-bit slots, pass 0 from the scratch buffer, pass 1 from the global
template <typename F>
uint32_t imports(uint32_t p, F each)
{
	for (int pass = 0; pass < 2; pass++)
	{
		const uint32_t n = ramWord(p);
		p += 4;
		uint32_t word = 0;
		for (uint32_t i = 0; i < n; i++)
		{
			if (word == 0)
			{
				word = ramWord(p);
				p += 4;
			}
			each(pass, word & 0xFF);
			word >>= 8;
		}
	}
	return p;
}

// this game frame's list, begun by the first of its parts or stage pieces
FighterList& nativeList(uint32_t ot)
{
	const uint32_t set = ramWord(PacketSet) & 1;
	FighterList& list = fighterLists[set];
	if (set != nativeSet || list.serial != frameSerial)
	{
		nativeSet = set;
		list.tris.clear();
		list.stage.clear();
		list.backdrop.clear();
		list.sky.clear();
		list.fireworks.clear();
		list.mist.clear();
		list.front.clear();
		list.sorted = false;
		list.ot = ot;
		list.serial = frameSerial;
		list.drawn = false;
	}
	return list;
}

void nativePart(struct CPUState *cpu, uint32_t record, uint32_t packets, uint32_t ot)
{
	const uint32_t vertexBlock = ramWord(record), normalBlock = ramWord(record + 8), polygons = ramWord(record + 12);
	if ((vertexBlock & 0xFF000000) != 0x80000000 || (normalBlock & 0xFF000000) != 0x80000000 || (polygons & 0xFF000000) != 0x80000000)
		return;
	FighterList& list = nativeList(ot);
	const uint32_t *g = cpu->gte_ctrl;
	const float k = 1.0f / 4096.0f;
	const float r11 = lo16(g[0]) * k, r12 = hi16(g[0]) * k, r13 = lo16(g[1]) * k, r21 = hi16(g[1]) * k, r22 = lo16(g[2]) * k,
			r23 = hi16(g[2]) * k, r31 = lo16(g[3]) * k, r32 = hi16(g[3]) * k, r33 = lo16(g[4]) * k;
	const float tx = (float)(int32_t)g[5], ty = (float)(int32_t)g[6], tz = (float)(int32_t)g[7];
	const float ofx = (int32_t)g[24] / 65536.0f, ofy = (int32_t)g[25] / 65536.0f, h = (float)(uint16_t)g[26];
	if (h <= 0)
		return;

	// the vertices
	uint32_t p = vertexBlock;
	const uint32_t ownStart = slotIndex(ramWord(p));
	p += 4;
	uint32_t count = 0;
	p = imports(p, [&](int pass, uint32_t slot) { vScratch[count++ & 255] = pass == 0 ? vScratch[slotIndex(slot)] : vGlobal[slotIndex(slot)]; });
	const uint32_t own = ramWord(p);
	p += 4;
	for (uint32_t i = 0; i < own; i++, p += 8)
	{
		const float x = ramHalf(p), y = ramHalf(p + 2), z = ramHalf(p + 4);
		vScratch[(count + i) & 255] = { r11 * x + r12 * y + r13 * z + tx, r21 * x + r22 * y + r23 * z + ty, r31 * x + r32 * y + r33 * z + tz };
	}
	const uint32_t end = count + own;
	uint32_t cur = ownStart;
	// seam blends: summed with (or averaged into) the global copy, stored to both
	p = nineBits(p, [&](uint32_t slot, bool halve) {
		NativeV3& v = vScratch[cur++ & 255];
		v = add(halve ? half(v) : v, vGlobal[slotIndex(slot)]);
		vGlobal[slotIndex(slot)] = v;
	});
	// adds: of a global (pass 0) or scratch (pass 1) copy
	for (int pass = 0; pass < 2; pass++)
		p = nineBits(p, [&](uint32_t slot, bool halve) {
			NativeV3& v = vScratch[cur++ & 255];
			v = add(pass == 0 ? vGlobal[slotIndex(slot)] : vScratch[slotIndex(slot)], halve ? half(v) : v);
		});
	// exports: to the global (pass 0) or scratch (pass 1) buffer, halved first when flagged
	for (int pass = 0; pass < 2; pass++)
		p = nineBits(p, [&](uint32_t slot, bool halve) {
			NativeV3& v = vScratch[cur++ & 255];
			if (halve)
				v = half(v);
			(pass == 0 ? vGlobal : vScratch)[slotIndex(slot)] = v;
		});
	// the last halving: a 2-bit code a vertex, halved unless the code is 1
	{
		const uint32_t n = ramWord(p);
		p += 4;
		uint32_t word = 0;
		for (uint32_t i = 0; i < n; i++)
		{
			if (word == 0)
			{
				word = ramWord(p);
				p += 4;
			}
			if ((word & 3) != 1)
				vScratch[cur & 255] = half(vScratch[cur & 255]);
			cur++;
			word >>= 2;
		}
	}
	(void)end;

	// the colours: the GTE's NCCT as integers (exactly its results), with this part's light matrices
	const int32_t l[9] = { lo16(g[8]), hi16(g[8]), lo16(g[9]), hi16(g[9]), lo16(g[10]), hi16(g[10]), lo16(g[11]), hi16(g[11]), lo16(g[12]) };
	const int32_t lc[9] = { lo16(g[16]), hi16(g[16]), lo16(g[17]), hi16(g[17]), lo16(g[18]), hi16(g[18]), lo16(g[19]), hi16(g[19]), lo16(g[20]) };
	const int64_t bk[3] = { (int64_t)(int32_t)g[13] << 12, (int64_t)(int32_t)g[14] << 12, (int64_t)(int32_t)g[15] << 12 };
	const uint32_t material = ramWord(PartColour);
	const int32_t matR = material & 0xFF, matG = (material >> 8) & 0xFF, matB = (material >> 16) & 0xFF;
	// the 3D layer's frame, when it is wanted
	float layerOfx = 0, layerOfy = 0, layerH = 0, layerY = 1;
	const bool layer = sceneCapture && sceneFrame(cpu, layerOfx, layerOfy, layerH, layerY);
	const float inverseY = layer && layerY > 0 ? 1 / layerY : 1, ky = inverseY;
	p = normalBlock;
	const uint32_t colourStart = ((ramWord(p) >> 2) - 1) & 255;
	p += 4;
	uint32_t colours = 0;
	p = imports(p, [&](int pass, uint32_t slot) { cScratch[colours++ & 255] = pass == 0 ? cScratch[slotIndex(slot)] : cGlobal[slotIndex(slot)]; });
	const uint32_t ownNormals = ramWord(p);
	p += 4;
	auto clampIr = [](int64_t v) { return (int32_t)std::clamp<int64_t>(v, 0, 0x7FFF); };
	for (uint32_t i = 0; i < ownNormals; i++, p += 8)
	{
		const int32_t nx = ramHalf(p), ny = ramHalf(p + 2), nz = ramHalf(p + 4);
		const int32_t a = clampIr((l[0] * nx + l[1] * ny + l[2] * nz) >> 12), b = clampIr((l[3] * nx + l[4] * ny + l[5] * nz) >> 12),
				c = clampIr((l[6] * nx + l[7] * ny + l[8] * nz) >> 12);
		const int32_t ir = clampIr((bk[0] + lc[0] * a + lc[1] * b + lc[2] * c) >> 12), ig = clampIr((bk[1] + lc[3] * a + lc[4] * b + lc[5] * c) >> 12),
				ib = clampIr((bk[2] + lc[6] * a + lc[7] * b + lc[8] * c) >> 12);
		const uint32_t out = (uint32_t)std::min(255, (matR * ir) >> 12) | (uint32_t)std::min(255, (matG * ig) >> 12) << 8 |
				(uint32_t)std::min(255, (matB * ib) >> 12) << 16;
		NativeColour& to = cScratch[(colours + i) & 255];
		to.rgb = out;
		if (sceneCapture)
		{
			const float fx = (float)nx, fy = (float)ny, fz = (float)nz;
			const float vx = r11 * fx + r12 * fy + r13 * fz, vy = (r21 * fx + r22 * fy + r23 * fz) * ky, vz = r31 * fx + r32 * fy + r33 * fz;
			const float len = std::sqrt(vx * vx + vy * vy + vz * vz);
			const float m = len > 0 ? 1 / len : 0;
			to.nx = vx * m, to.ny = vy * m, to.nz = vz * m;
		}
	}
	uint32_t curColour = colourStart;
	for (int pass = 0; pass < 2; pass++)
	{
		const uint32_t n = ramWord(p);
		p += 4;
		uint32_t word = 0;
		for (uint32_t i = 0; i < n; i++)
		{
			if (word == 0)
			{
				word = ramWord(p);
				p += 4;
			}
			(pass == 0 ? cGlobal : cScratch)[slotIndex(word & 0xFF)] = cScratch[curColour++ & 255];
			word >>= 8;
		}
	}

	// the polygons: four kinds, each a count and its entries, with the prebuilt packets' UVs, CLUT and page beside them
	p = polygons;
	uint32_t packet = packets;
	const int PacketWords[4] = { 8, 10, 10, 13 };
	for (int kind = 0; kind < 4; kind++)
	{
		const uint32_t n = ramWord(p);
		p += 4;
		const bool quad = kind == 1 || kind == 3, smooth = kind >= 2;
		// the words of a packet with each corner's UV (the first with the CLUT, the second with the page)
		const int uvWord[4] = { 3, smooth ? 6 : 5, smooth ? 9 : 7, smooth ? 12 : 9 };
		for (uint32_t i = 0; i < n; i++, packet += (uint32_t)PacketWords[kind] * 4)
		{
			const uint32_t w0 = ramWord(p), w1 = ramWord(p + 4), w2 = kind == 3 ? ramWord(p + 8) : 0;
			p += kind == 3 ? 12 : 8;
			const uint32_t iv[4] = { (w0 >> 2) & 0x7F, (w0 >> 9) & 0x7F, (w0 >> 16) & 0x7F, w0 >> 25 };
			const uint32_t flags = quad ? w1 : w0;
			uint32_t ic[4];
			if (kind == 0 || kind == 1)
				ic[0] = ic[1] = ic[2] = ic[3] = ((w1 & 0x1FC) >> 2) - 1;
			else if (kind == 2)
				ic[0] = ((w1 & 0x1FC) >> 2) - 1, ic[1] = (((w1 >> 7) & 0x1FC) >> 2) - 1, ic[2] = ((w1 >> 16) >> 2) - 1, ic[3] = 0;
			else
				ic[0] = ((w1 & 0x1FC) >> 2) - 1, ic[1] = ((w2 & 0x1FC) >> 2) - 1, ic[2] = (((w2 >> 7) & 0x1FC) >> 2) - 1, ic[3] = ((w2 >> 16) >> 2) - 1;
			// projected as the GTE does, with sub-pixel precision; behind the camera: not drawn
			NativeCornerOut c[4];
			const int corners = quad ? 4 : 3;
			bool behind = false;
			for (int j = 0; j < corners; j++)
			{
				const NativeV3& v = vScratch[iv[j]];
				if (v.z < 1.0f)
				{
					behind = true;
					break;
				}
				const float q = h / v.z;
				c[j].sx = ofx + v.x * q, c[j].sy = ofy + v.y * q, c[j].z = v.z;
				c[j].rgb = cScratch[ic[j] & 255].rgb;
				c[j].uv = (uint16_t)ramWord(packet + (uint32_t)uvWord[j] * 4);
			}
			if (behind)
				continue;
			// the game's facing test (NCLIP over corners 0, 1 and the last), unless double-sided; and its ordering table entry:
			// its first and last corners' depths (SZ / 32), less a bias from the polygon's data, whose sign a double-sided
			// polygon's facing turns
			const NativeCornerOut& fa = c[0], &fb = c[1], &fd = c[corners - 1];
			const bool faces = (fb.sx - fa.sx) * (fd.sy - fa.sy) - (fd.sx - fa.sx) * (fb.sy - fa.sy) > 0;
			if (!(flags & 0x1000000) && !faces)
				continue;
			const int32_t bias = (flags & 0x1000000) != 0 && !faces ? -((int32_t)flags >> 25) : (int32_t)flags >> 25;
			const uint16_t entry = (uint16_t)((((int32_t)((uint32_t)fa.z >> 5) + (int32_t)((uint32_t)(quad ? c[3].z : c[1].z) >> 5) - bias) * 2) & 0xFFC);
			const uint16_t clut = (uint16_t)(ramWord(packet + 3 * 4) >> 16), page = (uint16_t)(ramWord(packet + (uint32_t)uvWord[1] * 4) >> 16) & 0x1FF;
			auto triangle = [&](int i0, int i1, int i2) {
				list.tris.push_back({ { c[i0], c[i1], c[i2] }, page, clut, 0, entry });
				if (layer)
					for (int j : { i0, i1, i2 })
					{
						const NativeV3& v = vScratch[iv[j]];
						const NativeColour& nc = cScratch[ic[j] & 255];
						building.corners.push_back({ c[j].sx, c[j].sy, v.x, v.y * inverseY, v.z, nc.nx, nc.ny, nc.nz });
					}
			};
			triangle(0, 1, 2);
			if (quad)
				triangle(1, 3, 2);
		}
	}
}

// triangles to the GPU (the module's thread, in the list's walk)
void emitTriangles(const NativeTri *tris, size_t count, bool depth)
{
	if (count == 0)
		return;
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	int ax0 = 0, ay0 = 0, ax1 = 1024, ay1 = 512;
	gr_get_draw_area(&ax0, &ay0, &ax1, &ay1);
	const float fx = (float)ox, fy = (float)oy;
	// textured with depths (the fighters, the floor): in one block
	if (depth)
	{
		static std::vector<DeadstationTriangle> out;
		out.clear();
		out.reserve(count);
		for (size_t i = 0; i < count; i++)
		{
			const NativeTri& t = tris[i];
			const float x0 = t.c[0].sx + fx, y0 = t.c[0].sy + fy, x1 = t.c[1].sx + fx, y1 = t.c[1].sy + fy, x2 = t.c[2].sx + fx, y2 = t.c[2].sy + fy;
			if (std::max({ x0, x1, x2 }) < (float)ax0 || std::min({ x0, x1, x2 }) > (float)ax1 || std::max({ y0, y1, y2 }) < (float)ay0 ||
					std::min({ y0, y1, y2 }) > (float)ay1 || (t.kind & 1) != 0)
				continue;
			DeadstationTriangle o{};
			o.x[0] = x0, o.x[1] = x1, o.x[2] = x2, o.y[0] = y0, o.y[1] = y1, o.y[2] = y2;
			for (int j = 0; j < 3; j++)
				o.z[j] = std::min(t.c[j].z, 65535.f), o.rgb[j] = t.c[j].rgb, o.uv[j] = t.c[j].uv;
			o.page = t.page & 0x1FF, o.clut = t.clut, o.raw = 0;
			o.semi = (t.kind & 2) != 0 ? (uint8_t)(1 + ((t.page >> 5) & 3)) : 0;
			out.push_back(o);
		}
		deadstation_triangles(out.data(), out.size());
		return;
	}
	int semi = -2;
	for (size_t i = 0; i < count; i++)
	{
		const NativeTri& t = tris[i];
		const NativeCornerOut& a = t.c[0], &b = t.c[1], &c = t.c[2];
		const float x0 = a.sx + fx, y0 = a.sy + fy, x1 = b.sx + fx, y1 = b.sy + fy, x2 = c.sx + fx, y2 = c.sy + fy;
		if (std::max({ x0, x1, x2 }) < (float)ax0 || std::min({ x0, x1, x2 }) > (float)ax1 || std::max({ y0, y1, y2 }) < (float)ay0 ||
				std::min({ y0, y1, y2 }) > (float)ay1)
			continue;
		const int wantSemi = (t.kind & 2) != 0 ? (t.page >> 5) & 3 : -1;
		if (wantSemi != semi)
		{
			gr_set_semi_transparency(wantSemi >= 0, wantSemi >= 0 ? wantSemi : 0);
			semi = wantSemi;
		}
		if (depth)
			gr_set_triangle_depth(1, std::min(a.z, 65535.f), std::min(b.z, 65535.f), std::min(c.z, 65535.f));
		else
			gr_set_triangle_depth(0, 0.f, 0.f, 0.f);
		gr_set_precise_triangle(1, (int32_t)(x0 * 65536.f), (int32_t)(y0 * 65536.f), (int32_t)(x1 * 65536.f), (int32_t)(y1 * 65536.f),
				(int32_t)(x2 * 65536.f), (int32_t)(y2 * 65536.f));
		if ((t.kind & 1) != 0)
		{
			const uint16_t colour = (uint16_t)((a.rgb >> 3 & 31) | (a.rgb >> 11 & 31) << 5 | (a.rgb >> 19 & 31) << 10);
			gr_draw_flat_triangle((int)std::lround(x0), (int)std::lround(y0), (int)std::lround(x1), (int)std::lround(y1), (int)std::lround(x2),
					(int)std::lround(y2), colour);
		}
		else
			gr_draw_shaded_textured_triangle((int)std::lround(x0), (int)std::lround(y0), a.uv & 0xFF, a.uv >> 8, a.rgb, (int)std::lround(x1),
					(int)std::lround(y1), b.uv & 0xFF, b.uv >> 8, b.rgb, (int)std::lround(x2), (int)std::lround(y2), c.uv & 0xFF, c.uv >> 8, c.rgb,
					(uint16_t)((t.clut & 0x3F) * 16), (uint16_t)((t.clut >> 6) & 0x1FF), t.page & 0x1FF, 0);
	}
}

// a stage's added light to the GPU, at the draw offset as the rest of the frame
void emitGlows(const std::vector<DeadstationGlow>& glows)
{
	if (glows.empty())
		return;
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	static std::vector<DeadstationGlow> placed;
	placed = glows;
	for (DeadstationGlow& g : placed)
		for (int j = 0; j < 3; j++)
			g.x[j] += (float)ox, g.y[j] += (float)oy;
	deadstation_glows(placed.data(), placed.size());
}

// the frame's stage and fighters to the GPU, at their places in the game's order
FighterList *heldBack = nullptr;	// the stage waiting for the walk to reach this node, past the game's own clear
uint32_t heldBackUntil = 0;

void emitStage(FighterList& list)
{
	list.drawn = true;
	// in the game's order (0x8006D7CC): the floor (its table's entry 0xFF4), then the backdrop over its far edge (0xFC8), which is what makes
	// the horizon. The rooftop above the clouds alone (0x800ADC84 == 11) links its backdrop at the floor's entry, ahead of it: there the floor
	// lies over the backdrop's clouds, right up to the ropes.
	if (psx_mod_read_half(0x800ADC84) == 11)
	{
		emitTriangles(list.backdrop.data(), list.backdrop.size(), false);
		emitTriangles(list.stage.data(), list.stage.size(), true);
	}
	else
	{
		emitTriangles(list.stage.data(), list.stage.size(), true);
		emitTriangles(list.sky.data(), list.sky.size(), false);
		emitGlows(list.fireworks);
		emitTriangles(list.backdrop.data(), list.backdrop.size(), false);
	}
	emitGlows(list.mist);
}

// The order the walk comes to the table's entries. The game's frame end (0x80029A28, below) swaps three links, outside two modes of its own: from the top down to 0xDA4, then the nearer half from
// 0x7FC down, then the middle from 0xDA0 down to 0x800, and the last entry. So the middle goes over the nearer half: that's where what's announced lies, YOU LOSE, CONTINUE, the practice menus, over
// the fighters, who are in the nearer half. Going by the entry's number alone put the fighters off to the last entry whenever the nearer half had no packets of its own, over all of that.
inline uint32_t walkRank(uint32_t entry)
{
	if (rw(GameMode) - 7u <= 1u)
		return 0xFFCu - entry;
	if (entry > 0xDA0u)
		return 0xFFCu - entry;
	if (entry >= 0x800u)
		return 0x2000u + (0xDA0u - entry);
	return entry >= 4u ? 0x1000u + (0x7FCu - entry) : 0x3000u;
}

// if the walk's first packet from this node is one painting the whole screen flat, the node after it; else 0
uint32_t pastClear(uint32_t node)
{
	for (int i = 0; i <= (int)(OtTop / 4); i++)
	{
		const uint32_t word = ramWord(node), next = word & 0xFFFFFF;
		if (word >> 24 != 0)
		{
			const uint32_t command = ramWord(node + 4) >> 24, size = ramWord(node + 12);
			const bool whole = word >> 24 == 3 && (command == 0x60 || command == 0x02) && (size & 0xFFFF) >= 320 && (size >> 16) >= 240;
			return whole && next != 0xFFFFFF ? next & 0x1FFFFC : 0;
		}
		if (next == 0xFFFFFF)
			return 0;
		node = next & 0x1FFFFC;
	}
	return 0;
}

// DEADSTATION_TEKKEN3_WALK (tests): each second, the frame's walk of its table: the entries with packets, in the order reached, and where the fighters' polygons lie
bool walkLog = false;
struct WalkRun { uint32_t entry, packets, command; };
std::vector<WalkRun> walkNow;
uint32_t walkAtSerial = 0;

void onListNode(uint32_t addr)
{
	const uint32_t at = addr & 0x1FFFFC;
	if (walkLog)
		for (const FighterList& list : fighterLists)
		{
			const uint32_t entry = at - (list.ot & 0x1FFFFC);
			if (list.ot == 0 || entry > OtTop)
				continue;
			if (entry == OtTop && walkAtSerial != frameSerial)
			{
				if (frameSerial % 60 < 2 && !walkNow.empty())
				{
					uint32_t lo = 0xFFFF, hi = 0;
					for (const NativeTri& t : list.tris)
						lo = std::min<uint32_t>(lo, t.entry), hi = std::max<uint32_t>(hi, t.entry);
					std::string line;
					char one[48];
					for (const WalkRun& r : walkNow)
					{
						std::snprintf(one, sizeof(one), " %03X:%u:%02X", r.entry, r.packets, r.command);
						line += one;
					}
					std::fprintf(stderr, "deadstation walk: mode %u, %u fighter polygons at %03X..%03X;%s\n", (unsigned)rw(GameMode), (unsigned)list.tris.size(), lo, hi, line.c_str());
				}
				walkNow.clear();
				walkAtSerial = frameSerial;
			}
			const uint32_t word = ramWord(at);
			if (word >> 24 == 0)
			{
				// (an entry's own tag holds no packets: what's after it in the chain, up to the next entry, comes as nodes of their own)
				walkNow.push_back({ entry, 0, 0 });
			}
			break;
		}
	if (heldBack != nullptr && at == heldBackUntil)
	{
		emitStage(*heldBack);
		heldBack = nullptr;
	}
	if (walkLog)
	{
		const uint32_t word = ramWord(at);
		bool inTable = false;
		for (const FighterList& list : fighterLists)
			inTable = inTable || (list.ot != 0 && at - (list.ot & 0x1FFFFC) <= OtTop);
		if (!inTable && word >> 24 != 0 && !walkNow.empty())
			walkNow.back().packets++, walkNow.back().command = ramWord(at + 4) >> 24;
		else if (inTable && word >> 24 != 0 && !walkNow.empty())
			walkNow.back().packets++, walkNow.back().command = ramWord(at + 4) >> 24;
	}
	for (FighterList& list : fighterLists)
	{
		const uint32_t entry = at - (list.ot & 0x1FFFFC);
		if (list.ot == 0 || entry > OtTop)
			continue;
		// (a frame built more than two frame ends ago: this table is now some other screen's)
		const bool current = list.serial + 2 >= frameSerial && (!list.tris.empty() || !list.stage.empty() || !list.backdrop.empty());
		if (entry == OtTop)
		{
			heldBack = nullptr;
			list.drawn = false;
			list.fightersDone = 0;
			deadstation_scene_perspective(current);
			if (current && !list.sorted)
			{
				std::stable_sort(list.tris.begin(), list.tris.end(), [](const NativeTri& l, const NativeTri& r) { return walkRank(l.entry) < walkRank(r.entry); });
				list.sorted = true;
			}
			list.frontDone = false;
			if (current)
			{
				// a stage whose sky is a clear of its own (the rooftop's blue, the first packet after this entry): after that
				heldBackUntil = pastClear(at);
				heldBack = heldBackUntil != 0 ? &list : nullptr;
				if (heldBack == nullptr)
					emitStage(list);
			}
		}
		if (!current)
			return;
		// each fighter polygon at its entry: as the walk reaches an entry, those at it and any it would have reached by now (the game's frame
		// end unlinks empty entries, so a polygon's own may not be reached), so everything without depth lies as the game ordered it
		if (list.drawn)
		{
			const uint32_t rank = walkRank(entry);
			size_t end = list.fightersDone;
			while (end < list.tris.size() && walkRank(list.tris[end].entry) <= rank)
				end++;
			emitTriangles(list.tris.data() + list.fightersDone, end - list.fightersDone, true);
			list.fightersDone = end;
			// the last of them drawn: what floats in front of them
			if (end == list.tris.size() && !list.frontDone)
			{
				list.frontDone = true;
				emitGlows(list.front);
			}
		}
		return;
	}
}

// DEADSTATION_TEKKEN3_STAGE (tests): every fighter's home stage (the name table, 0x80097D40: pointers to entries, +10 the stage)
// set to that one, so a run fights there, whoever the opponent
int forceStage = -1;
void forceStageNow()
{
	for (uint32_t i = 0; i < 21 * 4; i++)
	{
		const uint32_t entry = rw(0x80097D40 + i * 4);
		if ((entry & 0xFF000000) == 0x80000000)
			psx_mod_write_byte(entry + 10, (uint8_t)forceStage);
	}
}

// Unlock Everything (Options, at once): what the game's own list of unlocks does at its last step (0x80056310, the
// list at 0x80098214): every fighter, every extra costume, and the flags behind the extra modes, in its record of the player's
// progress (0x80097EF0, which a memory card overwrites as it loads: so every frame).
bool unlockAll = false;
void applyUnlocks()
{
	constexpr uint32_t Progress = 0x80097EF0;
	if (!unlockAll)
		return;
	psx_mod_write_word(Progress, rw(Progress) | 0x1FFFFF);			// every fighter
	psx_mod_write_word(Progress + 4, rw(Progress + 4) | 0x50382);	// every extra costume
	psx_mod_write_byte(Progress + 54, 1);
	psx_mod_write_byte(Progress + 55, 1);
	psx_mod_write_byte(Progress + 47, 2);
	psx_mod_write_byte(Progress + 42, 14);							// the list's last step: nothing left to announce
}

// the game's frame end (the wait at 0x800296C4, entered once a frame): a frame of its logic done
int onFrameEnd(CPUState *, uint32_t)
{
	frameSerial++;
	// A frame with no fighters in it: no layer of them either. The presenter draws the latest it was given on whatever's showing, so what's published must end when they do,
	// or the last fight's bodies, shadows and lights stay on the title, the scores and the movies.
	if (modelsThisFrame)
		layerCleared = false;
	else if (!layerCleared)
	{
		layerCleared = true;
		building = deadstation::scene::Frame {};
		sceneHeaderDone = false;
		deadstation::scene::Frame none;
		deadstation::scene::publish(none);
	}
	modelsThisFrame = false;
	flavourClock++;
	patchMoods();
	applyUnlocks();
	if (forceStage >= 0 && frameSerial % 60 == 1)
		forceStageNow();
	deadstation_game_frames++;
	return 0;
}

// part_draw: the part's lights changed (relight), then, when the 3D layer is wanted, the game's part_draw run
// here and its polygons read
int onPartDraw(struct CPUState *cpu, uint32_t)
{
	if (inPartDraw)
		return 0;
	const uint32_t ra = cpu->gpr[31];
	if (ra != PartDrawReturn && ra != PartDrawReturn2)
		return 0;
	relight(cpu);
	if (nativeFighters)
	{
		nativePart(cpu, cpu->gpr[4], cpu->gpr[5], cpu->gpr[6]);
		return 1;
	}
	if (!sceneCapture)
		return 0;
	const uint32_t record = cpu->gpr[4];
	inPartDraw = true;
	psx_dispatch_call(cpu, PartDraw, ra);
	inPartDraw = false;
	capture(cpu, record);
	return 1;
}

void onVBlank()
{
	static unsigned frames = 0;
	static uint16_t shown[9];
	uint16_t now[9];
	for (int i = 0; i < 9; i++)
		now[i] = psx_mod_read_half(P + i * 2);
	if (memcmp(now, shown, sizeof(now)) != 0)
	{
		memcpy(shown, now, sizeof(now));
		std::fprintf(stderr, "deadstation tekken3: P = %d %d %d / %d %d %d / %d %d %d\n", (int16_t)now[0], (int16_t)now[1],
				(int16_t)now[2], (int16_t)now[3], (int16_t)now[4], (int16_t)now[5], (int16_t)now[6], (int16_t)now[7], (int16_t)now[8]);
	}
	if (++frames % 300 == 0)
	{
		std::fprintf(stderr, "deadstation tekken3: %u shadows (%u polygons), %u parts lit in 300 vblanks\n", statShadows,
				statPolygons, statParts);
		statShadows = statPolygons = statParts = 0;
	}
}

// ---- Tekken 3's disc loader (TEKKEN3.BNS) as host code ----
// bns_start_reads issues the queued reads (bns_queue_read: sorted, at most 8) as one libds DsPacket chain that the
// simulated drive then delivers sector by sector through libds's ready callbacks, in device time. Here each entry goes
// from the disc image straight into RAM when the call is made, and the queue is left as the chain's last callback leaves
// it (done, the drive idle). Only while no XA stream plays (pausing one, FUN_8006B54C's other path, needs the drive) and
// with no entry callbacks (guest code); otherwise the game's own reads.
bool nativeDisc = false;	// no drive (below)
void bnsStartAsync(const uint32_t *entries, int n, int count);
constexpr uint32_t BnsStartReads = 0x8006C084;
// head (u32), count (s16) +4, abort (u8) +6, pending (u8) +7; the entry under way: id (u16) +8, bytes left +0xC,
// destination +0x10, callback +0x14
constexpr uint32_t BnsQueue = 0x800A0698;
// entries, 0x1C each: id (u16), contiguous (u16) +2, CdlLOC +4, destination +8, sector +0xC, size +0x10, callback +0x14,
// next +0x18
constexpr uint32_t BnsEntries = 0x800A06B8;
constexpr uint32_t CdXaStream = 0x800A0620;		// s16: < 0, none
constexpr uint32_t CdState = 0x800A05D8;		// u16: 5..7 while the XA stream is being stopped
constexpr uint32_t CdBusy = 0x800A866C;

uint32_t fromBcd(uint8_t v)
{
	return (v >> 4) * 10u + (v & 15u);
}

int onBnsStartReads(CPUState *cpu, uint32_t)
{
	(void)cpu;
	const int16_t count = (int16_t)psx_mod_read_half(BnsQueue + 4);
	if (count <= 0 || count > 8 || psx_mod_read_byte(BnsQueue + 7) != 0)
		return 0;
	if ((int16_t)psx_mod_read_half(CdXaStream) >= 0 || (uint16_t)(psx_mod_read_half(CdState) - 5u) < 3u)
	{
		if (!nativeDisc)
			return 0;
		// music playing: stopped as the game does before its reads (no drive to share, but it restarts it the same way)
		const uint32_t ra = cpu->gpr[31];
		cpu->gpr[4] = 0;
		psx_dispatch_call(cpu, 0x8006B54C, ra);
		cpu->gpr[31] = ra;
	}
	uint32_t entries[8];
	int n = 0;
	for (uint32_t e = psx_mod_read_word(BnsQueue); e != 0; e = psx_mod_read_word(e + 0x18))
	{
		if (n == count || psx_mod_read_word(e + 0x14) != 0)
			return 0;
		entries[n++] = e;
	}
	if (n != count)
		return 0;
	if (nativeDisc)
	{
		bnsStartAsync(entries, n, count);
		return 1;
	}
	uint32_t bytes = 0;
	for (int i = 0; i < n; i++)
	{
		const uint32_t e = entries[i], loc = psx_mod_read_word(e + 4);
		const uint32_t lba = (fromBcd(loc & 0xFF) * 60u + fromBcd(loc >> 8 & 0xFF)) * 75u + fromBcd(loc >> 16 & 0xFF) - 150u;
		const int32_t size = (int32_t)psx_mod_read_word(e + 0x10);
		if (size > 0 && !psx_native_cd_read(lba, psx_mod_read_word(e + 8), (uint32_t)size))
		{
			std::fprintf(stderr, "deadstation tekken3: BNS entry %u unreadable (lba %u): the game's reads\n",
					psx_mod_read_half(e), lba);
			return 0;
		}
		bytes += (uint32_t)size;
	}
	// FUN_8006B54C(0) with no stream: two stores; then the chain's end
	psx_mod_write_half(0x800A061C, 4);
	psx_mod_write_word(0x800A0634, 0xFFFFFFFFu);
	psx_mod_write_word(BnsEntries + 0x18 + (uint32_t)count * 0x1C, 0);
	const uint32_t last = entries[n - 1];
	const int32_t lastSize = (int32_t)psx_mod_read_word(last + 0x10);
	psx_mod_write_word(BnsQueue, last);
	psx_mod_write_half(BnsQueue + 8, psx_mod_read_half(last));
	psx_mod_write_word(BnsQueue + 0xC, 0);
	psx_mod_write_word(BnsQueue + 0x10, psx_mod_read_word(last + 8) + ((uint32_t)lastSize + 0x7FFu) / 0x800u * 0x800u);
	psx_mod_write_word(BnsQueue + 0x14, 0);
	psx_mod_write_half(CdState, 1);
	psx_mod_write_word(CdBusy, 0);
	static unsigned told = 0;
	if (told++ < 16)
		std::fprintf(stderr, "deadstation tekken3: BNS %d entries, %u bytes from the disc image\n", n, bytes);
	return 1;
}

// ---- the pads and the frame's wait as host code ----
//
// LIBPAD 4.2 (PADENTRY/PDMAIINI/PADPORTD) reads the pads over the serial port every vertical blank, a byte at a time,
// each byte's acknowledge an interrupt through the BIOS (with its multitap probing, hundreds of serial events a frame).
// Here none of it runs: PadInitDirect keeps the two buffers the game gives it, PadStartCom installs nothing, and each
// vertical blank the first port's buffer is written as LIBPAD leaves it for a digital pad (byte 0: 0 received, 1: 0x41
// a digital pad, 2-3: the buttons, active low, as the serial port sends them), the second port's as no pad (0xFF). The
// queries the game's reader (0x80029C10) makes answer for that pad: PadGetState stable (6), PadInfoMode's current ID 4
// (digital), no extended ID (no DualShock), no actuators. DEADSTATION_SDK_OFF=pad: the game's own LIBPAD.
constexpr uint32_t PadInitDirect = 0x800946A8, PadStartComEntry = 0x800926F8, PadStartComMain = 0x80092EB4,
		PadStopComEntry = 0x80092718, PadStopComMain = 0x80092F80, PadGetState = 0x80092784, PadInfoMode = 0x80092844,
		PadInfoAct = 0x8009293C, PadInfoComb = 0x80092A10, PadSetActAlign = 0x80092AB8, PadSetMainMode = 0x80092AF0,
		PadSetAct = 0x80092B38, PadChkVsync = 0x800926D8;
uint32_t padBuffer[2] = { 0, 0 };

void writePads()
{
	if (padBuffer[0] != 0)
	{
		const uint16_t buttons = sio_get_pad_buttons_slot(0);
		psx_mod_write_byte(padBuffer[0], 0x00);
		psx_mod_write_byte(padBuffer[0] + 1, 0x41);
		psx_mod_write_byte(padBuffer[0] + 2, (uint8_t)(buttons & 0xFF));
		psx_mod_write_byte(padBuffer[0] + 3, (uint8_t)(buttons >> 8));
	}
	if (padBuffer[1] != 0)
	{
		psx_mod_write_byte(padBuffer[1], 0xFF);
		psx_mod_write_byte(padBuffer[1] + 1, 0x00);
	}
}
int onPadInitDirect(CPUState *cpu, uint32_t)
{
	padBuffer[0] = cpu->gpr[4], padBuffer[1] = cpu->gpr[5];
	writePads();
	return 1;
}
int onPadNothing(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = 1;
	return 1;
}
int onPadZero(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = 0;
	return 1;
}
int onPadGetState(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = (cpu->gpr[4] & 0xFF) == 0x00 ? 6 : 0;	// PadStateStable, PadStateDiscon
	return 1;
}
int onPadInfoMode(CPUState *cpu, uint32_t)
{
	// InfoModeCurID (1): 4, a digital pad; the rest (extended ID, its offset, the ID table): 0
	cpu->gpr[2] = (cpu->gpr[4] & 0xFF) == 0x00 && cpu->gpr[5] == 1 ? 4 : 0;
	return 1;
}
void onPadVBlank()
{
	writePads();
}

// The fight's frame ends waiting for the vertical blank callback (0x8002979C: swap, draw the ordering table, the
// sequencer's tick), which sets 0x8009BC5C, in a loop at 0x800296C4 that stirs two random numbers (0x8004CE54 among
// them). The flag is set by root counter 2's interrupt (0x8002979C, started by the vertical blank's callback 0x800299E0
// some lines after each blank: the swap and the ordering table drawn), not by the blank itself. Entered from that loop
// with the flag still clear, guest time goes straight to the next blank or timer interrupt, whichever is first (the disc's
// and other devices' events on the way are raised as guest time passes them): the loop's next interrupt check takes it. (Going to the next blank each time skipped past the counter's interrupt: every
// game frame took two blanks, the game at half speed.) DEADSTATION_SDK_OFF=framewait: the loop runs as it is.
constexpr uint32_t FrameWait = 0x800296C4, FrameWaitEnd = 0x80029784, Stir = 0x8004CE54, FrameFlag = 0x8009BC5C;

int onStir(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (ra >= FrameWait && ra < FrameWaitEnd && psx_mod_read_word(FrameFlag) == 0)
	{
		psx_cyc_batch_flush();
		const uint32_t next = std::min(timers_cycles_to_irq(i_mask), interrupts_cycles_to_vblank() + 1u);
		psx_advance_cycles(std::max(next, 1u));
	}
	return 0;
}

// ---- No drive: LIBDS as host code ----
//
// The game's data are files and its music is audio; on the PlayStation they came through a CD drive, here they do not.
// None of LIBDS's commands reach the drive model, which stays idle: no spin-up, no directory search or table of contents
// read through it, no seeks, no sector interrupts, no waits on any of it (those waits at the simulated hardware's cost were
// most of the game's start-up, and the XA stream's events a large share of a fight's). The game's own music logic
// (0x8006AE14, 0x8006B54C, 0x8006B6FC) and movie player stay as they are, and call these:
//  - DsSearchFile: the file found in the image's ISO 9660 tree by host code; DsGetToc: the image's track table;
//  - a read or play command: the stream starts here, and each vertical blank the sectors due at the drive's speed are
//    read: XA audio sectors (of the filter's file and channel) decoded straight into the SPU's CD input, audio tracks
//    likewise; for a movie (DsRead2's streaming mode) its video sectors written straight into the game's StSetRing buffer
//    in the form LIBCD's interrupt handler leaves them, for the game's own StGetNext and StFreeRing;
//  - DsMix: the volumes those sectors are mixed at. DEADSTATION_SDK_OFF=disc: the game's own LIBDS and the drive model.
extern "C" {
int psx_native_raw_sector(uint32_t lba, uint8_t *raw);
int psx_native_track_count(void);
uint32_t psx_native_track_start(int track);
uint32_t psx_native_disc_sectors(void);
void psx_native_xa_start(void);
double spu_cd_pace(void);
void psx_native_xa_sector(const uint8_t *raw);
void psx_native_cdda_sector(const uint8_t *raw);
void psx_native_cd_mix(uint8_t l_to_l, uint8_t l_to_r, uint8_t r_to_r, uint8_t r_to_l);
int psx_cdrom_read_user_sector(uint32_t lba, uint8_t *out);
}
constexpr uint32_t CdInitVol = 0x800843B4;		// LIBCD: the SPU's CD volume and CD audio enable
constexpr uint32_t MusicPosition = 0x800A0618;	// where the game's music monitor sees the stream
constexpr uint32_t StRing = 0x800B0418, StRingSize = 0x800B0538;	// StSetRing's buffer: headers (32 bytes), then bodies (2016)

// The disc thread: nothing on the module's thread ever waits for the disc. Reading the image means decompressing it (CHD
// hunks: milliseconds each, tens of them for a file) on a slow SD card, which stalled the game's frames by up to a quarter
// of a second. Here the game's asynchronous loader stays asynchronous, as on the PlayStation: a BNS read hands its entries
// to this thread with the game's "pending" flag set, and the data go to RAM at the next vertical blank after they're
// read (or at the game's next poll of the flag). A stream's sectors are read ahead into a queue the blank takes them from.
std::mutex isoLock;			// the image's reader: one caller at a time
constexpr uint32_t BnsPending = BnsQueue + 7;
struct BnsJob
{
	uint32_t entry = 0, lba = 0, dest = 0, size = 0;
	std::vector<uint8_t> data;
};
struct Sector
{
	uint32_t lba, generation;
	uint8_t raw[2352];
};
constexpr unsigned SectorQueue = 96;
constexpr uint32_t SectorsAhead = 128;	// about 0.85 s of a stream at double speed
Sector *sectorQueue = nullptr;
std::atomic<uint32_t> queueHead { 0 }, queueTail { 0 };	// one writer each: the thread's tail, the blank's head
std::atomic<uint32_t> workerGeneration { 0 }, workerNext { 0 }, streamAt { 0 };
struct StreamCommand
{
	uint32_t generation = 0, lba = 0, end = 0xFFFFFFFFu;
	uint8_t mode = 0, file = 0, channel = 0;
	bool active = false, audioTrack = false, ring = false;
} streamCommand;
std::vector<BnsJob> bnsJobs;
int bnsCount = 0;
uint32_t bnsLastEntry = 0;
bool bnsQueued = false;				// (under discLock) for the thread
std::atomic<bool> bnsDone { false };
bool bnsWaiting = false;			// on the module's thread: a read handed over, not finished
std::mutex discLock;
std::condition_variable discWake;
std::once_flag discStarted;

void discThread()
{
	StreamCommand stream;
	uint32_t next = 0;
	uint32_t generation = 0;
	for (;;)
	{
		bool jobs = false;
		{
			std::unique_lock<std::mutex> hold(discLock);
			for (;;)
			{
				if (bnsQueued)
				{
					bnsQueued = false;
					jobs = true;
					break;
				}
				if (streamCommand.generation != generation)
				{
					stream = streamCommand;
					generation = stream.generation;
					next = stream.lba;
					workerNext = next;
					workerGeneration = generation;
				}
				if (stream.active && next < stream.end && queueTail.load() - queueHead.load() < SectorQueue && next < streamAt.load() + SectorsAhead)
					break;
				discWake.wait_for(hold, std::chrono::milliseconds(20));
			}
		}
		if (jobs)
		{
			for (BnsJob& job : bnsJobs)
			{
				job.data.assign(job.size, 0);
				uint8_t sector[2048];
				for (uint32_t done = 0, lba = job.lba; done < job.size; done += 2048, lba++)
				{
					std::lock_guard<std::mutex> hold(isoLock);
					if (!psx_cdrom_read_user_sector(lba, sector))
					{
						std::fprintf(stderr, "deadstation tekken3: sector %u unreadable\n", lba);
						break;
					}
					std::memcpy(job.data.data() + done, sector, std::min<uint32_t>(2048, job.size - done));
				}
			}
			bnsDone = true;
			continue;
		}
		Sector& out = sectorQueue[queueTail.load() % SectorQueue];
		bool ok;
		{
			std::lock_guard<std::mutex> hold(isoLock);
			ok = psx_native_raw_sector(next, out.raw);
		}
		if (!ok)
		{
			stream.active = false;
			continue;
		}
		out.lba = next, out.generation = generation;
		const uint8_t submode = out.raw[0x12];
		const bool audio = (submode & 0x04) != 0;
		const bool keep = stream.audioTrack || (audio ? (stream.mode & 0x40) != 0 && ((stream.mode & 0x08) == 0 || (out.raw[0x10] == stream.file && out.raw[0x11] == stream.channel)) : stream.ring);
		if (keep)
			queueTail.fetch_add(1);
		workerNext = ++next;
	}
}
void streamTo(uint32_t lba, uint32_t end, bool active, bool audioTrack, bool ring, uint8_t mode, uint8_t file, uint8_t channel)
{
	std::call_once(discStarted, [] {
		sectorQueue = new Sector[SectorQueue];
		std::thread(discThread).detach();
	});
	std::lock_guard<std::mutex> hold(discLock);
	streamCommand.generation++;
	streamCommand.lba = lba, streamCommand.end = end, streamCommand.active = active, streamCommand.audioTrack = audioTrack, streamCommand.ring = ring;
	streamCommand.mode = mode, streamCommand.file = file, streamCommand.channel = channel;
	streamAt = lba;
	discWake.notify_one();
}

// a BNS read: its entries handed to the disc thread, and the game told it's pending, as bns_start_reads leaves things
void bnsStartAsync(const uint32_t *entries, int n, int count)
{
	bnsJobs.clear();
	bnsJobs.resize((size_t)n);
	for (int i = 0; i < n; i++)
	{
		const uint32_t e = entries[i], loc = psx_mod_read_word(e + 4);
		const uint32_t lba = (fromBcd(loc & 0xFF) * 60u + fromBcd(loc >> 8 & 0xFF)) * 75u + fromBcd(loc >> 16 & 0xFF);
		BnsJob& job = bnsJobs[(size_t)i];
		const int32_t size = (int32_t)psx_mod_read_word(e + 0x10);
		job.entry = e, job.lba = lba >= 150 ? lba - 150 : 0, job.dest = psx_mod_read_word(e + 8), job.size = size > 0 ? (uint32_t)size : 0;
	}
	bnsCount = count;
	bnsLastEntry = entries[n - 1];
	psx_mod_write_half(0x800A061C, 4);
	psx_mod_write_word(0x800A0634, 0xFFFFFFFFu);
	psx_mod_write_word(BnsEntries + 0x18 + (uint32_t)count * 0x1C, 0);
	psx_mod_write_byte(BnsPending, 1);
	psx_mod_write_half(CdState, 8);
	psx_mod_write_word(CdBusy, 0);
	bnsWaiting = true;
	std::call_once(discStarted, [] {
		sectorQueue = new Sector[SectorQueue];
		std::thread(discThread).detach();
	});
	std::lock_guard<std::mutex> hold(discLock);
	bnsDone = false;
	bnsQueued = true;
	discWake.notify_one();
}
// the read finished: its data into RAM as a DMA would (each word through the RAM path, code pages marked, overlay capture
// told), and the queue left as the game's last callback leaves it
void bnsFinish()
{
	bnsWaiting = false;
	uint32_t bytes = 0;
	for (const BnsJob& job : bnsJobs)
	{
		if (job.size == 0)
			continue;
		const uint32_t start = job.dest & 0x1FFFFC, live = 0x200000;
		const uint32_t span = std::min(job.size, start < live ? live - start : 0u);
		if (span != 0)
			overlay_capture_before_dma(start, span);
		for (uint32_t i = 0; i + 4 <= span; i += 4)
		{
			uint32_t word;
			std::memcpy(&word, job.data.data() + i, 4);
			psx_write_word(start + i, word);
		}
		if (span != 0)
		{
			dirty_ram_mark_executable_range(start, span & ~3u);
			overlay_capture_on_dma(start, span, memory_get_ram_ptr() + start);
		}
		bytes += job.size;
	}
	const uint32_t last = bnsLastEntry;
	const int32_t lastSize = (int32_t)psx_mod_read_word(last + 0x10);
	psx_mod_write_word(BnsQueue, last);
	psx_mod_write_half(BnsQueue + 8, psx_mod_read_half(last));
	psx_mod_write_word(BnsQueue + 0xC, 0);
	psx_mod_write_word(BnsQueue + 0x10, psx_mod_read_word(last + 8) + ((uint32_t)lastSize + 0x7FFu) / 0x800u * 0x800u);
	psx_mod_write_word(BnsQueue + 0x14, 0);
	psx_mod_write_half(CdState, 1);
	psx_mod_write_word(CdBusy, 0);
	psx_mod_write_byte(BnsPending, 0);
	static unsigned told = 0;
	if (told++ < 16)
		std::fprintf(stderr, "deadstation tekken3: BNS %d entries, %u bytes read on the disc thread\n", bnsCount, bytes);
	bnsJobs.clear();
}
// bns_reads_pending, the game's poll: done at once when the data have arrived; otherwise guest time to the next blank (no
// spinning here)
constexpr uint32_t BnsReadsPending = 0x8006C23C;
int onBnsReadsPending(CPUState *cpu, uint32_t)
{
	if (bnsWaiting)
	{
		if (bnsDone.load())
			bnsFinish();
		else
		{
			psx_cyc_batch_flush();
			psx_advance_cycles(interrupts_cycles_to_vblank() + 1u);
		}
	}
	cpu->gpr[2] = psx_mod_read_byte(BnsPending);
	return 1;
}

struct HostDrive
{
	bool streaming = false, audioTrack = false, trackEnded = false, ring = false;
	uint32_t lba = 0, loc = 0, trackEnd = 0;
	uint8_t mode = 0, filterFile = 1, filterChannel = 0;
	float budget = 0;
	uint32_t ringWrite = 0;
	uint32_t generation = 0;
	uint32_t syncCallback = 0, readyCallback = 0, dataCallback = 0;
} drive;

uint32_t fromMsf(uint32_t at)
{
	const uint32_t m = psx_mod_read_byte(at), s = psx_mod_read_byte(at + 1), f = psx_mod_read_byte(at + 2);
	const uint32_t lba = (fromBcd((uint8_t)m) * 60u + fromBcd((uint8_t)s)) * 75u + fromBcd((uint8_t)f);
	return lba >= 150 ? lba - 150 : 0;
}
void toMsf(uint32_t lba, uint32_t at)
{
	const uint32_t t = lba + 150, m = t / 4500, s = t / 75 % 60, f = t % 75;
	psx_mod_write_byte(at, (uint8_t)(m / 10 * 16 + m % 10));
	psx_mod_write_byte(at + 1, (uint8_t)(s / 10 * 16 + s % 10));
	psx_mod_write_byte(at + 2, (uint8_t)(f / 10 * 16 + f % 10));
}
void driveStop()
{
	if (drive.streaming)
		streamTo(0, 0, false, false, false, 0, 0, 0);
	drive.streaming = drive.ring = drive.audioTrack = false;
}
unsigned driveXaSectors = 0, driveAudioSectors = 0, driveRingFrames = 0, driveTold = 0;
void driveRead(uint32_t lba, bool ring)
{
	if (driveTold++ < 40)
		std::fprintf(stderr, "deadstation tekken3: stream from sector %u, mode %02X, filter %u/%u%s\n", lba, drive.mode, drive.filterFile,
				drive.filterChannel, ring ? ", a movie" : "");
	drive.lba = lba, drive.streaming = true, drive.audioTrack = false, drive.ring = ring, drive.budget = 0;
	if (ring)
		drive.ringWrite = 0;
	psx_native_xa_start();
	drive.generation = streamCommand.generation + 1;
	streamTo(lba, 0xFFFFFFFFu, true, false, ring, drive.mode, drive.filterFile, drive.filterChannel);
}
void drivePlay(uint32_t lba)
{
	if (driveTold++ < 40)
		std::fprintf(stderr, "deadstation tekken3: audio track from sector %u\n", lba);
	drive.lba = lba, drive.streaming = true, drive.audioTrack = true, drive.ring = false, drive.trackEnded = false, drive.budget = 0;
	// to the next track's start
	drive.trackEnd = psx_native_disc_sectors();
	const int tracks = psx_native_track_count();
	for (int t = 1; t <= tracks; t++)
	{
		const uint32_t start = psx_native_track_start(t);
		if (start > lba && start < drive.trackEnd)
			drive.trackEnd = start;
	}
	drive.generation = streamCommand.generation + 1;
	streamTo(lba, drive.trackEnd, true, true, false, drive.mode, 0, 0);
}
void driveCommand(uint32_t command, uint32_t param, uint32_t result)
{
	switch (command)
	{
	case 0x02:	// Setloc
		if (param != 0)
			drive.loc = fromMsf(param);
		break;
	case 0x03:	// Play
		drivePlay(drive.loc);
		break;
	case 0x06:	// ReadN
	case 0x1B:	// ReadS
		driveRead(drive.loc, false);
		break;
	case 0x07:	// Standby
	case 0x08:	// Stop
	case 0x09:	// Pause
		driveStop();
		break;
	case 0x0D:	// Setfilter
		if (param != 0)
			drive.filterFile = psx_mod_read_byte(param), drive.filterChannel = psx_mod_read_byte(param + 1);
		break;
	case 0x0E:	// Setmode
		if (param != 0)
			drive.mode = psx_mod_read_byte(param);
		break;
	case 0x10:	// GetlocL: the sector's header
		if (result != 0)
			toMsf(drive.lba, result);
		break;
	case 0x11:	// GetlocP: the game's callback (0x8006BE40) only keeps the position
		psx_mod_write_word(MusicPosition, drive.lba);
		break;
	default:
		break;
	}
}
int onDsReturnOne(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = 1;
	return 1;
}
int onDsReturnZero(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = 0;
	return 1;
}
int onDsInit(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	psx_dispatch_call(cpu, CdInitVol, ra);
	cpu->gpr[31] = ra;
	psx_native_cd_mix(0x80, 0, 0x80, 0);
	cpu->gpr[2] = 1;
	return 1;
}
int onDsReset(CPUState *cpu, uint32_t)
{
	driveStop();
	cpu->gpr[2] = 1;
	return 1;
}
// DsPacket(mode, position, command, callback, count)
int onDsPacket(CPUState *cpu, uint32_t)
{
	drive.mode = (uint8_t)cpu->gpr[4];
	if (cpu->gpr[5] != 0)
		drive.loc = fromMsf(cpu->gpr[5]);
	driveCommand(cpu->gpr[6], 0, 0);
	cpu->gpr[2] = 1;
	return 1;
}
// DsCommand(command, parameters, callback, count)
int onDsCommand(CPUState *cpu, uint32_t)
{
	driveCommand(cpu->gpr[4] & 0xFF, cpu->gpr[5], 0);
	cpu->gpr[2] = 1;
	return 1;
}
// DsControl, DsControlB, DsControlF(command, parameters, result)
int onDsControl(CPUState *cpu, uint32_t)
{
	driveCommand(cpu->gpr[4] & 0xFF, cpu->gpr[5], cpu->gpr[6]);
	cpu->gpr[2] = 1;
	return 1;
}
int onDsMix(CPUState *cpu, uint32_t)
{
	const uint32_t v = cpu->gpr[4];
	psx_native_cd_mix(psx_mod_read_byte(v), psx_mod_read_byte(v + 1), psx_mod_read_byte(v + 2), psx_mod_read_byte(v + 3));
	cpu->gpr[2] = 1;
	return 1;
}
int onDsStatus(CPUState *cpu, uint32_t)
{
	// the bits the game looks at: 0x20 reading, 0x80 playing, 0x40 seeking (never), 0x02 the motor on
	cpu->gpr[2] = drive.streaming ? (drive.audioTrack ? 0x82 : 0x22) : 0x02;
	return 1;
}
int onDsSync(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = 2;	// complete
	return 1;
}
int onDsReady(CPUState *cpu, uint32_t)
{
	// 4: an audio track's end (the game's monitor stops it there); otherwise nothing new
	cpu->gpr[2] = drive.trackEnded ? 4 : 0;
	drive.trackEnded = false;
	return 1;
}
int onDsSyncCallback(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = drive.syncCallback, drive.syncCallback = cpu->gpr[4];
	return 1;
}
int onDsReadyCallback(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = drive.readyCallback, drive.readyCallback = cpu->gpr[4];
	return 1;
}
int onDsDataCallback(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = drive.dataCallback, drive.dataCallback = cpu->gpr[4];
	return 1;
}
// DsRead2(position, mode): bit 8, a movie: its sectors into StSetRing's buffer
int onDsRead2(CPUState *cpu, uint32_t)
{
	// DEADSTATION_RAM_DUMP=FILE (tests): the PlayStation's RAM as a movie starts (its player's code, for capture)
	if (const char *dump = std::getenv("DEADSTATION_RAM_DUMP"))
		if (FILE *f = std::fopen(dump, "wb"))
		{
			std::fwrite(g_psx_ram, 1, 0x200000, f);
			std::fclose(f);
		}
	drive.mode = (uint8_t)cpu->gpr[5];
	driveRead(fromMsf(cpu->gpr[4]), (cpu->gpr[5] & 0x100) != 0);
	cpu->gpr[2] = 1;
	return 1;
}
// DsGetToc(table): [0] the disc's end, [t] track t's start, as minute, second, 0, 0
int onDsGetToc(CPUState *cpu, uint32_t)
{
	const uint32_t table = cpu->gpr[4];
	const int tracks = psx_native_track_count();
	toMsf(psx_native_disc_sectors(), table);
	for (int t = 1; t <= tracks; t++)
		toMsf(psx_native_track_start(t), table + 4u * (uint32_t)t);
	cpu->gpr[2] = (uint32_t)tracks;
	return 1;
}
// a file's place in the image's ISO 9660 tree: its path as the game gives it (\DIR\NAME.EXT;1)
bool isoFind(const char *path, uint32_t& lba, uint32_t& size)
{
	uint8_t sector[2048];
	std::lock_guard<std::mutex> hold(isoLock);
	if (!psx_cdrom_read_user_sector(16, sector))
		return false;
	auto le32 = [](const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; };
	uint32_t dir = le32(sector + 156 + 2), dirSize = le32(sector + 156 + 10);
	for (const char *part = path; *part != 0;)
	{
		while (*part == '\\')
			part++;
		const char *partEnd = part;
		while (*partEnd != 0 && *partEnd != '\\')
			partEnd++;
		const size_t length = (size_t)(partEnd - part);
		if (length == 0)
			break;
		bool found = false;
		for (uint32_t done = 0; done < dirSize && !found; done += 2048, dir++)
		{
			if (!psx_cdrom_read_user_sector(dir, sector))
				return false;
			for (uint32_t o = 0; o < 2048 && sector[o] != 0; o += sector[o])
				if (sector[o + 32] == length && std::memcmp(sector + o + 33, part, length) == 0)
				{
					lba = le32(sector + o + 2), size = le32(sector + o + 10);
					found = true;
					break;
				}
		}
		if (!found)
			return false;
		dir = lba, dirSize = size;
		part = partEnd;
	}
	return true;
}
// DsSearchFile(CdlFILE *, name): position (BCD), size, the name
int onDsSearchFile(CPUState *cpu, uint32_t)
{
	char path[64] = {};
	for (uint32_t i = 0; i < sizeof(path) - 1; i++)
		if ((path[i] = (char)psx_mod_read_byte(cpu->gpr[5] + i)) == 0)
			break;
	const uint32_t file = cpu->gpr[4];
	uint32_t lba = 0, size = 0;
	if (!isoFind(path, lba, size))
	{
		std::fprintf(stderr, "deadstation tekken3: no file %s in the disc image\n", path);
		cpu->gpr[2] = 0;
		return 1;
	}
	psx_mod_write_word(file, 0);
	toMsf(lba, file);
	psx_mod_write_word(file + 4, size);
	const char *base = std::strrchr(path, '\\');
	base = base != nullptr ? base + 1 : path;
	for (uint32_t i = 0; i < 16; i++)
		psx_mod_write_byte(file + 8 + i, i < std::strlen(base) ? (uint8_t)base[i] : 0);
	std::fprintf(stderr, "deadstation tekken3: %s at sector %u, %u bytes\n", path, lba, size);
	cpu->gpr[2] = file;
	return 1;
}
// a movie's video sector into the game's ring, as LIBCD's handler leaves it: the frame's sectors contiguous (their
// headers beside them, a frame that won't fit before the end leaving a wrap mark), status 2 on its first header when
// complete. False: no room yet (the stream waits: no frame is ever dropped).
bool ringSector(const uint8_t *user)
{
	const uint16_t *id = (const uint16_t *)user;
	const uint32_t index = id[2], count = id[3];
	const uint32_t ring = ramWord(StRing) & 0x1FFFFF, slots = ramWord(StRingSize);
	if (id[0] != 0x0160 || count == 0 || count > slots || index >= count)
		return true;
	auto header = [&](uint32_t slot) { return (uint16_t *)(g_psx_ram + ring + slot * 0x20); };
	if (index == 0)
	{
		uint32_t at = drive.ringWrite;
		// (a frame may not end on the ring's last slot: StFreeRing would leave the game's read position past the ring)
		if (at + count >= slots)
		{
			if (at < slots)
			{
				if (*header(at) > 1)
					return false;
				*header(at) = 1;
			}
			at = 0;
		}
		for (uint32_t i = 0; i < count; i++)
			if (*header(at + i) > 1)
				return false;
		drive.ringWrite = at;
	}
	const uint32_t slot = drive.ringWrite + index;
	if (slot >= slots)
		return true;
	std::memcpy(header(slot), user, 0x20);
	std::memcpy(g_psx_ram + ring + slots * 0x20 + slot * 0x7E0, user + 0x20, 0x7E0);
	*header(slot) = 3;	// in use, not ready
	if (index + 1 == count)
	{
		*header(drive.ringWrite) = 2;	// the frame ready for StGetNext
		driveRingFrames++;
		drive.ringWrite += count;
		if (drive.ringWrite >= slots)
			drive.ringWrite = 0;
	}
	return true;
}
// StGetNext with no frame ready: the movie's loop would spin until the blank that brings the next (its own body then runs
// and answers "not ready"), so guest time goes to that blank
constexpr uint32_t StGetNextFunction = 0x80084CE8, StReadSlot = 0x800AFA54;
int onStGetNext(CPUState *, uint32_t)
{
	if (!drive.ring)
		return 0;
	const uint32_t ring = ramWord(StRing) & 0x1FFFFF;
	uint32_t slot = ramWord(StReadSlot);
	uint16_t status = *(const uint16_t *)(g_psx_ram + ring + slot * 0x20);
	if (status == 1)
		status = *(const uint16_t *)(g_psx_ram + ring);
	if (status != 2)
	{
		psx_cyc_batch_flush();
		psx_advance_cycles(interrupts_cycles_to_vblank() + 1u);
	}
	return 0;
}

// each vertical blank: the sectors the drive would have reached
void onDriveVBlank()
{
	static unsigned blanks = 0;
	if (++blanks % 600 == 0)
		std::fprintf(stderr, "deadstation tekken3: so far %u XA sectors, %u audio track sectors, %u movie frames\n", driveXaSectors,
				driveAudioSectors, driveRingFrames);
	if (bnsWaiting && bnsDone.load())
		bnsFinish();
	if (!drive.streaming)
		return;
	// (as fast as the CD's sound is being played: sound doesn't go by frames. A console's 75 sectors a second at a 60 Hz blank)
	drive.budget = std::min(drive.budget + (drive.audioTrack ? 1.25f : (drive.mode & 0x80) ? 2.5f : 1.25f) * (float)spu_cd_pace(), 16.f);
	discWake.notify_one();
	while (drive.budget >= 1.f && drive.streaming)
	{
		if (drive.audioTrack && drive.lba >= drive.trackEnd)
		{
			drive.streaming = false, drive.trackEnded = true;
			return;
		}
		// an earlier stream's sectors out first: a full queue of them would stop the disc thread reading this one's
		uint32_t head = queueHead.load();
		const uint32_t tail = queueTail.load();
		while (head != tail && sectorQueue[head % SectorQueue].generation != drive.generation)
			queueHead = ++head;
		// not read yet: the stream waits for the disc thread (nothing here waits)
		if (workerGeneration.load() != drive.generation || workerNext.load() <= drive.lba)
			return;
		if (head != tail && sectorQueue[head % SectorQueue].lba == drive.lba)
		{
			const uint8_t *raw = sectorQueue[head % SectorQueue].raw;
			if (drive.audioTrack)
				psx_native_cdda_sector(raw), driveAudioSectors++;
			else if ((raw[0x12] & 0x04) != 0)
			{
				psx_native_xa_sector(raw);
				driveXaSectors++;
				if (!drive.ring)
					psx_mod_write_word(MusicPosition, drive.lba);
			}
			else if (drive.ring && !ringSector(raw + 24))
				return;	// the frame waits for room; the budget stays
			queueHead = head + 1;
		}
		drive.lba++;
		streamAt = drive.lba;
		drive.budget -= 1.f;
	}
}

// VSync(-1): the blank count from RAM, without the library's reads of GPUSTAT and a timer it doesn't use (each read
// catches every simulated device up to the present). DEADSTATION_SDK_OFF=vsyncquery: the game's own.
constexpr uint32_t VSyncFunction = 0x800859A8, VBlankCount = 0x8009AC68;
int onVSync(CPUState *cpu, uint32_t)
{
	if ((int32_t)cpu->gpr[4] >= 0)
		return 0;
	cpu->gpr[2] = ramWord(VBlankCount);
	return 1;
}

// ---- The skeleton's matrix math as host code ----
//
// Leaf functions that do only arithmetic, through the GTE's MVMVA on the PlayStation (here each GTE operation is a call
// through its simulation): the same integer results, including the GTE's 16-bit saturation of its IR registers and the
// rotation left in the GTE as these leave it. DEADSTATION_SDK_OFF=gtemath: the game's own.
inline uint32_t guestWord(uint32_t a)
{
	return (a >> 24) == 0x1F ? psx_mod_read_word(a) : *(const uint32_t *)(g_psx_ram + (a & 0x1FFFFC));
}
inline void guestWriteWord(uint32_t a, uint32_t v)
{
	if ((a >> 24) == 0x1F)
		psx_mod_write_word(a, v);
	else
		*(uint32_t *)(g_psx_ram + (a & 0x1FFFFC)) = v;
}
inline int32_t saturate16(int64_t v)
{
	return v > 0x7FFF ? 0x7FFF : v < -0x8000 ? -0x8000 : (int32_t)v;
}
struct Rotation
{
	int32_t m[9];
	void read(uint32_t matrix, CPUState *cpu)
	{
		const uint32_t w0 = guestWord(matrix), w1 = guestWord(matrix + 4), w2 = guestWord(matrix + 8), w3 = guestWord(matrix + 12), w4 = guestWord(matrix + 16);
		m[0] = lo16(w0), m[1] = hi16(w0), m[2] = lo16(w1), m[3] = hi16(w1), m[4] = lo16(w2), m[5] = hi16(w2), m[6] = lo16(w3), m[7] = hi16(w3), m[8] = lo16(w4);
		cpu->gte_ctrl[0] = w0, cpu->gte_ctrl[1] = w1, cpu->gte_ctrl[2] = w2, cpu->gte_ctrl[3] = w3, cpu->gte_ctrl[4] = (uint32_t)(int32_t)(int16_t)w4;
	}
	int64_t row(int r, int32_t x, int32_t y, int32_t z) const
	{
		return (int64_t)m[3 * r] * x + (int64_t)m[3 * r + 1] * y + (int64_t)m[3 * r + 2] * z;
	}
};
// out's rotation = a's rotation x the matrix b (a column at a time, as the GTE does)
void multiplyRotation(const Rotation& a, uint32_t b, uint32_t out)
{
	const uint32_t w0 = guestWord(b), w1 = guestWord(b + 4), w2 = guestWord(b + 8), w3 = guestWord(b + 12), w4 = guestWord(b + 16);
	const int32_t m00 = lo16(w0), m01 = hi16(w0), m02 = lo16(w1), m10 = hi16(w1), m11 = lo16(w2), m12 = hi16(w2), m20 = lo16(w3), m21 = hi16(w3), m22 = lo16(w4);
	const int32_t i00 = saturate16(a.row(0, m00, m10, m20) >> 12), i10 = saturate16(a.row(1, m00, m10, m20) >> 12), i20 = saturate16(a.row(2, m00, m10, m20) >> 12);
	const int32_t i01 = saturate16(a.row(0, m01, m11, m21) >> 12), i11 = saturate16(a.row(1, m01, m11, m21) >> 12), i21 = saturate16(a.row(2, m01, m11, m21) >> 12);
	const int32_t i02 = saturate16(a.row(0, m02, m12, m22) >> 12), i12 = saturate16(a.row(1, m02, m12, m22) >> 12), i22 = saturate16(a.row(2, m02, m12, m22) >> 12);
	guestWriteWord(out, (uint32_t)i01 << 16 | ((uint32_t)i00 & 0xFFFF));
	guestWriteWord(out + 4, (uint32_t)i10 << 16 | ((uint32_t)i02 & 0xFFFF));
	guestWriteWord(out + 8, (uint32_t)i12 << 16 | ((uint32_t)i11 & 0xFFFF));
	guestWriteWord(out + 12, (uint32_t)i21 << 16 | ((uint32_t)i20 & 0xFFFF));
	guestWriteWord(out + 16, (uint32_t)i22);
}
// MulMatrix0(m0, m1, m2)
int onMulMatrix0(CPUState *cpu, uint32_t)
{
	Rotation a;
	a.read(cpu->gpr[4], cpu);
	multiplyRotation(a, cpu->gpr[5], cpu->gpr[6]);
	cpu->gpr[2] = cpu->gpr[6];
	return 1;
}
// comp_matrix(parent, local, out, translated): out = parent x local; its translation parent x translated's + parent's
int onCompMatrix(CPUState *cpu, uint32_t)
{
	const uint32_t parent = cpu->gpr[4], out = cpu->gpr[6], t = cpu->gpr[7];
	Rotation a;
	a.read(parent, cpu);
	multiplyRotation(a, cpu->gpr[5], out);
	const int32_t x = (int16_t)guestWord(t + 0x14), y = (int16_t)guestWord(t + 0x18), z = (int16_t)guestWord(t + 0x1C);
	guestWriteWord(out + 0x14, (uint32_t)((int32_t)(a.row(0, x, y, z) >> 12) + (int32_t)guestWord(parent + 0x14)));
	guestWriteWord(out + 0x18, (uint32_t)((int32_t)(a.row(1, x, y, z) >> 12) + (int32_t)guestWord(parent + 0x18)));
	guestWriteWord(out + 0x1C, (uint32_t)((int32_t)(a.row(2, x, y, z) >> 12) + (int32_t)guestWord(parent + 0x1C)));
	cpu->gpr[2] = out;
	return 1;
}
// 0x8003B71C(parent, local, out): the same with local's own 32-bit translation, split at 15 bits as the GTE's 16-bit input
// needs: rotated in two parts, the high one by 8 (2^15 >> 12)
int onCompMatrixLong(CPUState *cpu, uint32_t)
{
	const uint32_t parent = cpu->gpr[4], local = cpu->gpr[5], out = cpu->gpr[6];
	Rotation a;
	a.read(parent, cpu);
	multiplyRotation(a, local, out);
	int32_t hi[3], lo[3];
	for (int i = 0; i < 3; i++)
	{
		const int32_t v = (int32_t)guestWord(local + 0x14 + 4 * (uint32_t)i);
		if (v < 0)
			hi[i] = -((int32_t)(0u - (uint32_t)v) >> 15), lo[i] = -(int32_t)((0u - (uint32_t)v) & 0x7FFF);
		else
			hi[i] = v >> 15, lo[i] = v & 0x7FFF;
		hi[i] = (int16_t)hi[i], lo[i] = (int16_t)lo[i];
	}
	for (int r = 0; r < 3; r++)
	{
		const int32_t high = (int32_t)a.row(r, hi[0], hi[1], hi[2]), low = (int32_t)(a.row(r, lo[0], lo[1], lo[2]) >> 12);
		guestWriteWord(out + 0x14 + 4 * (uint32_t)r, (uint32_t)(low + high * 8 + (int32_t)guestWord(parent + 0x14 + 4 * (uint32_t)r)));
	}
	cpu->gpr[2] = out;
	return 1;
}
// SquareRoot0(a), by the GTE's leading bit count and the library's table
constexpr uint32_t SquareRootTable = 0x80098D08;
uint32_t squareRoot(int32_t a)
{
	const uint32_t bits = a < 0 ? ~(uint32_t)a : (uint32_t)a;
	int lz = bits == 0 ? 32 : __builtin_clz(bits);
	if (lz == 32)
		return 0;
	lz &= ~1;
	const int32_t i = lz < 24 ? a >> (24 - lz) : a << (lz - 24);
	return (uint32_t)((int32_t)ramHalf(SquareRootTable + (uint32_t)(i - 0x40) * 2) << ((31 - lz) >> 1)) >> 12;
}
int onSquareRoot0(CPUState *cpu, uint32_t)
{
	cpu->gpr[2] = squareRoot((int32_t)cpu->gpr[4]);
	return 1;
}

// ---- The stage natively ----
//
// The floor (0x80048FF8: a grid of tiles, 0x8009E824 x 0x8009E828, each a textured quad with a shade by distance at its
// corners) and the backdrop (the loop in 0x8006D014: cells of vertices and polygons, 0x80036AF8 and 0x8006CC28)
// as host code, no GTE and no PlayStation packets: the
// game's matrices, its data, the frame's triangles in the same list as the fighters'. DEADSTATION_SDK_OFF=stage: the game's.
constexpr uint32_t FloorDraw = 0x80048FF8, FloorLit = 0x80049848, FloorOrigin = 0x8004ADC4, BackdropVertices = 0x80036AF8, BackdropPolygons = 0x8006CC28;
constexpr uint32_t BackdropLoop = 0x8006D014, BackdropLoopEnd = 0x8006D7CC;
struct View
{
	float r[9], t[3], ofx, ofy, h;
	void read(const CPUState *cpu)
	{
		const uint32_t *g = cpu->gte_ctrl;
		const float k = 1.0f / 4096.0f;
		r[0] = lo16(g[0]) * k, r[1] = hi16(g[0]) * k, r[2] = lo16(g[1]) * k, r[3] = hi16(g[1]) * k, r[4] = lo16(g[2]) * k, r[5] = hi16(g[2]) * k;
		r[6] = lo16(g[3]) * k, r[7] = hi16(g[3]) * k, r[8] = lo16(g[4]) * k;
		t[0] = (float)(int32_t)g[5], t[1] = (float)(int32_t)g[6], t[2] = (float)(int32_t)g[7];
		ofx = (int32_t)g[24] / 65536.0f, ofy = (int32_t)g[25] / 65536.0f, h = (float)(uint16_t)g[26];
	}
	// a vertex at p (x, y, z halves) on the screen; false: behind the camera
	bool project(uint32_t p, NativeCornerOut& out) const
	{
		return projectAt(ramHalf(p), ramHalf(p + 2), ramHalf(p + 4), out);
	}
	bool projectAt(float x, float y, float z, NativeCornerOut& out) const
	{
		const float vz = r[6] * x + r[7] * y + r[8] * z + t[2];
		if (vz < 1.0f)
		{
			out.z = 0;
			return false;
		}
		const float q = h / vz;
		out.sx = ofx + (r[0] * x + r[1] * y + r[2] * z + t[0]) * q;
		out.sy = ofy + (r[3] * x + r[4] * y + r[5] * z + t[1]) * q;
		out.z = vz;
		return true;
	}
};
NativeCornerOut stageVertices[256];
bool backdropReady = false;

inline bool anyInside(float a, float b, float c, float d, float size)
{
	return (a >= 0 && a < size) || (b >= 0 && b < size) || (c >= 0 && c < size) || (d >= 0 && d < size);
}
inline void quadTriangles(std::vector<NativeTri>& to, const NativeCornerOut *c, uint16_t page, uint16_t clut, uint8_t kind)
{
	to.push_back({ { c[0], c[1], c[2] }, page, clut, kind });
	to.push_back({ { c[1], c[3], c[2] }, page, clut, kind });
}

// The floor's two routines share everything but how a vertex is shaded: kind 0 (0x80048FF8) by its depth; kind 1 and 2
// (0x80049848, and it sets whether shadows show first) by the light on each fighter, or the one over their middle.
void addFlavour(FighterList& list, const View& view, int32_t worldX, int32_t worldZ, float horizon);
struct FloorLights
{
	int kind = 0;
	int32_t x[2] = { 0, 0 }, z[2] = { 0, 0 };	// the fighters (kind 2: x[0], z[0] their middle)
};
int floorGrid(CPUState *cpu, uint32_t ra, int32_t cx, int32_t cz, const FloorLights& lit)
{
	const int32_t tileSize = (int32_t)rw(0x8009E780), cols = (int32_t)rw(0x8009E824), rows = (int32_t)rw(0x8009E828);
	if (tileSize <= 0 || cols <= 0 || rows <= 0 || cols > 16 || rows > 16)
		return 0;
	int32_t ix = cx / tileSize, iz = cz / tileSize;
	if (cx < 0)
		ix--;
	if (cz < 0)
		iz--;
	const int32_t worldX = ix * tileSize, worldZ = iz * tileSize;
	ix %= cols, iz %= rows;
	if (ix < 0)
		ix += cols;
	if (iz < 0)
		iz += rows;
	int32_t row = rows - (iz + 1);
	// the game's own routine sets the matrix for the grid at this origin
	cpu->gpr[4] = (uint32_t)worldX, cpu->gpr[5] = 0, cpu->gpr[6] = (uint32_t)worldZ;
	psx_dispatch_call(cpu, FloorOrigin, ra);
	cpu->gpr[31] = ra;
	const uint32_t mesh = rw(0x800ADD14);
	if (mesh == 0)
		return 1;
	const uint32_t vertices = rw(mesh + 0xC), count = rw(mesh + 0x10), table = rw(mesh + 0x1C);
	const uint32_t pattern = rw(0x8009E7F8);
	const int32_t shade = (int16_t)psx_mod_read_half(pattern + 0x326);
	uint32_t packet = mesh + (rw(PacketSet) & 1) * 0x1450 + 0x5E4;
	FighterList& list = nativeList(ramWord(ramWord(OrderingTables) + 4));
	View view;
	view.read(cpu);
	if (view.h <= 0)
		return 1;
	// (the grid's vertices are flat: a word each, x above z, as 0x8004A364 feeds them to the GTE)
	float gridX0 = 1e30f, gridX1 = -1e30f, gridZ0 = 1e30f, gridZ1 = -1e30f;
	for (uint32_t i = 0; i < count && i < 256; i++)
	{
		const uint32_t word = ramWord(vertices + i * 4);
		const float gx = (float)hi16(word), gz = (float)lo16(word);
		view.projectAt(gx, 0.f, gz, stageVertices[i]);
		gridX0 = std::min(gridX0, gx), gridX1 = std::max(gridX1, gx), gridZ0 = std::min(gridZ0, gz), gridZ1 = std::max(gridZ1, gz);
	}
	drawnFloor[0] = worldX + gridX0, drawnFloor[1] = worldX + gridX1, drawnFloor[2] = worldZ + gridZ0, drawnFloor[3] = worldZ + gridZ1;
	drawnFloorFresh = true;
	// kinds 1 and 2: each vertex's light (0x80049848), by the square of its distance
	static uint32_t litShade[256];
	if (lit.kind != 0)
	{
		const int64_t cap = lit.kind == 1 ? 254 : (int32_t)rw(0x800ADD94);
		for (uint32_t i = 0; i < count && i < 256; i++)
		{
			const uint32_t word = ramWord(vertices + i * 4);
			const int64_t wx = worldX + (int32_t)hi16(word), wz = worldZ + (int32_t)lo16(word);
			const auto reach = [&](int f, int64_t power) {
				const int64_t dx = lit.x[f] - wx, dz = lit.z[f] - wz, d = dx * dx + dz * dz;
				return power / (d != 0 ? d : 1);
			};
			const int64_t v = lit.kind == 1 ? reach(0, 0x0F200000) + reach(1, 0x0F200000) : reach(0, 800000000);
			const uint32_t c = (uint32_t)std::min(v, cap);
			litShade[i] = c | c << 8 | c << 16;
		}
	}
	// the distance the shade starts from
	int32_t from;
	if (rw(GameMode) == 8)
		from = std::abs((int32_t)rw(0x800A85DC));
	else
	{
		const int32_t dx = (int32_t)rw(0x800A85D4) - ((int32_t)rw(0x800AA190) + (int32_t)rw(0x800ABA1C)) / 2;
		const int32_t dz = (int32_t)rw(0x800A85DC) - ((int32_t)rw(0x800AA198) + (int32_t)rw(0x800ABA24)) / 2;
		from = (int32_t)squareRoot(dx * dx + dz * dz);
	}
	const int32_t shadeScale = 0x4000000 / std::max<int32_t>((int32_t)rw(0x800AE21C), 1);
	auto shadeAt = [&](float z) {
		const int32_t d = std::max<int32_t>((int32_t)((uint32_t)z >> 5 & 0xFFFF) - from, 0);
		const int32_t i = std::min<int32_t>((d * shadeScale) >> 16, 0x3FF);
		const int32_t v = (((int32_t)psx_mod_read_byte(0x8001DF6C + (uint32_t)i) * shade) >> 7) + 0x80;
		const uint32_t c = (uint32_t)std::clamp<int32_t>(v, 0, 0xFE);
		return c | c << 8 | c << 16;
	};
	// the floor's far edge on the screen: its grid's highest corner in front of the eye
	{
		float horizon = 480.f;
		for (uint32_t i = 0; i < count && i < 256; i++)
			if (stageVertices[i].z > 0)
				horizon = std::min(horizon, stageVertices[i].sy);
		addFlavour(list, view, worldX, worldZ, std::max(horizon, 0.f));
	}
	static const uint8_t flipOrder[4][4] = { { 0, 1, 2, 3 }, { 1, 0, 3, 2 }, { 2, 3, 0, 1 }, { 3, 2, 1, 0 } };
	for (int32_t r = 0; r < rows; r++)
	{
		int32_t col = ix;
		for (int32_t c = 0; c < cols; c++, packet += 13 * 4)
		{
			const uint32_t corners = ramWord(table + (uint32_t)((c * 4 + r * 0x28) & 0xFFFF));
			NativeCornerOut q[4];
			for (int j = 0; j < 4; j++)
				q[j] = stageVertices[corners >> (8 * j) & 0xFF];
			const bool visible = q[0].z > 0 && q[1].z > 0 && q[2].z > 0 && q[3].z > 0 && anyInside(q[0].sy, q[1].sy, q[2].sy, q[3].sy, 480.f) &&
					anyInside(q[0].sx, q[1].sx, q[2].sx, q[3].sx, 368.f);
			if (visible)
			{
				const uint32_t cell = pattern + (uint32_t)row * 0x50 + 4 + (uint32_t)col * 8;
				const uint32_t w0 = ramWord(cell), w1 = ramWord(cell + 4);
				const uint32_t rotation = 0x8009E800 + (w0 >> 30) * 8;
				const uint8_t *order = flipOrder[(w0 >> 25) & 3];
				const uint32_t base = w0 & 0xFFFF;
				for (int j = 0; j < 4; j++)
				{
					q[j].uv = (uint16_t)(base + (uint16_t)ramHalf(rotation + (uint32_t)order[j] * 2));
					uint32_t rgb = lit.kind != 0 ? litShade[corners >> (8 * j) & 0xFF] : shade != 0 ? shadeAt(q[j].z) : (ramWord(packet + 4 + (uint32_t)j * 12) & 0xFFFFFF);
					if (currentStage == CloudStage)
					{
						const uint32_t word = ramWord(vertices + (corners >> (8 * j) & 0xFF) * 4);
						rgb = cloudShade(rgb, (float)(worldX + (int32_t)hi16(word)), (float)(worldZ + (int32_t)lo16(word)));
					}
					q[j].rgb = moodColour(rgb, true);
				}
				const uint8_t kind = (ramWord(packet + 4) >> 24 & 2) != 0 ? 2 : 0;
				quadTriangles(list.stage, q, (uint16_t)((w0 & 0x39FF0000u) >> 16), (uint16_t)(w1 >> 16), kind);
			}
			if (++col >= cols)
				col = 0;
		}
		if (--row < 0)
			row = rows - 1;
	}
	return 1;
}

// A stage's ground mist (no setting: a stage's flavour): a soft band along the far edge of the floor, added over the backdrop and
// the far floor, behind the fighters, in slow-moving patches. Columns of quads, brightest at the floor's edge, nothing at the band's
// top and bottom. The game's screen is 368 x 480.
struct Mist
{
	int stage;
	uint8_t r, g, b;		// at its thickest
	float above, below;		// how far it reaches up the backdrop and down the floor, in the game's lines
};
constexpr Mist Mists[] = {
	{ 5, 0x62, 0x62, 0x66, 150.f, 90.f },	// Lei's Hong Kong street: grey, in the distance
	{ 0, 0x6C, 0x46, 0x22, 130.f, 70.f },	// the city rooftop: amber haze at sunset
	{ 2, 0x34, 0x48, 0x34, 100.f, 120.f },	// the forest: ground mist among the trees
	{ 9, 0x30, 0x3C, 0x54, 120.f, 60.f },	// the guarded yard: fog along the wall
};

void addMist(FighterList& list, float horizon)
{
	for (const Mist& m : Mists)
	{
		if (m.stage != currentStage)
			continue;
		// rows from nothing at the top, thickest at the floor's edge, to nothing below it, each with its own slow pattern: patches
		// that drift, where one pattern down the whole height would read as vertical stripes
		constexpr int Columns = 24, Rows = 4;
		const float t = (float)frameSerial * 0.01f;
		const float y[Rows] = { horizon - m.above, horizon - m.above * 0.45f, horizon, horizon + m.below };
		constexpr float Thickness[Rows] = { 0.f, 0.5f, 1.f, 0.f };
		const auto cornerY = [&](int column, int row) {
			return row == 1 ? y[row] + 14.f * std::sin((float)column * 0.21f + t * 0.7f) : y[row];
		};
		const auto colour = [&](int column, int row) {
			const float c = (float)column, r = (float)row;
			const float k = Thickness[row] * (1.f + 0.16f * std::sin(c * 0.24f + t + r * 1.7f) + 0.09f * std::sin(c * 0.53f - t * 1.4f + r * 2.9f));
			return (uint32_t)std::min(m.r * k, 255.f) | (uint32_t)std::min(m.g * k, 255.f) << 8 | (uint32_t)std::min(m.b * k, 255.f) << 16;
		};
		for (int i = 0; i < Columns; i++)
		{
			const float x0 = 368.f * (float)i / Columns, x1 = 368.f * (float)(i + 1) / Columns;
			for (int r = 0; r + 1 < Rows; r++)
			{
				list.mist.push_back({ { x0, x1, x0 }, { cornerY(i, r), cornerY(i + 1, r), cornerY(i, r + 1) }, { colour(i, r), colour(i + 1, r), colour(i, r + 1) } });
				list.mist.push_back({ { x1, x1, x0 }, { cornerY(i + 1, r), cornerY(i + 1, r + 1), cornerY(i, r + 1) }, { colour(i + 1, r), colour(i + 1, r + 1), colour(i, r + 1) } });
			}
		}
		return;
	}
}

// What floats in a stage (no setting: a stage's flavour): soft spots of added light, each with its place in the world from a number
// and the clock alone, so nothing is kept and nothing of the game's is touched. Each stays where it is in the world as the fighters
// move, coming round again past the edge of a box about them; one nearer the eye than they are is drawn over them.
enum class Float : uint8_t { Soft, Flutter, Rain, Puff };
struct Floating
{
	int stage;
	Float kind;
	int count;
	uint8_t r, g, b;
	float size;				// half its width, in world units
	float vx, vy, vz;		// a second (y is down)
	float sway;				// how far it wanders about that
	float top;				// the highest it goes, above the floor (y < 0)
	float box;				// how far from the fighters' middle, each way
	float pulse;			// > 0: its brightness beats, so many a second
	bool fadesUp;			// dimmer the higher it is
};
constexpr Floating Floats[] = {
	{ 2, Float::Soft, 90, 0xB8, 0xF0, 0x60, 46.f, 40.f, -30.f, 30.f, 280.f, -2700.f, 4000.f, 1.7f, false },		// the forest: motes of light
	{ 4, Float::Soft, 300, 0xE0, 0xE8, 0xF8, 36.f, 60.f, 430.f, 20.f, 160.f, -4600.f, 4300.f, 0.f, false },		// the palace: light snow
	{ 6, Float::Soft, 170, 0xFF, 0x70, 0x18, 34.f, 30.f, -650.f, -20.f, 140.f, -3400.f, 4000.f, 5.f, true },		// the temple: rising embers
	{ 7, Float::Flutter, 170, 0xFF, 0x98, 0xB4, 62.f, 430.f, 300.f, 120.f, 270.f, -4200.f, 4300.f, 0.f, false },	// the schoolyard: blossom petals
	{ 8, Float::Soft, 110, 0xA8, 0x98, 0x78, 34.f, 18.f, 26.f, -14.f, 130.f, -3200.f, 3600.f, 0.6f, false },		// the dojo: dust motes
	{ 5, Float::Rain, 260, 0x70, 0x80, 0x98, 1.2f, -700.f, 7000.f, 160.f, 0.f, -4800.f, 4300.f, 0.f, false },		// Lei's street: rain through the mist
	{ 14, Float::Puff, 22, 0x4C, 0x4E, 0x54, 230.f, 40.f, -420.f, 15.f, 150.f, -3000.f, 2800.f, 0.f, true },		// the lab: steam
};

// a soft spot: brightest at its middle, nothing at its edge. The game's screen has more lines than columns for the same distance.
constexpr float LinesPerColumn = 1.74f;
void softSpot(std::vector<DeadstationGlow>& to, float x, float y, float halfWidth, float halfHeight, uint32_t rgb)
{
	// (round when it is big enough to be seen as a shape)
	static constexpr float Round[9][2] = { { 1, 0 }, { 0.707f, 0.707f }, { 0, 1 }, { -0.707f, 0.707f }, { -1, 0 }, { -0.707f, -0.707f }, { 0, -1 }, { 0.707f, -0.707f }, { 1, 0 } };
	const int step = halfWidth >= 3.f ? 1 : 2;
	for (int k = 0; k < 8; k += step)
	{
		const float *a = Round[k], *b = Round[k + step];
		to.push_back({ { x, x + a[0] * halfWidth, x + b[0] * halfWidth }, { y, y + a[1] * halfHeight, y + b[1] * halfHeight }, { rgb, 0, 0 } });
	}
}
inline uint32_t glowColour(float r, float g, float b, float k)
{
	return (uint32_t)std::min(r * k, 255.f) | (uint32_t)std::min(g * k, 255.f) << 8 | (uint32_t)std::min(b * k, 255.f) << 16;
}

// A drop landing: a ripple opening on the floor and gone: a soft band fading out to both sides, flat in the world, so what stands in front of it
// hides it (a little nearer than the floor, or the floor hides it). The same on every stage it rains or leaks on.
constexpr float SplashFor = 0.45f;
void addSplash(FighterList& list, const View& view, float x, float z, float t)
{
	constexpr float Reach = 190.f, Strength = 1.f, Nearer = 0.985f;
	const float radius = Reach * (0.2f + 0.8f * t), inner = radius * 0.72f, outer = radius * 1.12f;
	const uint32_t rgb = glowColour(0x90, 0xA0, 0xC0, (1.f - t) * Strength);
	constexpr int Sides = 10;
	NativeCornerOut in0, mid0, out0;
	if (!view.projectAt(x + inner, -4.f, z, in0) || !view.projectAt(x + radius, -4.f, z, mid0) || !view.projectAt(x + outer, -4.f, z, out0))
		return;
	for (int k = 1; k <= Sides; k++)
	{
		const float a = (float)k * (6.2832f / Sides), ca = std::cos(a), sa = std::sin(a);
		NativeCornerOut in1, mid1, out1;
		if (!view.projectAt(x + ca * inner, -4.f, z + sa * inner, in1) || !view.projectAt(x + ca * radius, -4.f, z + sa * radius, mid1) ||
				!view.projectAt(x + ca * outer, -4.f, z + sa * outer, out1))
			return;
		const float zi0 = in0.z * Nearer, zm0 = mid0.z * Nearer, zo0 = out0.z * Nearer, zi1 = in1.z * Nearer, zm1 = mid1.z * Nearer, zo1 = out1.z * Nearer;
		list.front.push_back({ { in0.sx, mid0.sx, mid1.sx }, { in0.sy, mid0.sy, mid1.sy }, { 0, rgb, rgb }, { zi0, zm0, zm1 } });
		list.front.push_back({ { in0.sx, mid1.sx, in1.sx }, { in0.sy, mid1.sy, in1.sy }, { 0, rgb, 0 }, { zi0, zm1, zi1 } });
		list.front.push_back({ { mid0.sx, out0.sx, out1.sx }, { mid0.sy, out0.sy, out1.sy }, { rgb, 0, 0 }, { zm0, zo0, zo1 } });
		list.front.push_back({ { mid0.sx, out1.sx, mid1.sx }, { mid0.sy, out1.sy, mid1.sy }, { rgb, 0, rgb }, { zm0, zo1, zm1 } });
		in0 = in1, mid0 = mid1, out0 = out1;
	}
}

// The hall on Julia's and Heihachi's stage is under a roof, the storm beyond its windows: so it doesn't rain in here, it leaks. A few places in the roof over the fighters, each letting go a
// drop at its own beat, straight down, each drop ringing out where it lands. The leaks are fixed in the world, in a block repeated over the floor, as the floor's pattern is.
constexpr int LeakStage = 12, Leaks = 7;
constexpr float LeakTile = 5200.f, LeakHeight = 4600.f;
void addLeaks(FighterList& list, const View& view, int32_t worldX, int32_t worldZ, float midX, float midZ)
{
	const float now = (float)flavourClock / 60.f;
	const auto nearest = [](float at, float centre, float tile) { return at + tile * std::round((centre - at) / tile); };
	for (int i = 0; i < Leaks; i++)
	{
		const uint32_t a = mix32((uint32_t)i * 2654435761u + 0x51u), b = mix32(a), c = mix32(b);
		const float x = nearest((float)(a & 0xFFFF) / 65536.f * LeakTile, midX, LeakTile), z = nearest((float)(b & 0xFFFF) / 65536.f * LeakTile, midZ, LeakTile);
		// (some drip quickly, some are slow; a drop falls as a thing does, from a stop)
		const float period = 0.9f + (float)(c & 255u) / 255.f * 2.4f, into = std::fmod(now + (float)((c >> 8) & 255u) * 0.041f, period);
		constexpr float Gravity = 9000.f;
		const float fall = 0.5f * Gravity * into * into;
		NativeCornerOut head;
		if (fall < LeakHeight)
		{
			const float y = -LeakHeight + fall, speed = Gravity * into;
			if (!view.projectAt(x - (float)worldX, y, z - (float)worldZ, head) || head.z < 200.f)
				continue;
			NativeCornerOut tail;
			const float back = std::min(speed * 0.022f, 260.f);
			if (!view.projectAt(x - (float)worldX, y - back, z - (float)worldZ, tail))
				continue;
			const float w = std::max(0.45f, 9.f * view.h / head.z);
			const uint32_t rgb = glowColour(0x90, 0xA0, 0xC0, 0.85f);
			list.front.push_back({ { tail.sx - w, tail.sx + w, head.sx - w }, { tail.sy, tail.sy, head.sy }, { 0, 0, rgb }, { tail.z, tail.z, head.z } });
			list.front.push_back({ { tail.sx + w, head.sx + w, head.sx - w }, { tail.sy, head.sy, head.sy }, { 0, rgb, rgb }, { tail.z, head.z, head.z } });
			continue;
		}
		// where it lands
		const float landed = std::sqrt(2.f * LeakHeight / Gravity), since = into - landed;
		if (since <= SplashFor)
			addSplash(list, view, x - (float)worldX, z - (float)worldZ, since / SplashFor);
	}
}

void addFloats(FighterList& list, const View& view, int32_t worldX, int32_t worldZ)
{
	const float now = (float)flavourClock / 60.f;
	const float midX = ((int32_t)rw(0x800AA190) + (int32_t)rw(0x800ABA1C)) * 0.5f, midZ = ((int32_t)rw(0x800AA198) + (int32_t)rw(0x800ABA24)) * 0.5f;
	NativeCornerOut middle;
	const float fighters = view.projectAt(midX - (float)worldX, -1000.f, midZ - (float)worldZ, middle) ? middle.z : 1e9f;
	const auto around = [](float v, float from, float span) {
		v = std::fmod(v - from, span);
		return from + (v < 0 ? v + span : v);
	};
	for (const Floating& f : Floats)
	{
		if (f.stage != currentStage)
			continue;
		for (int i = 0; i < f.count; i++)
		{
			const uint32_t a = mix32((uint32_t)i * 2654435761u + (uint32_t)f.stage), b = mix32(a), c = mix32(b);
			const float phase = (float)(c & 1023u) * 0.00614f;	// 0..2 pi
			const float x = around((float)(a & 0xFFFF) + f.vx * now + f.sway * std::sin(now * 0.7f + phase), midX - f.box, 2.f * f.box);
			const float z = around((float)(b & 0xFFFF) + f.vz * now + f.sway * std::cos(now * 0.53f + phase), midZ - f.box, 2.f * f.box);
			const float y = around((float)(c >> 16) + f.vy * now, f.top, -f.top);
			// every other drop lands: it fell through the floor's level this long ago, when it went back to the top, so a ring where it came down
			// (first: the drop itself is far above the screen by then, and is passed over just below)
			if (f.kind == Float::Rain && (i & 1) == 0)
			{
				const float since = (y - f.top) / f.vy;
				if (since < SplashFor)
					addSplash(list, view, x - f.vx * since - (float)worldX, z - f.vz * since - (float)worldZ, since / SplashFor);
			}
			NativeCornerOut head;
			if (!view.projectAt(x - (float)worldX, y, z - (float)worldZ, head) || head.z < 200.f)
				continue;
			if (head.sx < -40.f || head.sx > 408.f || head.sy < -80.f || head.sy > 560.f)
				continue;
			const float height = y / f.top;		// 0 at the floor, 1 at its top
			float k = std::clamp(1.3f - head.z / 10000.f, 0.25f, 1.f);
			if (f.pulse > 0)
			{
				const float beat = 0.5f + 0.5f * std::sin(now * f.pulse + phase * 3.f);
				k *= f.kind == Float::Soft && f.pulse < 2.f ? beat * beat * 0.85f + 0.15f : 0.55f + 0.45f * beat;
			}
			if (f.fadesUp)
				k *= 1.f - height * height;
			std::vector<DeadstationGlow>& to = head.z < fighters ? list.front : list.mist;
			const float scale = view.h / head.z;
			if (f.kind == Float::Rain)
			{
				// (just past the lens it would be a bar across the screen, not a drop)
				if (head.z < 1100.f)
					continue;
				// a streak from where it was a moment ago, fading behind it: in the world, tested against what's in the way, and fainter the nearer the eye, as it's more of a blur
				NativeCornerOut tail;
				constexpr float Streak = 0.045f;
				if (!view.projectAt(x - f.vx * Streak - (float)worldX, y - f.vy * Streak, z - f.vz * Streak - (float)worldZ, tail))
					continue;
				const float w = std::max(0.35f, f.size * scale * 3.5f), near = std::clamp((head.z - 1100.f) / 2400.f, 0.3f, 1.f);
				const uint32_t rgb = glowColour(f.r, f.g, f.b, k * near);
				list.front.push_back({ { tail.sx - w, tail.sx + w, head.sx - w }, { tail.sy, tail.sy, head.sy }, { 0, 0, rgb }, { tail.z, tail.z, head.z } });
				list.front.push_back({ { tail.sx + w, head.sx + w, head.sx - w }, { tail.sy, head.sy, head.sy }, { 0, rgb, rgb }, { tail.z, head.z, head.z } });
				continue;
			}
			float half = std::clamp(f.size * scale, 0.9f, 60.f), tall = half;
			if (f.kind == Float::Flutter)
			{
				// turning as it falls: seen edge on, then flat
				const float turn = std::fabs(std::cos(now * 2.1f + phase * 2.f));
				half *= 0.35f + 0.65f * turn;
				tall *= 0.45f + 0.55f * (1.f - turn);
			}
			else if (f.kind == Float::Puff)
			{
				// spreading as it rises
				const float spread = 0.5f + 1.5f * height;
				half = std::min(half * spread, 90.f), tall = std::min(tall * spread, 90.f);
			}
			softSpot(to, head.sx, head.sy, half, tall * LinesPerColumn, glowColour(f.r, f.g, f.b, k));
		}
	}
	if (currentStage == LeakStage)
		addLeaks(list, view, worldX, worldZ, midX, midZ);
	// the fireworks: between the sky and the buildings
	if (currentStage == FireworkStage)
	{
		static constexpr float Every[Fireworks] = { 4.7f, 6.1f, 8.3f };
		static constexpr float Colours[6][3] = { { 1, 0.3f, 0.25f }, { 1, 0.75f, 0.2f }, { 0.35f, 0.6f, 1 }, { 0.4f, 1, 0.5f }, { 1, 0.4f, 0.9f }, { 0.9f, 0.9f, 1 } };
		for (int i = 0; i < Fireworks; i++)
		{
			Firework& fw = fireworks[i];
			const float t = (float)flavourClock / 60.f + (float)i * 1.9f;
			const uint32_t round = (uint32_t)(t / Every[i]);
			fw.age = t - (float)round * Every[i];
			if (fw.age >= FireworkLife)
			{
				fw.live = false;
				continue;
			}
			if (!fw.live || fw.round != round)
			{
				// where it opens: ahead of the camera, a little to one side or the other, as high as the sky shows on the screen now
				fw.live = true, fw.round = round;
				fw.seed = mix32(round * 7919u + (uint32_t)i * 104729u);
				const float flat = std::max(std::sqrt(view.r[6] * view.r[6] + view.r[8] * view.r[8]), 0.001f);
				const float aheadX = view.r[6] / flat, aheadZ = view.r[8] / flat;
				const float turn = ((float)(fw.seed & 1023u) / 1023.f - 0.5f) * 1.1f, c = std::cos(turn), s = std::sin(turn);
				fw.dirX = aheadX * c + aheadZ * s, fw.dirZ = aheadZ * c - aheadX * s;
				fw.x = midX + fw.dirX * FireworkDistance, fw.z = midZ + fw.dirZ * FireworkDistance;
				const float px = fw.x - (float)worldX, pz = fw.z - (float)worldZ;
				const float depth = std::max(view.r[6] * px + view.r[8] * pz + view.t[2], 1000.f);
				const float line = 14.f + (float)((fw.seed >> 9) % 44u);	// (the strip of sky over the arcade)
				const float up = view.r[4] != 0 ? view.r[4] : 1.f;
				fw.y = ((line - view.ofy) * depth / view.h - view.t[1] - view.r[3] * px - view.r[5] * pz) / up;
				// (about 28 of the screen's columns across its middle, whatever the camera's scale)
				const float scale = std::max(std::sqrt(view.r[0] * view.r[0] + view.r[1] * view.r[1] + view.r[2] * view.r[2]) * view.h / depth, 0.0001f);
				fw.radius = 28.f / scale;
				std::memcpy(fw.colour, Colours[(fw.seed >> 18) % 6u], sizeof(fw.colour));
			}
			const float life = 1.f - fw.age / FireworkLife, opened = 1.f - std::exp(-2.6f * fw.age);
			const float red = fw.colour[0] * 255.f, green = fw.colour[1] * 255.f, blue = fw.colour[2] * 255.f;
			NativeCornerOut at;
			if (fw.age < 0.3f && view.projectAt(fw.x - (float)worldX, fw.y, fw.z - (float)worldZ, at))
				softSpot(list.fireworks, at.sx, at.sy, 42.f, 42.f * LinesPerColumn, glowColour(red, green, blue, 0.8f * (1.f - fw.age / 0.3f)));
			constexpr int Sparks = 56;
			for (int k = 0; k < Sparks; k++)
			{
				const uint32_t r = mix32(fw.seed + (uint32_t)k), q = mix32(r);
				const float angle = (float)(r & 0xFFFFu) * (6.2832f / 65536.f), lift = (float)(q & 0xFFFFu) / 32768.f - 1.f;
				const float ring = std::sqrt(std::max(1.f - lift * lift, 0.f));
				const float reach = fw.radius * (0.72f + 0.28f * (float)((r >> 16) & 255u) / 255.f) * opened;
				const float wx = fw.x + std::cos(angle) * ring * reach, wz = fw.z + std::sin(angle) * ring * reach;
				const float wy = fw.y + lift * reach + fw.radius * 0.28f * fw.age * fw.age;	// (falling as it fades)
				if (!view.projectAt(wx - (float)worldX, wy, wz - (float)worldZ, at))
					continue;
				const float twinkle = 0.65f + 0.35f * std::sin(fw.age * 26.f + (float)(q >> 16));
				const float bright = life * std::sqrt(life) * twinkle;
				softSpot(list.fireworks, at.sx, at.sy, 3.4f, 3.4f * LinesPerColumn, glowColour(red, green, blue, std::min(bright * 1.6f, 1.f)));
			}
		}
	}
}

// A searchlight's beam in the air: from its lamp to its pool, soft across, over the fighters
void addBeams(FighterList& list, const View& view, int32_t worldX, int32_t worldZ)
{
	const Mood *mood = currentMood();
	if (mood == nullptr || mood->searchlights == 0)
		return;
	const float midX = ((int32_t)rw(0x800AA190) + (int32_t)rw(0x800ABA1C)) * 0.5f, midZ = ((int32_t)rw(0x800AA198) + (int32_t)rw(0x800ABA24)) * 0.5f;
	for (int k = 0; k < mood->searchlights; k++)
	{
		const Searchlight& l = mood->searchlight[k];
		float lamp[3], aim[3];
		searchlightNow(l, midX, midZ, lamp, aim);
		const float px = aim[0] - (float)worldX, pz = aim[2] - (float)worldZ;
		NativeCornerOut foot, top;
		if (!view.projectAt(px, 0.f, pz, foot))
			continue;
		// (its lamp behind the eye: from where the beam comes into view)
		bool seen = false;
		for (float s = 0.f; s < 0.9f && !seen; s += 0.1f)
			seen = view.projectAt(lamp[0] + (aim[0] - lamp[0]) * s - (float)worldX, lamp[1] * (1.f - s), lamp[2] + (aim[2] - lamp[2]) * s - (float)worldZ, top);
		if (!seen)
			continue;
		const float depth = std::max(view.r[6] * px + view.r[8] * pz + view.t[2], 1000.f);
		const float scale = std::sqrt(view.r[0] * view.r[0] + view.r[1] * view.r[1] + view.r[2] * view.r[2]) * view.h / depth;
		const float dx = foot.sx - top.sx, dy = (foot.sy - top.sy) / LinesPerColumn;
		const float along = std::sqrt(dx * dx + dy * dy);
		if (along < 1.f)
			continue;
		const float nx = -dy / along * l.radius * scale, ny = dx / along * l.radius * scale * LinesPerColumn;
		const uint32_t high = glowColour(l.colour[0] * 255.f, l.colour[1] * 255.f, l.colour[2] * 255.f, 0.1f);
		// nothing along both its sides and at its foot, so no edge shows; narrow at the lamp
		constexpr float Narrow = 0.12f;
		const float tx = nx * Narrow, ty = ny * Narrow;
		list.front.push_back({ { top.sx - tx, top.sx, foot.sx - nx }, { top.sy - ty, top.sy, foot.sy - ny }, { 0, high, 0 } });
		list.front.push_back({ { top.sx, foot.sx, foot.sx - nx }, { top.sy, foot.sy, foot.sy - ny }, { high, 0, 0 } });
		list.front.push_back({ { top.sx, top.sx + tx, foot.sx }, { top.sy, top.sy + ty, foot.sy }, { high, 0, 0 } });
	}
}

// Dawn on Law's stage: the air itself lit gold, added over the backdrop and behind the fighters: deep at the top of the sky, brightest low down where the sun is coming up, gone a little
// way down the floor.
constexpr int DawnStage = 1;
void addDawn(FighterList& list, float horizon)
{
	if (currentStage != DawnStage)
		return;
	constexpr float Left = -8.f, Right = 376.f, Top = -8.f, Bottom = 496.f;
	const float below = std::min(horizon + 150.f, Bottom);
	const auto band = [&](float y0, uint32_t c0, float y1, uint32_t c1) {
		list.mist.push_back({ { Left, Right, Left }, { y0, y0, y1 }, { c0, c0, c1 } });
		list.mist.push_back({ { Right, Right, Left }, { y0, y1, y1 }, { c0, c1, c1 } });
	};
	band(Top, glowColour(58, 22, 0, 1.f), horizon, glowColour(112, 66, 8, 1.f));
	band(horizon, glowColour(112, 66, 8, 1.f), below, 0);
}

void addFlavour(FighterList& list, const View& view, int32_t worldX, int32_t worldZ, float horizon)
{
	addDawn(list, horizon);
	addMist(list, horizon);
	addFloats(list, view, worldX, worldZ);
	addBeams(list, view, worldX, worldZ);
}

int onFloorDraw(CPUState *cpu, uint32_t)
{
	const uint32_t param = cpu->gpr[4];
	const int32_t tileSize = (int32_t)rw(0x8009E780);
	if (tileSize <= 0)
		return 0;
	return floorGrid(cpu, cpu->gpr[31], (int32_t)rw(param) + tileSize / 2, (int32_t)rw(param + 8) + tileSize / 2, FloorLights());
}

// 0x80049848(fighter, fighter): the floor lit from the fighters, for the stages whose kind says so (another game mode's, and
// any other kind: the game's). Whether shadows show is set here first, as the game's: by the camera's pitch.
int onFloorLit(CPUState *cpu, uint32_t)
{
	const int kind = (int16_t)psx_mod_read_half(0x8009E820);
	const int32_t tileSize = (int32_t)rw(0x8009E780);
	if ((kind != 1 && kind != 2) || rw(GameMode) == 8 || tileSize <= 0)
		return 0;
	const uint32_t f0 = cpu->gpr[4], f1 = cpu->gpr[5];
	const int16_t pitchSine = (int16_t)psx_mod_read_half(0x8001E8C4 + ((rw(0x800A85C0) + 239) & 0xFFF) * 2);
	psx_mod_write_word(ShadowsOn, pitchSine > 0 ? 1 : 0);
	if (pitchSine == 0)
		return 1;
	const int32_t x0 = (int32_t)rw(f0 + 3944), z0 = (int32_t)rw(f0 + 3952), x1 = (int32_t)rw(f1 + 3944), z1 = (int32_t)rw(f1 + 3952);
	const int32_t midX = (x0 + x1) / 2, midZ = (z0 + z1) / 2;
	FloorLights lit;
	lit.kind = kind;
	if (kind == 1)
		lit.x[0] = x0, lit.z[0] = z0, lit.x[1] = x1, lit.z[1] = z1;
	else
		lit.x[0] = midX, lit.z[0] = midZ;
	return floorGrid(cpu, cpu->gpr[31], midX + tileSize / 2, midZ + tileSize / 2, lit);
}

// DEADSTATION_TEKKEN3_STAGECHECK (tests): each backdrop vertex here against what the GTE's RTPT makes of it, its saturations and all
bool stageCheck = false;
struct StageCheck { uint32_t vertices = 0, far = 0, clampedView = 0, clampedScreen = 0, clampedRatio = 0, off2 = 0, off8 = 0; float worst = 0, zMost = 0; } stageChecked;
void checkAgainstGte(const CPUState *cpu, int x, int y, int z, const NativeCornerOut& mine)
{
	const uint32_t *g = cpu->gte_ctrl;
	const int64_t r11 = lo16(g[0]), r12 = hi16(g[0]), r13 = lo16(g[1]), r21 = hi16(g[1]), r22 = lo16(g[2]), r23 = hi16(g[2]), r31 = lo16(g[3]), r32 = hi16(g[3]), r33 = lo16(g[4]);
	const int64_t mac1 = ((int64_t)(int32_t)g[5] * 4096 + r11 * x + r12 * y + r13 * z) >> 12;
	const int64_t mac2 = ((int64_t)(int32_t)g[6] * 4096 + r21 * x + r22 * y + r23 * z) >> 12;
	const int64_t mac3 = ((int64_t)(int32_t)g[7] * 4096 + r31 * x + r32 * y + r33 * z) >> 12;
	const int64_t ir1 = std::clamp<int64_t>(mac1, -32768, 32767), ir2 = std::clamp<int64_t>(mac2, -32768, 32767);
	const int64_t sz = std::clamp<int64_t>(mac3, 0, 65535), h = (uint16_t)g[26];
	int64_t ratio = sz > 0 ? (h * 0x20000 / sz + 1) / 2 : 0x1FFFF;
	stageChecked.vertices++;
	if (mac3 <= 0)
	{
		stageChecked.far++;
		return;
	}
	if (ratio > 0x1FFFF)
		ratio = 0x1FFFF, stageChecked.clampedRatio++;
	if (ir1 != mac1 || ir2 != mac2 || sz != mac3)
		stageChecked.clampedView++;
	const int64_t sx = ((int64_t)(int32_t)g[24] + ir1 * ratio) >> 16, sy = ((int64_t)(int32_t)g[25] + ir2 * ratio) >> 16;
	const int64_t csx = std::clamp<int64_t>(sx, -1024, 1023), csy = std::clamp<int64_t>(sy, -1024, 1023);
	if (csx != sx || csy != sy)
		stageChecked.clampedScreen++;
	const float dx = (float)csx - mine.sx, dy = (float)csy - mine.sy, off = std::max(std::fabs(dx), std::fabs(dy));
	stageChecked.worst = std::max(stageChecked.worst, off), stageChecked.zMost = std::max(stageChecked.zMost, (float)mac3);
	stageChecked.off2 += off > 2.f, stageChecked.off8 += off > 8.f;
	static uint32_t tellAt = 0;
	if (frameSerial >= tellAt)
	{
		tellAt = frameSerial + 60;
		std::fprintf(stderr, "deadstation stagecheck: stage %d: %u vertices, behind %u; view clamped %u, ratio %u, screen %u; off by over 2 px %u, over 8 px %u, worst %.1f; farthest %.0f; camera r %d %d %d t %d %d %d h %d\n", currentStage, stageChecked.vertices, stageChecked.far, stageChecked.clampedView, stageChecked.clampedRatio, stageChecked.clampedScreen, stageChecked.off2, stageChecked.off8, (double)stageChecked.worst, (double)stageChecked.zMost, (int)r31, (int)r32, (int)r33, (int)(int32_t)g[5], (int)(int32_t)g[6], (int)(int32_t)g[7], (int)h);
		stageChecked = StageCheck {};
	}
}

// a backdrop cell's vertices: kept here (the game's next call, its polygons, reads them from the scratchpad on the PlayStation)
int onBackdropVertices(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (ra < BackdropLoop || ra >= BackdropLoopEnd)
		return 0;
	View view;
	view.read(cpu);
	const uint32_t p = cpu->gpr[4];
	const int32_t count = (int32_t)cpu->gpr[7];
	if (view.h <= 0 || count < 0 || count > 256)
		return 0;
	for (int32_t i = 0; i < count; i++)
		view.project(p + (uint32_t)i * 8, stageVertices[i]);
	if (stageCheck)
		for (int32_t i = 0; i < count; i++)
			checkAgainstGte(cpu, ramHalf(p + (uint32_t)i * 8), ramHalf(p + (uint32_t)i * 8 + 2), ramHalf(p + (uint32_t)i * 8 + 4), stageVertices[i]);
	backdropReady = true;
	return 1;
}
// 0x8006CC28(polygons, packets, table, &{count, budget}): textured quads and triangles, flat quads; nothing written to
// the packets (the word it leaves for the next cell's packets stays 0: none used)
int onBackdropPolygons(CPUState *cpu, uint32_t)
{
	if (!backdropReady)
		return 0;
	backdropReady = false;
	uint32_t p = cpu->gpr[4];
	const uint32_t state = cpu->gpr[7];
	const int32_t count = (int32_t)psx_mod_read_word(state);
	FighterList& list = nativeList(ramWord(ramWord(OrderingTables) + 4));
	for (int32_t k = 0; k < count; k++)
	{
		const uint32_t w0 = ramWord(p), code = w0 >> 24;
		NativeCornerOut q[4];
		if (code == 0x2C || code == 0x24)
		{
			const uint32_t w1 = ramWord(p + 4), w2 = ramWord(p + 8), w3 = ramWord(p + 12), index = ramWord(p + 16);
			const int corners = code == 0x2C ? 4 : 3;
			bool behind = false;
			for (int j = 0; j < corners; j++)
			{
				q[j] = stageVertices[index >> (8 * j) & 0xFF];
				behind = behind || q[j].z <= 0;
				q[j].rgb = moodColour(w0 & 0xFFFFFF, false);
			}
			q[0].uv = (uint16_t)w1, q[1].uv = (uint16_t)w2, q[2].uv = (uint16_t)w3;
			if (corners == 4)
				q[3].uv = (uint16_t)(w3 >> 16);
			if (!behind)
			{
				// (the amusement park's sky: its own polygons, each stretching a few texels of one colour, CLUT 0x7800; the buildings' arches
				// have theirs painted in)
				const auto span = [](uint32_t a, uint32_t b, uint32_t c) { return std::max({ a, b, c }) - std::min({ a, b, c }); };
				const bool sky = currentStage == FireworkStage && corners == 4 && (w1 >> 16) == 0x7800 && span(w1 & 255, w2 & 255, w3 & 255) <= 8 &&
						span((w1 >> 8) & 255, (w2 >> 8) & 255, (w3 >> 8) & 255) <= 8;
				if (sky)
					quadTriangles(list.sky, q, (uint16_t)(w2 >> 16), (uint16_t)(w1 >> 16), 0);
				else if (corners == 4)
					quadTriangles(list.backdrop, q, (uint16_t)(w2 >> 16), (uint16_t)(w1 >> 16), 0);
				else
					list.backdrop.push_back({ { q[0], q[1], q[2] }, (uint16_t)(w2 >> 16), (uint16_t)(w1 >> 16), 0 });
			}
			p += 20;
		}
		else if (code == 0x28)
		{
			const uint32_t index = ramWord(p + 4);
			bool behind = false;
			for (int j = 0; j < 4; j++)
			{
				q[j] = stageVertices[index >> (8 * j) & 0xFF];
				behind = behind || q[j].z <= 0;
				q[j].rgb = moodColour(w0 & 0xFFFFFF, false), q[j].uv = 0;
			}
			if (!behind)
				quadTriangles(list.backdrop, q, 0, 0, 1);
			p += 8;
		}
		else
			break;
	}
	psx_mod_write_word(state, 0);
	return 1;
}

// ---- The ordering table's compaction at the frame's end (0x80029A28), as host code ----
//
// It unlinks runs of empty entries so the GPU's walk skips them, over both halves of the table, and (outside two game modes)
// swaps three links, which draws the middle of the table after the nearer half. The game's checks the clock (a simulated
// timer read, up to 1024 of them a frame) to give up near the frame's deadline; here the loop takes microseconds and
// always finishes. DEADSTATION_SDK_OFF=ot: the game's own.
constexpr uint32_t CompactOt = 0x80029A28;

// Fight GUI: Hide (Options, at once). The health bars, the timer and the names lie in the ordering table's far entries,
// 0xFC4 down to just above 0xFB4: in a fight's frames those entries are taken out of the walk. What is announced (YOU LOSE,
// REPLAY) lies elsewhere and stays.
bool hideGui = false;
constexpr uint32_t GuiFirstEntry = 0xFC4, GuiPastEntry = 0xFB4;

int onCompactOt(CPUState *cpu, uint32_t)
{
	const uint32_t ot = ramWord(OrderingTables);
	const uint32_t tag = ramWord(ot + 16), length = ramWord(ot);
	if (hideGui)
	{
		// (read only: nativeList begins a new frame's lists, and would lose the one just built)
		const uint32_t table = ramWord(ot + 4);
		for (const FighterList& list : fighterLists)
			if (list.ot == table && list.serial + 2 >= frameSerial && !list.tris.empty())
			{
				const uint32_t first = table + GuiFirstEntry;
				psx_mod_write_word(first, (ramWord(first) & 0xFF000000u) | ((table + GuiPastEntry) & 0xFFFFFFu));
				break;
			}
	}
	if (rw(GameMode) - 7u > 1u)
	{
		const uint32_t a = tag - 600, b = tag - 0xFF8, c = tag - 0x7FC;
		const uint32_t va = ramWord(a), vb = ramWord(b), vc = ramWord(c);
		psx_mod_write_word(a, (va & 0xFF000000u) | (vc & 0xFFFFFFu));
		psx_mod_write_word(b, (vb & 0xFF000000u) | (va & 0xFFFFFFu));
		psx_mod_write_word(c, (vc & 0xFF000000u) | (vb & 0xFFFFFFu));
	}
	const int entries = 1 << ((length - 1) & 31);
	for (uint32_t start : { tag - 0x800, tag })
	{
		uint32_t open = 0, at = start;
		for (int i = 0; i < entries; i++, at -= 4)
		{
			const uint32_t word = ramWord(at);
			if ((word & 0xFFFFFFu) == ((at - 4) & 0xFFFFFFu))
			{
				if (open == 0)
					open = at;
			}
			else if (open != 0)
			{
				// (to this entry and not past it, as the game has it, to its first packet: the walk has to come to it, or what's in the fighters' order that lies under them, or
				// over them, can't be told apart: the fighters' polygons aren't in the table)
				psx_mod_write_word(open, (ramWord(open) & 0xFF000000u) | (at & 0xFFFFFFu));
				open = 0;
			}
		}
	}
	(void)cpu;
	return 1;
}

// the vertical blank (a plugin has one callback: PSXRecomp refuses a second, silently)
bool nativePads = false;
void onEveryVBlank()
{
	if (stats)
		onVBlank();
	if (nativePads)
		onPadVBlank();
	if (nativeDisc)
		onDriveVBlank();
}

struct Registration
{
	Registration()
	{
		stats = std::getenv("DEADSTATION_TEKKEN3_STATS") != nullptr;
		psx_mod_register_function_filter_plugin(Id, ShadowProjection, onShadowProjection);
		psx_mod_register_function_entry_plugin(Id, AfterShadow, onAfterShadow);
		if (const char *level = std::getenv("DEADSTATION_TEKKEN3_SPARKS"))
			sparkLevel = std::clamp(std::atoi(level), 0, 10);
		if (const char *test = std::getenv("DEADSTATION_TEKKEN3_TESTLIGHTS"))
			testLights = std::atoi(test);
		testStrike = std::getenv("DEADSTATION_TEKKEN3_TESTSTRIKE") != nullptr;
		stageCheck = std::getenv("DEADSTATION_TEKKEN3_STAGECHECK") != nullptr;
		noMoods = std::getenv("DEADSTATION_SDK_OFF") != nullptr && std::strstr(std::getenv("DEADSTATION_SDK_OFF"), "moods") != nullptr;
		if (const char *stage = std::getenv("DEADSTATION_TEKKEN3_STAGE"))
			forceStage = std::atoi(stage);
		if (const char *gui = std::getenv("DEADSTATION_TEKKEN3_GUI"))
			hideGui = std::strcmp(gui, "hide") == 0;
		unlockAll = std::getenv("DEADSTATION_TEKKEN3_UNLOCK") != nullptr;
		// Spark Lighting and Fight GUI: applied as they change in Options, the game going on
		deadstation_live_settings([](const char *key, const char *value) {
			if (std::strcmp(key, "sparks") == 0)
			{
				if (value != nullptr)
					sparkLevel = std::clamp(std::atoi(value), 0, 10);
				return true;
			}
			if (std::strcmp(key, "gui") == 0)
			{
				if (value != nullptr)
					hideGui = std::strcmp(value, "hide") == 0;
				return true;
			}
			if (std::strcmp(key, "unlock") == 0)
			{
				if (value != nullptr)
					unlockAll = std::strcmp(value, "on") == 0;
				return true;
			}
			return false;
		});
		sceneCapture = std::getenv("DEADSTATION_TEKKEN3_SCENE") != nullptr || std::getenv("DEADSTATION_TEKKEN3_DUMP") != nullptr;
		psx_mod_register_function_filter_plugin(Id, PartDraw, onPartDraw);
		const char *sdkOff = std::getenv("DEADSTATION_SDK_OFF");
		const bool nativeSdk = std::getenv("DEADSTATION_SDK") == nullptr || std::strcmp(std::getenv("DEADSTATION_SDK"), "emulated") != 0;
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "pad") == nullptr))
		{
			psx_mod_register_function_filter_plugin(Id, PadInitDirect, onPadInitDirect);
			for (uint32_t f : { PadStartComEntry, PadStartComMain, PadStopComEntry, PadStopComMain, PadSetAct, PadSetActAlign,
					PadSetMainMode, PadChkVsync })
				psx_mod_register_function_filter_plugin(Id, f, onPadNothing);
			for (uint32_t f : { PadInfoAct, PadInfoComb })
				psx_mod_register_function_filter_plugin(Id, f, onPadZero);
			psx_mod_register_function_filter_plugin(Id, PadGetState, onPadGetState);
			psx_mod_register_function_filter_plugin(Id, PadInfoMode, onPadInfoMode);
			nativePads = true;
		}
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "framewait") == nullptr))
			psx_mod_register_function_filter_plugin(Id, Stir, onStir);
		psx_mod_register_function_filter_plugin(Id, FrameWait, onFrameEnd);
		nativeFighters = nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "fighters") == nullptr);
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "gtemath") == nullptr))
		{
			psx_mod_register_function_filter_plugin(Id, 0x80081D68, onMulMatrix0);
			psx_mod_register_function_filter_plugin(Id, 0x8003B5C4, onCompMatrix);
			psx_mod_register_function_filter_plugin(Id, 0x8003B71C, onCompMatrixLong);
			psx_mod_register_function_filter_plugin(Id, 0x80081CD8, onSquareRoot0);
		}
		if (nativeFighters && (sdkOff == nullptr || std::strstr(sdkOff, "stage") == nullptr))
		{
			psx_mod_register_function_filter_plugin(Id, FloorDraw, onFloorDraw);
			psx_mod_register_function_filter_plugin(Id, FloorLit, onFloorLit);
			psx_mod_register_function_filter_plugin(Id, BackdropVertices, onBackdropVertices);
			psx_mod_register_function_filter_plugin(Id, BackdropPolygons, onBackdropPolygons);
		}
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "vsyncquery") == nullptr))
			psx_mod_register_function_filter_plugin(Id, VSyncFunction, onVSync);
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "disc") == nullptr))
		{
			nativeDisc = true;
			psx_mod_register_function_filter_plugin(Id, 0x8008EBD8, onDsInit);
			psx_mod_register_function_filter_plugin(Id, 0x8008ED1C, onDsReset);	// DsReset
			psx_mod_register_function_filter_plugin(Id, 0x8008EE28, onDsReset);	// DsClose
			psx_mod_register_function_filter_plugin(Id, 0x8008EE48, onDsCommand);
			psx_mod_register_function_filter_plugin(Id, 0x8008F08C, onDsPacket);
			psx_mod_register_function_filter_plugin(Id, 0x8008F3DC, onDsSync);
			psx_mod_register_function_filter_plugin(Id, 0x80090AE8, onDsSync);		// DsDataSync
			psx_mod_register_function_filter_plugin(Id, 0x8008F5CC, onDsReady);
			psx_mod_register_function_filter_plugin(Id, 0x8008F660, onDsReturnOne);	// DsFlush
			psx_mod_register_function_filter_plugin(Id, 0x8008F6F0, onDsReturnOne);	// DsSystemStatus
			psx_mod_register_function_filter_plugin(Id, 0x8008F73C, onDsReturnZero);	// DsQueueLen
			psx_mod_register_function_filter_plugin(Id, 0x8008F74C, onDsStatus);
			psx_mod_register_function_filter_plugin(Id, 0x8008F76C, onDsReturnZero);	// DsShellOpen
			psx_mod_register_function_filter_plugin(Id, 0x80090A88, onDsMix);
			psx_mod_register_function_filter_plugin(Id, 0x80090AA8, onDsReturnOne);	// DsGetSector
			psx_mod_register_function_filter_plugin(Id, 0x80090AC8, onDsReturnOne);	// DsGetSector2
			for (uint32_t f : { 0x80090CF8u, 0x80090D20u, 0x80090D88u })				// DsControlF, DsControl, DsControlB
				psx_mod_register_function_filter_plugin(Id, f, onDsControl);
			psx_mod_register_function_filter_plugin(Id, 0x80090DF8, onDsGetToc);
			psx_mod_register_function_filter_plugin(Id, 0x80091458, onDsRead2);
			psx_mod_register_function_filter_plugin(Id, StGetNextFunction, onStGetNext);
			psx_mod_register_function_filter_plugin(Id, BnsReadsPending, onBnsReadsPending);
			psx_mod_register_function_filter_plugin(Id, 0x80091558, onDsSearchFile);
			psx_mod_register_function_filter_plugin(Id, 0x80091F38, onDsReturnOne);	// DsStartReadySystem
			psx_mod_register_function_filter_plugin(Id, 0x80091FBC, onDsReturnOne);	// DsEndReadySystem
			psx_mod_register_function_filter_plugin(Id, 0x80092020, onDsReturnOne);	// DsReadySystemMode
			psx_mod_register_function_filter_plugin(Id, 0x8009268C, onDsSyncCallback);
			psx_mod_register_function_filter_plugin(Id, 0x800926A0, onDsReadyCallback);
			psx_mod_register_function_filter_plugin(Id, 0x800926B4, onDsDataCallback);
		}
		if (nativeSdk && (sdkOff == nullptr || std::strstr(sdkOff, "ot") == nullptr))
			psx_mod_register_function_filter_plugin(Id, CompactOt, onCompactOt);
		if (nativeFighters)
			walkLog = std::getenv("DEADSTATION_TEKKEN3_WALK") != nullptr;
			psx_gpu_list_node_hook = onListNode;
		// the PlayStation SDK as host code (sdk/brick/psx_sdk.h): Tekken 3's copies (PsyQ 4.1/4.2, signature-matched)
		psx_mod_register_vblank_plugin(Id, onEveryVBlank);
		PsxSdkAddresses sdk;
		sdk.drawOTag = 0x8007C998;		// LIBGPU SYS.OBJ
		sdk.loadImage = 0x8007C650;
		sdk.moveImage = 0x8007C710;
		sdk.gpuSize = 0x80098BBC;
		sdk.vsyncWait = 0x80085B20;		// LIBETC VSYNC.OBJ's v_wait
		sdk.vsyncCounter = 0x8009AC68;	// its vertical blank count
		const bool native = std::getenv("DEADSTATION_SDK") == nullptr || std::strcmp(std::getenv("DEADSTATION_SDK"), "emulated") != 0;
		if (native)
			psx_sdk_install(Id, sdk);
		const char *off = std::getenv("DEADSTATION_SDK_OFF");
		if (native && (off == nullptr || std::strstr(off, "bns") == nullptr))
			psx_mod_register_function_filter_plugin(Id, BnsStartReads, onBnsStartReads);
	}
} registration;
}

#ifndef __aarch64__
// The PC build (tests): what a host's present would get (the Brick's is sdk/brick/host.cpp), logged
// whenever its size changes; the runtime's own present then goes on.
extern "C" int psx_host_present(const uint32_t *pixels, int width, int height)
{
	// DEADSTATION_TEKKEN3_DUMP=DIR: every 120th frame presented, its pixels (PPM) and the 3D layer published
	// with it (triangles: x0 y0 z0 x1 y1 z1 x2 y2 z2 per line, the game's pixels)
	static const char *dump = std::getenv("DEADSTATION_TEKKEN3_DUMP");
	static unsigned presented = 0;
	static unsigned withLights = 0;
	deadstation::scene::Frame f;
	if (dump != nullptr)
		deadstation::scene::take(f);
	// (also the first frames whose layer has flashes, every 3rd)
	if (dump != nullptr && (++presented % 120 == 0 || (!f.lights.empty() && withLights < 12 && presented % 3 == 0 && ++withLights)))
	{
		char path[512];
		std::snprintf(path, sizeof(path), "%s/frame-%05u.ppm", dump, presented);
		if (FILE *out = std::fopen(path, "wb"))
		{
			std::fprintf(out, "P6\n%d %d\n255\n", width, height);
			for (int i = 0; i < width * height; i++)
			{
				const uint32_t c = pixels[i];
				const unsigned char rgb[3] = { (unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c };
				std::fwrite(rgb, 1, 3, out);
			}
			std::fclose(out);
		}
		std::snprintf(path, sizeof(path), "%s/frame-%05u.scene", dump, presented);
		if (FILE *out = std::fopen(path, "w"))
		{
			std::fprintf(out, "# size %g %g\n# sun %g %g %g\n# ambient %g\n# serial %llu\n", f.width, f.height, f.sun[0], f.sun[1], f.sun[2],
					f.ambient, (unsigned long long)f.serial);
			for (const deadstation::scene::Light& l : f.lights)
				std::fprintf(out, "# light %g %g %g %g %g %g\n", l.x, l.y, l.z, l.r, l.g, l.b);
			for (const Corner& c : f.floor)
				std::fprintf(out, "# floor %g %g %g %g %g %g %g %g\n", c.sx, c.sy, c.x, c.y, c.z, c.nx, c.ny, c.nz);
			for (const Corner& c : f.corners)
				std::fprintf(out, "%g %g %g %g %g %g %g %g\n", c.sx, c.sy, c.x, c.y, c.z, c.nx, c.ny, c.nz);
			std::fclose(out);
		}
	}
	static int w, h;
	if (stats && (width != w || height != h))
	{
		std::fprintf(stderr, "deadstation tekken3: frames %d x %d\n", width, height);
		w = width;
		h = height;
	}
	return 0;
}
#endif

// The fighters' 3D layer, drawn by the presenter: their shadows on the floor are always on
extern "C" void deadstation_module_start(void)
{
	setenv("DEADSTATION_TEKKEN3_SCENE", "1", 0);
}

// The settings, as the plugin and the presenter read them. Resolution 2: the game's frames at the panel's own size, VRAM at twice the
// PlayStation's. Spark Lighting, Fight GUI and Unlock Everything are applied as they change too (deadstation_live_settings).
extern "C" void deadstation_module_setting(const char *key, const char *value)
{
	if (std::strcmp(key, "scale") == 0)
	{
		if (std::strcmp(value, "1") != 0)
		{
			setenv("DEADSTATION_RESOLUTION", "1024", 0);
			setenv("PSX_INTERNAL_RESOLUTION", "480", 0);
		}
	}
	else if (std::strcmp(key, "unlock") == 0)
	{
		if (std::strcmp(value, "on") == 0)
			setenv("DEADSTATION_TEKKEN3_UNLOCK", "1", 0);
	}
	else if (std::strcmp(key, "gui") == 0)
	{
		if (std::strcmp(value, "hide") == 0)
			setenv("DEADSTATION_TEKKEN3_GUI", "hide", 0);
	}
	else if (std::strcmp(key, "sparks") == 0)
		setenv("DEADSTATION_TEKKEN3_SPARKS", value, 0);
}
