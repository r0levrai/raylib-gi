/*******************************************************************************************
*
* Raylib Global Illumination, adapted to c++/raylib/opengl from the excellent 
* js/threejs/webgl interractive article at https://jason.today/gi
*
* With help from Gemini 2.5 pro to bootstrap the plumbing
*
* Core Pipeline:
* 1. Drawing Surface: User draws onto a texture using the mouse.
* 2. Seed Texture: Creates initial data for JFA (stores UVs where drawn).
* 3. Jump Flood Algorithm (JFA): Calculates nearest seed UV for each pixel.
* 4. Distance Field: Calculates distance to the nearest seed UV.
* 5. GI Raymarch: Performs raymarching using the distance field for acceleration.
*
********************************************************************************************/

#include "raylib.h"
#include "raymath.h" // For Vector2DistanceSqr
#define RAYGUI_IMPLEMENTATION
#include "raygui.h" // For checkboxes, buttons and sliders

#include <cmath>     // For log2, pow, ceil
#include <vector>
#include <string>

//----------------------------------------------------------------------------------
// Defines and Macros
//----------------------------------------------------------------------------------
#define MAX_JFA_PASSES 12 // Should be ceil(log2(max(width, height)))
#define INITIAL_RAYMARCH_STEPS 32
#define MAX_RAYMARCH_STEPS 48 // limit in the gui
#define TAU (2.0f * PI)

//----------------------------------------------------------------------------------
// Shader Code (GLSL 330)
//----------------------------------------------------------------------------------

// --- Common Vertex Shader ---
// reverse the Y axis (raylib vs opengl coordinates), so each texture can naturally
// sample each other
const char *vertexShaderSrc = R"(
#version 330 core
layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec2 vertexTexCoord;

out vec2 fragTexCoord;

uniform mat4 mvp; // Raylib default MVP uniform

void main()
{
    // Flip texture vertically
    fragTexCoord = vec2(vertexTexCoord.x, 1.0 - vertexTexCoord.y);

    // Keep output position the same
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)";

// --- Drawing Shader (SDF Line for GPU drawing) ---
const char *drawFragShaderSrc = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 fragColor;

uniform sampler2D inputTexture; // Previous frame's drawing
uniform vec3 drawColor;
uniform vec2 fromPos; // Mouse position from (pixels)
uniform vec2 toPos;   // Mouse position to (pixels)
uniform float radiusSquared;
uniform vec2 resolution;
uniform bool isDrawing;

// Calculates squared distance from point p to line segment (from, to)
float sdfLineSquared(vec2 p, vec2 from, vec2 to) {
    vec2 toStart = p - from;
    vec2 line = to - from;
    float lineLengthSquared = dot(line, line);
    float t = clamp(dot(toStart, line) / lineLengthSquared, 0.0, 1.0);
    vec2 closestVector = toStart - line * t;
    return dot(closestVector, closestVector);
}

void main()
{
    //vec2 fragTexCoordYFlip = vec2(fragTexCoord.x, resolution.y - fragTexCoord.y);
    // \-> flipped in the vertex shader now, so each texture can naturally index each other
    vec4 current = texture(inputTexture, fragTexCoord);
    if (isDrawing) {
        vec2 pixelCoord = fragTexCoord * resolution;
        if (sdfLineSquared(pixelCoord, fromPos, toPos) <= radiusSquared) {
            current = vec4(drawColor, 1.0);
        }
    }
    fragColor = current;
}
)";


// --- Seed Shader ---
const char *seedFragShaderSrc = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 fragColor;

uniform sampler2D surfaceTexture; // The drawing texture

void main()
{
    // Get alpha from the drawing texture
    float alpha = texture(surfaceTexture, fragTexCoord).a;

    // Store UV coordinates in RG channels if alpha > 0, otherwise store 0
    // Store 1.0 in A channel to mark it as processed/valid seed info
    // We store alpha in the blue channel for potential future use (unused here)
    fragColor = vec4(fragTexCoord * alpha, alpha, 1.0);
}
)";

// --- JFA Shader ---
const char *jfaFragShaderSrc = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 fragColor; // Output: vec4(nearest_seed_uv.x, nearest_seed_uv.y, original_alpha_at_seed, 1.0)

uniform sampler2D inputTexture; // Previous JFA pass or Seed texture
uniform vec2 oneOverResolution; // 1.0 / resolution
uniform float stepOffset;       // Current step offset (power of 2)

void main()
{
    vec4 nearestSeedData = vec4(-2.0); // Stores vec2(uv), alpha, 1.0. Initialize out of bounds.
    float nearestDistSq = 999999.9;

    // Sample 3x3 neighborhood including center
    for (float y = -1.0; y <= 1.0; y += 1.0) {
        for (float x = -1.0; x <= 1.0; x += 1.0) {
            vec2 sampleUV = fragTexCoord + vec2(x, y) * stepOffset * oneOverResolution;
            sampleUV = clamp(sampleUV, 0.0, 1.0); // Clamp to avoid sampling outside bounds

            vec4 sampleData = texture(inputTexture, sampleUV);
            vec2 sampleSeedUV = sampleData.xy; // UV stored in RG

            // Check if the sampled pixel contains valid seed UV data.
            // Valid seeds have alpha > 0 in the seed texture pass, stored in Z.
            // Or check if the UV itself is not zero (simplest check if alpha isn't stored/used).
            // Using the UV check here as per the previous logic.
            if (sampleSeedUV.x > 0.0 || sampleSeedUV.y > 0.0) {
                vec2 diff = sampleSeedUV - fragTexCoord;
                float distSq = dot(diff, diff);

                if (distSq < nearestDistSq) {
                    nearestDistSq = distSq;
                    nearestSeedData = sampleData; // Store the data from the nearest seed found so far
                }
            }
        }
    }

    // If this pixel itself contained a seed, ensure it's considered.
    // Check its own UV > 0 and if it's closer than any neighbor found.
    vec4 currentPixelData = texture(inputTexture, fragTexCoord);
    if ((currentPixelData.x > 0.0 || currentPixelData.y > 0.0)) {
         vec2 diff = currentPixelData.xy - fragTexCoord;
         float distSq = dot(diff, diff);
         if (distSq < nearestDistSq) {
            nearestSeedData = currentPixelData;
         }
    }

    // If no valid seed was found in the neighborhood (or self), nearestSeedData remains vec4(-2.0)
    fragColor = nearestSeedData;
}
)";


// --- Distance Field Shader ---
const char *distanceFieldFragShaderSrc = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 fragColor;

uniform sampler2D jfaTexture; // Texture containing nearest seed UVs in RG channels

void main()
{
    vec4 jfaData = texture(jfaTexture, fragTexCoord);
    vec2 nearestSeedUV = jfaData.xy;

    float dist = 1.0; // Default to max distance if no valid seed found
    // Check if a valid nearest seed was found (not the initial -2.0 value)
    if (nearestSeedUV.x >= 0.0) { // Check x is sufficient as -2.0 is invalid
         dist = distance(fragTexCoord, nearestSeedUV);
    }

    // Clamp distance (optional, but good practice)
    dist = clamp(dist, 0.0, 1.0);

    // Output distance in R channel, GBA can be 0 or used for other data
    fragColor = vec4(dist, dist, dist, 1.0); // Visualize as grayscale
}
)";

// --- Global Illumination Raymarch Shader ---
const char *giFragShaderSrc = R"(
#version 330 core
in vec2 fragTexCoord;
out vec4 fragColor;

uniform sampler2D sceneTexture;      // Original drawing
uniform sampler2D distanceTexture;   // Precomputed distance field (distance in R channel)
uniform sampler2D lastFrameTexture;  // For temporal accumulation
uniform vec2 oneOverResolution;
uniform int rayCount;
uniform int maxSteps;
uniform bool showNoise;
uniform bool showGrain;           // Angular noise
uniform bool useTemporalAccum;
uniform bool enableSun;
uniform float time;               // For temporal noise seed
uniform float sunAngle;           // Radians

const float PI = 3.14159265;
const float TAU = 2.0 * PI;
const float EPS = 0.001; // Small epsilon for hitting surface

// Basic sky/sun simulation colors
const vec3 skyColor = vec3(0.02, 0.08, 0.2);
const vec3 sunColor = vec3(0.95, 0.95, 0.9);

// Simple pseudo-random number generator
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Simple sun/sky calculation based on angle
vec3 sunAndSky(float rayAngle) {
    float angleToSun = mod(rayAngle - sunAngle + TAU, TAU); // Ensure positive modulo
    // Adjust falloff width (e.g., PI*0.5 makes sun cover quarter circle smoothly)
    float sunIntensity = smoothstep(PI * 0.5, 0.0, angleToSun);
    return sunColor * sunIntensity + skyColor;
}

bool outOfBounds(vec2 uv) {
  return uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0;
}

void main()
{
    vec2 uv = fragTexCoord;
    vec4 baseColor = texture(sceneTexture, uv); // Color/alpha of the current pixel in the drawing

    vec4 accumulatedRadiance = vec4(0.0);

    // If the current pixel is part of the drawing (occluder/emitter), don't calculate GI for it.
    if (baseColor.a > 0.1) {
        accumulatedRadiance = baseColor;
    } else {
        // Pixel is empty space, calculate incoming radiance
        float oneOverRayCount = 1.0 / float(rayCount);
        float angleStepSize = TAU * oneOverRayCount;

        // Temporal/spatial noise offset
        float noiseCoef = useTemporalAccum ? time : 0.0;
        float noiseOffset = showNoise ? rand(uv + noiseCoef) : 0.0;

        for(int i = 0; i < rayCount; i++) {
            // Calculate base angle for this ray, distribute with noise
            float angle = angleStepSize * (float(i) + noiseOffset);
            // Add optional grain (per-ray angular noise)
             if (showGrain) angle += rand(uv + vec2(i, noiseCoef)) * angleStepSize * 0.5; // Smaller grain perturbation

            vec2 rayDirection = vec2(cos(angle), -sin(angle)); // Use -sin for Y-down UV typical in textures
            vec2 sampleUv = uv;
            vec4 radianceDelta = vec4(0.0);
            bool hitSurface = false;

            for (int step = 0; step < maxSteps; step++) {
                // Check bounds before sampling distance texture
                if (outOfBounds(sampleUv)) break;

                // Sample distance field: distance to nearest surface in UV space
                float dist = texture(distanceTexture, sampleUv).r;

                // Check if we are very close to a surface
                // Scale EPS check by pixel size for robustness at different resolutions
                if (dist < EPS * length(oneOverResolution)) {
                    // We hit a surface. Accumulate its color.
                    // Sample slightly offset *towards* the surface to ensure hitting it, avoid sampling self edge
                    vec2 hitUv = sampleUv - rayDirection * oneOverResolution * 0.5;
                    hitUv = clamp(hitUv, 0.0, 1.0);
                    radianceDelta = texture(sceneTexture, hitUv);
                    hitSurface = true;
                    break; // Stop marching this ray
                }

                // Advance ray position by the safe distance
                // Ensure minimum step to avoid getting stuck, scaled by resolution
                float stepDist = max(dist, length(oneOverResolution) * 0.5);
                sampleUv += rayDirection * stepDist; // Step in UV space
            }

            // If the ray didn't hit any surface and sun is enabled, add sky/sun contribution
            if (!hitSurface && enableSun) {
                radianceDelta = vec4(sunAndSky(angle), 1.0);
            }

            accumulatedRadiance += radianceDelta;
        }
         // Average the radiance collected from all rays
        accumulatedRadiance *= oneOverRayCount;
    }

    // Combine base color (if pixel was drawn) with calculated radiance
    // Use max to ensure drawn pixels retain their color if brighter than incoming light
    vec4 finalColor = vec4(max(baseColor.rgb, accumulatedRadiance.rgb), 1.0); // Ensure alpha is 1 for final output

    // Temporal Accumulation
    if (useTemporalAccum && time > 0.0) { // Check time to avoid using uninitialized texture
        vec4 prevFrameColor = texture(lastFrameTexture, uv);
        // Mix current frame with previous frame. Adjust mix factor (0.9 = 90% previous)
        fragColor = mix(finalColor, prevFrameColor, 0.9);
    } else {
        fragColor = finalColor;
    }
}
)";


//----------------------------------------------------------------------------------
// Types and Structures Definition
//----------------------------------------------------------------------------------
typedef struct AppState {
    // Textures & Render Targets
    RenderTexture2D drawSurface;
    RenderTexture2D seedTexture;        // Stores initial UVs for JFA
    RenderTexture2D jfaTextureA;        // Ping-pong buffers for JFA
    RenderTexture2D jfaTextureB;
    RenderTexture2D distanceFieldTexture; // Stores distance to nearest surface
    RenderTexture2D giResultA;          // Ping-pong buffers for GI result / temporal accum
    RenderTexture2D giResultB;

    // Shaders
    Shader drawShader;
    Shader seedShader;
    Shader jfaShader;
    Shader distanceFieldShader;
    Shader giShader;

    // Shader Locations (cached)
    int draw_loc_resolution;
    int draw_loc_fromPos;
    int draw_loc_toPos;
    int draw_loc_radiusSquared;
    int draw_loc_drawColor;
    int draw_loc_isDrawing;
    int draw_loc_inputTexture; // Location for the input texture uniform

    int seed_loc_surfaceTexture;

    int jfa_loc_inputTexture;
    int jfa_loc_oneOverResolution;
    int jfa_loc_stepOffset;

    int df_loc_jfaTexture;
    int df_loc_resolution; // Optional for DF shader

    int gi_loc_sceneTexture;
    int gi_loc_distanceTexture;
    int gi_loc_lastFrameTexture;
    int gi_loc_oneOverResolution;
    int gi_loc_rayCount;
    int gi_loc_maxSteps;
    int gi_loc_showNoise;
    int gi_loc_showGrain;
    int gi_loc_useTemporalAccum;
    int gi_loc_enableSun;
    int gi_loc_time;
    int gi_loc_sunAngle;

    // Drawing State
    Vector2 lastMousePos;
    Vector2 currentMousePos;
    Color drawColor;
    float drawRadius;
    bool isDrawing;
    bool mouseMovedSinceClick;
    int debugView;


    // GI Parameters / UI State
    bool showNoise;
    bool showGrain;
    bool useTemporalAccum;
    bool enableSun;
    float sunAngle; // In radians
    int maxSteps;
    float maxStepsFloat; // Temporary float for GuiSliderBar
    int rayCount;
    float time; // For temporal accumulation noise

    // Control which buffer is current input/output
    bool jfaPingPong;
    bool giPingPong;

    // JFA calculation state
    int jfaPassesNeeded;

} AppState;

//----------------------------------------------------------------------------------
// Module Functions Declaration
//----------------------------------------------------------------------------------
void InitApp(AppState *state, int width, int height);
void UpdateApp(AppState *state);
void DrawApp(AppState *state, int width, int height);
void DeInitApp(AppState *state);
void DrawQuad(AppState *state); // Draw final full screen quad


//----------------------------------------------------------------------------------
// Main Entry Point
//----------------------------------------------------------------------------------
int main(void)
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1920; // Adjust as needed
    const int screenHeight = 1080;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT); // Enable VSYNC
    InitWindow(screenWidth, screenHeight, "Raylib Global Illumination Demo");
    SetTargetFPS(0); // Set desired frame rate (0 = unlocked, use VSYNC)

    AppState state = {0};
    InitApp(&state, screenWidth, screenHeight);
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose()) // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        UpdateApp(&state);
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        DrawApp(&state, GetScreenWidth(), GetScreenHeight()); // Use current screen size
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    DeInitApp(&state);
    CloseWindow(); // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}

//----------------------------------------------------------------------------------
// Module Functions Definition
//----------------------------------------------------------------------------------

void DrawQuad(AppState *state) {
     DrawTexture(state->seedTexture.texture, 0, 0, WHITE);
}

Shader defaultShader = LoadShader(0, 0);
bool CheckShader(Shader shader)
{
    return !(shader.id == 0 || shader.id == defaultShader.id || !IsShaderValid(shader));
}

void InitApp(AppState *state, int width, int height) {
    // --- Initialize State ---
    state->lastMousePos = GetMousePosition();
    state->currentMousePos = state->lastMousePos;
    state->drawColor = MAROON; // Initial drawing color
    state->drawRadius = 6.0f;
    state->isDrawing = false;
    state->mouseMovedSinceClick = false;
    state->debugView = 0;

    state->showNoise = true;
    state->showGrain = true;
    state->useTemporalAccum = false;
    state->enableSun = true;
    state->sunAngle = 4.2f; // Radians
    state->maxSteps = INITIAL_RAYMARCH_STEPS;
    state->maxStepsFloat = (float)state->maxSteps; // Init float version for slider
    state->rayCount = 32; // Number of rays for GI
    state->time = 0.0f;

    state->jfaPingPong = false;
    state->giPingPong = false;

    state->jfaPassesNeeded = (int)ceil(log2((float)fmax(width, height)));
    if (state->jfaPassesNeeded > MAX_JFA_PASSES) state->jfaPassesNeeded = MAX_JFA_PASSES;


    // --- Load Shaders ---
    state->drawShader = LoadShaderFromMemory(vertexShaderSrc, drawFragShaderSrc);
    state->seedShader = LoadShaderFromMemory(vertexShaderSrc, seedFragShaderSrc);
    state->jfaShader = LoadShaderFromMemory(vertexShaderSrc, jfaFragShaderSrc);
    state->distanceFieldShader = LoadShaderFromMemory(vertexShaderSrc, distanceFieldFragShaderSrc);
    state->giShader = LoadShaderFromMemory(vertexShaderSrc, giFragShaderSrc);
    
    // --- Abort on load failure so we don't miss a shader compilation error ---
    if (!CheckShader(state->drawShader) ||
        !CheckShader(state->seedShader) ||
        !CheckShader(state->jfaShader) ||
        !CheckShader(state->distanceFieldShader) || 
        !CheckShader(state->giShader)
    ) {
        TraceLog(LOG_ERROR, "Failed to find/compile a shader! Look above for the logs.");
        std::abort();
    }

    Shader defaultShader = LoadShader(0, 0);
    if (state->drawShader.id != 0 && state->drawShader.id == defaultShader.id || !IsShaderValid(state->drawShader)) {}

    // --- Get Shader Locations ---
    // Draw Shader
    state->draw_loc_resolution = GetShaderLocation(state->drawShader, "resolution");
    state->draw_loc_fromPos = GetShaderLocation(state->drawShader, "fromPos");
    state->draw_loc_toPos = GetShaderLocation(state->drawShader, "toPos");
    state->draw_loc_radiusSquared = GetShaderLocation(state->drawShader, "radiusSquared");
    state->draw_loc_drawColor = GetShaderLocation(state->drawShader, "drawColor");
    state->draw_loc_isDrawing = GetShaderLocation(state->drawShader, "isDrawing");
    state->draw_loc_inputTexture = GetShaderLocation(state->drawShader, "inputTexture");

    // Seed Shader
    state->seed_loc_surfaceTexture = GetShaderLocation(state->seedShader, "surfaceTexture");

    // JFA Shader
    state->jfa_loc_inputTexture = GetShaderLocation(state->jfaShader, "inputTexture");
    state->jfa_loc_oneOverResolution = GetShaderLocation(state->jfaShader, "oneOverResolution");
    state->jfa_loc_stepOffset = GetShaderLocation(state->jfaShader, "stepOffset");

    // Distance Field Shader
    state->df_loc_jfaTexture = GetShaderLocation(state->distanceFieldShader, "jfaTexture");
    state->df_loc_resolution = GetShaderLocation(state->distanceFieldShader, "resolution"); // Keep if shader uses it

    // GI Shader
    state->gi_loc_sceneTexture = GetShaderLocation(state->giShader, "sceneTexture");
    state->gi_loc_distanceTexture = GetShaderLocation(state->giShader, "distanceTexture");
    state->gi_loc_lastFrameTexture = GetShaderLocation(state->giShader, "lastFrameTexture");
    state->gi_loc_oneOverResolution = GetShaderLocation(state->giShader, "oneOverResolution");
    state->gi_loc_rayCount = GetShaderLocation(state->giShader, "rayCount");
    state->gi_loc_maxSteps = GetShaderLocation(state->giShader, "maxSteps");
    state->gi_loc_showNoise = GetShaderLocation(state->giShader, "showNoise");
    state->gi_loc_showGrain = GetShaderLocation(state->giShader, "showGrain");
    state->gi_loc_useTemporalAccum = GetShaderLocation(state->giShader, "useTemporalAccum");
    state->gi_loc_enableSun = GetShaderLocation(state->giShader, "enableSun");
    state->gi_loc_time = GetShaderLocation(state->giShader, "time");
    state->gi_loc_sunAngle = GetShaderLocation(state->giShader, "sunAngle");


    // --- Create Render Targets ---
    // Use standard 8-bit format for drawing surface and final result
    state->drawSurface = LoadRenderTexture(width, height);
    SetTextureFilter(state->drawSurface.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(state->drawSurface.texture, TEXTURE_WRAP_CLAMP);

    state->giResultA = LoadRenderTexture(width, height);
    state->giResultB = LoadRenderTexture(width, height);
    SetTextureFilter(state->giResultA.texture, TEXTURE_FILTER_POINT);
    SetTextureFilter(state->giResultB.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(state->giResultA.texture, TEXTURE_WRAP_CLAMP);
    SetTextureWrap(state->giResultB.texture, TEXTURE_WRAP_CLAMP);


    // TODO: investigate the need for higher precision texture datatypes?
    state->seedTexture = LoadRenderTexture(width, height); // Use default format
    SetTextureFilter(state->seedTexture.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(state->seedTexture.texture, TEXTURE_WRAP_CLAMP);

    state->jfaTextureA = LoadRenderTexture(width, height); // Use default format
    state->jfaTextureB = LoadRenderTexture(width, height); // Use default format
    SetTextureFilter(state->jfaTextureA.texture, TEXTURE_FILTER_POINT);
    SetTextureFilter(state->jfaTextureB.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(state->jfaTextureA.texture, TEXTURE_WRAP_CLAMP);
    SetTextureWrap(state->jfaTextureB.texture, TEXTURE_WRAP_CLAMP);

    state->distanceFieldTexture = LoadRenderTexture(width, height); // Use default format
    SetTextureFilter(state->distanceFieldTexture.texture, TEXTURE_FILTER_POINT);
    SetTextureWrap(state->distanceFieldTexture.texture, TEXTURE_WRAP_CLAMP);


    // Clear initial render textures
    BeginTextureMode(state->drawSurface); ClearBackground(BLANK); EndTextureMode();
    BeginTextureMode(state->seedTexture); ClearBackground(BLANK); EndTextureMode();
    BeginTextureMode(state->jfaTextureA); ClearBackground(BLANK); EndTextureMode();
    BeginTextureMode(state->jfaTextureB); ClearBackground(BLANK); EndTextureMode();
    BeginTextureMode(state->distanceFieldTexture); ClearBackground(WHITE); EndTextureMode(); // Init distance with max
    BeginTextureMode(state->giResultA); ClearBackground(BLACK); EndTextureMode();
    BeginTextureMode(state->giResultB); ClearBackground(BLACK); EndTextureMode();
}

void UpdateApp(AppState *state) {
    state->currentMousePos = GetMousePosition();

    // Check if mouse is over the GUI area to prevent drawing behind UI
    bool mouseOverGui = false;
    int guiWidth = 200;
    int guiMargin = 10;
    if (CheckCollisionPointRec(state->currentMousePos,
        (Rectangle){(float)GetScreenWidth() - guiWidth - guiMargin, 0, (float)guiWidth + guiMargin, (float)GetScreenHeight()})) {
        mouseOverGui = true;
    }


    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && !mouseOverGui) {
        state->isDrawing = true;
        state->lastMousePos = state->currentMousePos;
        state->mouseMovedSinceClick = false;
        // Reset temporal accumulation when drawing starts
        state->time = 0.0f;
        state->giPingPong = false; // Ensure we render fresh GI next frame
    }

    if (state->isDrawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
         if (Vector2DistanceSqr(state->currentMousePos, state->lastMousePos) > 0.1f) { // Check if mouse moved significantly
             state->mouseMovedSinceClick = true;
         }
        // Drawing happens in the DrawApp function using shaders
    }

    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
        if (state->isDrawing) { // Only process release if we were drawing
            state->isDrawing = false;
            // If the mouse didn't move, the drawing shader pass will handle the single point draw on release frame
        }
        // Reset temporal accumulation timer after drawing stops
        state->time = 0.0f;
    }

    // Update time for temporal effects if not drawing
    if (!state->isDrawing && state->useTemporalAccum) {
        // Increment time, potentially capping it or using frame time
        state->time += GetFrameTime() * 60.0f; // Example: increment based on frame time
    } else if (!state->useTemporalAccum) {
         state->time = 0.0f; // Reset time if temporal accum is off
    }

    // Update last mouse position for the next frame's drawing segment
    // This happens *after* checking IsMouseButtonReleased
    // Update lastMousePos only if drawing continues into the next frame
    // if (state->isDrawing) {
    //    state->lastMousePos = state->currentMousePos; // This logic is now handled in DrawApp
    // }
}


void DrawApp(AppState *state, int width, int height) {
    
    // --- Update Shader Uniforms ---
    Vector2 resolution = {(float)state->drawSurface.texture.width, (float)state->drawSurface.texture.height};
    Vector2 oneOverResolution = {1.0f / resolution.x, 1.0f / resolution.y};
    float radiusSquared = state->drawRadius * state->drawRadius;
    // Convert Color to vec3 for shader (normalize)
    Vector3 drawColorVec = {(float)state->drawColor.r / 255.0f, (float)state->drawColor.g / 255.0f, (float)state->drawColor.b / 255.0f};


    // --- 1. Drawing Pass ---

    // Only perform draw pass if mouse was pressed or is being held down
    // Also draw one last segment on release if mouse didn't move (single click)
    bool performDrawPass = state->isDrawing || (IsMouseButtonReleased(MOUSE_LEFT_BUTTON) && !state->mouseMovedSinceClick);

    if (performDrawPass) {
        BeginTextureMode(state->drawSurface);
        // Don't clear, accumulate drawing
        BeginShaderMode(state->drawShader);
            // Set uniforms for drawing shader
            SetShaderValue(state->drawShader, state->draw_loc_resolution, &resolution, SHADER_UNIFORM_VEC2);
            SetShaderValue(state->drawShader, state->draw_loc_fromPos, &state->lastMousePos, SHADER_UNIFORM_VEC2);
            SetShaderValue(state->drawShader, state->draw_loc_toPos, &state->currentMousePos, SHADER_UNIFORM_VEC2);
            SetShaderValue(state->drawShader, state->draw_loc_radiusSquared, &radiusSquared, SHADER_UNIFORM_FLOAT);
            SetShaderValue(state->drawShader, state->draw_loc_drawColor, &drawColorVec, SHADER_UNIFORM_VEC3);
            int isDrawingShaderFlag = 1; // Tell shader to draw this segment
            SetShaderValue(state->drawShader, state->draw_loc_isDrawing, &isDrawingShaderFlag, SHADER_UNIFORM_INT);
            // Bind the input texture (previous frame's drawing)
            SetShaderValueTexture(state->drawShader, state->draw_loc_inputTexture, state->drawSurface.texture);

            DrawQuad(state); // Draw fullscreen quad to apply shader

        EndShaderMode();
        EndTextureMode();

        // Update lastMousePos *after* using it in the shader for the current segment
        // Only update if drawing continues
         if (state->isDrawing) state->lastMousePos = state->currentMousePos;

    }


    // --- 2. Seed Pass ---
    BeginTextureMode(state->seedTexture);
    ClearBackground(BLANK); // Clear seed texture before drawing
    BeginShaderMode(state->seedShader);
        SetShaderValueTexture(state->seedShader, state->seed_loc_surfaceTexture, state->drawSurface.texture);
        DrawQuad(state);
    EndShaderMode();
    EndTextureMode();

    // --- 3. JFA Pass ---
    Texture2D currentJfaInput = state->seedTexture.texture;
    state->jfaPingPong = false; // Start writing to A

    for (int i = 0; i < state->jfaPassesNeeded; ++i) {
        RenderTexture2D *jfaOutputTarget = state->jfaPingPong ? &state->jfaTextureA : &state->jfaTextureB;
        float stepOffset = powf(2.0f, (float)(state->jfaPassesNeeded - 1 - i));

        BeginTextureMode(*jfaOutputTarget);
        ClearBackground(BLANK); // Clear target before drawing
        BeginShaderMode(state->jfaShader);
            SetShaderValueTexture(state->jfaShader, state->jfa_loc_inputTexture, currentJfaInput);
            SetShaderValue(state->jfaShader, state->jfa_loc_oneOverResolution, &oneOverResolution, SHADER_UNIFORM_VEC2);
            SetShaderValue(state->jfaShader, state->jfa_loc_stepOffset, &stepOffset, SHADER_UNIFORM_FLOAT);
            DrawQuad(state);
        EndShaderMode();
        EndTextureMode();

        currentJfaInput = jfaOutputTarget->texture; // Output becomes next input
        state->jfaPingPong = !state->jfaPingPong;   // Swap buffers
    }
    // Final JFA result is in 'currentJfaInput'

    // --- 4. Distance Field Pass ---
    BeginTextureMode(state->distanceFieldTexture);
    ClearBackground(WHITE); // Clear with max distance color (white)
    BeginShaderMode(state->distanceFieldShader);
        SetShaderValueTexture(state->distanceFieldShader, state->df_loc_jfaTexture, currentJfaInput);
        // SetShaderValue(state->distanceFieldShader, state->df_loc_resolution, &resolution, SHADER_UNIFORM_VEC2); // Optional
        DrawQuad(state);
    EndShaderMode();
    EndTextureMode();


    // --- 5. GI Raymarch Pass ---
    RenderTexture2D *giInputTarget = state->giPingPong ? &state->giResultB : &state->giResultA;
    RenderTexture2D *giOutputTarget = state->giPingPong ? &state->giResultA : &state->giResultB;

    BeginTextureMode(*giOutputTarget);
    ClearBackground(BLACK); // Clear before drawing GI
    BeginShaderMode(state->giShader);
        // Set Textures
        SetShaderValueTexture(state->giShader, state->gi_loc_sceneTexture, state->drawSurface.texture); // Use latest drawing output
        SetShaderValueTexture(state->giShader, state->gi_loc_distanceTexture, state->distanceFieldTexture.texture);
        SetShaderValueTexture(state->giShader, state->gi_loc_lastFrameTexture, giInputTarget->texture); // Previous GI frame

        // Set Uniforms
        SetShaderValue(state->giShader, state->gi_loc_oneOverResolution, &oneOverResolution, SHADER_UNIFORM_VEC2);
        SetShaderValue(state->giShader, state->gi_loc_rayCount, &state->rayCount, SHADER_UNIFORM_INT);
        SetShaderValue(state->giShader, state->gi_loc_maxSteps, &state->maxSteps, SHADER_UNIFORM_INT);
        int showNoiseInt = state->showNoise ? 1 : 0;
        SetShaderValue(state->giShader, state->gi_loc_showNoise, &showNoiseInt, SHADER_UNIFORM_INT);
         int showGrainInt = state->showGrain ? 1 : 0;
        SetShaderValue(state->giShader, state->gi_loc_showGrain, &showGrainInt, SHADER_UNIFORM_INT);
        int useTemporalInt = state->useTemporalAccum ? 1 : 0;
        SetShaderValue(state->giShader, state->gi_loc_useTemporalAccum, &useTemporalInt, SHADER_UNIFORM_INT);
         int enableSunInt = state->enableSun ? 1 : 0;
        SetShaderValue(state->giShader, state->gi_loc_enableSun, &enableSunInt, SHADER_UNIFORM_INT);
        SetShaderValue(state->giShader, state->gi_loc_time, &state->time, SHADER_UNIFORM_FLOAT);
        SetShaderValue(state->giShader, state->gi_loc_sunAngle, &state->sunAngle, SHADER_UNIFORM_FLOAT);

        DrawQuad(state);
    EndShaderMode();
    EndTextureMode();
    state->giPingPong = !state->giPingPong; // Swap buffers for next frame's temporal input


    // --- Final Draw to Screen ---
    BeginDrawing();
    ClearBackground(PINK); // Clear screen background


    // Draw the final result to the screen

    constexpr int nViews = 5;
    Texture debugViews[nViews] {
        giOutputTarget->texture,
        state->drawSurface.texture,
        state->seedTexture.texture,
        currentJfaInput,
        state->distanceFieldTexture.texture,
    };
    if (IsKeyPressed(KEY_RIGHT))
        state->debugView = (state->debugView + 1) % nViews;
    if (IsKeyPressed(KEY_LEFT))
        state->debugView = (state->debugView - 1 + nViews) % nViews;

    DrawTexturePro(debugViews[state->debugView],
                   (Rectangle){ 0.0f, 0.0f, (float)giOutputTarget->texture.width, (float)giOutputTarget->texture.height }, // Source Rectangle (no Y flip, handled in the vertex shader)
                   (Rectangle){ 0.0f, 0.0f, (float)width, (float)height }, // Destination Rectangle (stretch to screen)
                   (Vector2){ 0, 0 }, 0.0f, WHITE); // Offset, rotation, vertex colors (unused)

    // --- Draw UI Controls ---
    int currentY = 10;
    int guiMargin = 10;
    int guiHeight = 20;
    int guiWidth = 200;
    int sliderWidth = 120;
    Rectangle guiArea = {(float)width - guiWidth - guiMargin, 0, (float)guiWidth + guiMargin, (float)height}; // Area for UI
    //DrawRectangleRec(guiArea, ColorAlpha(GRAY, 0.5f)); // Add a background to signal drawing here won't work to the user

    GuiCheckBox((Rectangle){guiArea.x, (float)currentY, (float)guiHeight, (float)guiHeight}, "Enable Noise", &state->showNoise);
    currentY += guiHeight + guiMargin / 2;
    GuiCheckBox((Rectangle){guiArea.x, (float)currentY, (float)guiHeight, (float)guiHeight}, "Enable Grain", &state->showGrain);
    currentY += guiHeight + guiMargin / 2;
    GuiCheckBox((Rectangle){guiArea.x, (float)currentY, (float)guiHeight, (float)guiHeight}, "Temporal Accum", &state->useTemporalAccum);
    currentY += guiHeight + guiMargin / 2;
    GuiCheckBox((Rectangle){guiArea.x, (float)currentY, (float)guiHeight, (float)guiHeight}, "Enable Sun", &state->enableSun);
    currentY += guiHeight + guiMargin;

    // Use temporary float for maxSteps slider, then cast back to int
    state->maxStepsFloat = (float)state->maxSteps; // Update float value from int state
    GuiSliderBar((Rectangle){guiArea.x, (float)currentY, (float)sliderWidth, (float)guiHeight}, "Max Steps", TextFormat("%d", state->maxSteps), &state->maxStepsFloat, 1, MAX_RAYMARCH_STEPS);
    state->maxSteps = (int)state->maxStepsFloat; // Update int state from float slider value
    currentY += guiHeight + guiMargin;

    // Pass address of sunAngle directly
    GuiSliderBar((Rectangle){guiArea.x, (float)currentY, (float)sliderWidth, (float)guiHeight}, "Sun Angle", TextFormat("%.2f", state->sunAngle), &state->sunAngle, 0.0f, TAU);
    currentY += guiHeight + guiMargin;

    // Simple Color Picker Buttons
    if (GuiButton((Rectangle){guiArea.x, (float)currentY, (float)guiHeight, (float)guiHeight}, "")) state->drawColor = MAROON;
    DrawRectangle(guiArea.x + 1, currentY + 1, guiHeight - 2, guiHeight - 2, MAROON);
    if (GuiButton((Rectangle){guiArea.x + guiHeight + 2, (float)currentY, (float)guiHeight, (float)guiHeight}, "")) state->drawColor = ORANGE;
     DrawRectangle(guiArea.x + guiHeight + 2 + 1, currentY + 1, guiHeight - 2, guiHeight - 2, ORANGE);
    if (GuiButton((Rectangle){guiArea.x + 2*(guiHeight + 2), (float)currentY, (float)guiHeight, (float)guiHeight}, "")) state->drawColor = YELLOW;
     DrawRectangle(guiArea.x + 2*(guiHeight + 2) + 1, currentY + 1, guiHeight - 2, guiHeight - 2, YELLOW);
    if (GuiButton((Rectangle){guiArea.x + 3*(guiHeight + 2), (float)currentY, (float)guiHeight, (float)guiHeight}, "")) state->drawColor = BLACK;
    DrawRectangle(guiArea.x + 3*(guiHeight + 2) + 1, currentY + 1, guiHeight - 2, guiHeight - 2, BLACK);


    DrawFPS(10, 10); // Show FPS
    EndDrawing();
}

void DeInitApp(AppState *state) {
    // Unload Textures
    UnloadRenderTexture(state->drawSurface);
    UnloadRenderTexture(state->seedTexture);
    UnloadRenderTexture(state->jfaTextureA);
    UnloadRenderTexture(state->jfaTextureB);
    UnloadRenderTexture(state->distanceFieldTexture);
    UnloadRenderTexture(state->giResultA);
    UnloadRenderTexture(state->giResultB);

    // Unload Shaders
    UnloadShader(state->drawShader);
    UnloadShader(state->seedShader);
    UnloadShader(state->jfaShader);
    UnloadShader(state->distanceFieldShader);
    UnloadShader(state->giShader);
}
