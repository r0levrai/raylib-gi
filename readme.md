Dynamic 2D global illumination with OpenGL, Raylib and C++. Adapted from the excellent interactive article at https://jason.today/gi
# Screenshot
![example screenshot](<screenshot.png>)
# Controls
`Left and Right Arrows`: cycle view through intermediate buffers (final gi > paint surface > uv seed > jfa > distance field > final gi)
# Feedback
Despite having experience in gamedev and graphics I'm still very new to OpenGL and C++. Any comment, fixes or upgrades would be greatly appreciated :)
# Ideas for potential upgrades
1. Decouple the input/output resolution from the distance field/gi resolution
2. Track down the difference in noise/grain from https://jason.today/gi
3. Time the different passes
4. Expose a slider for the number of rays and a button to show/hide UI
5. Expose a function to call it as a postprocessing effect in any 2D raylib scene (with an existing i/o render texture, or the screen if left empty)
6. Use as a base for radiance cascade, a more structured and efficient way of raymarching the distance field. That could be a fork
# License
For consistency with the raylib examples, this code is licensed under an unmodified zlib/libpng license, which is an OSI-certified, BSD-like license that allows static linking with closed source software.